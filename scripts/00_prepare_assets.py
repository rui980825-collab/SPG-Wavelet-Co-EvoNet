import os
import cv2
import json
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import csv

# ================= 配置区域 =================
# 自动获取项目根目录 (假设脚本在 scripts/ 目录下)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # 指向 .

# 重新定义所有的路径为绝对路径
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw" / "dataset" / "images"
OUTPUT_ROOT = DATA_ROOT / "patches"

# 阈值设置
MIN_PATCH_SIZE = 4  # 小于4像素的垃圾丢弃
BG_CROP_SIZE = 256  # 背景切片大小（优化为256，以便采样到更多背景）
BG_PER_IMG = 3  # 每张原图最多挖几个背景
EXCLUSION_MARGIN = 5  # 禁区外扩像素


# ===========================================

def setup_dirs():
    """创建输出目录结构"""
    dirs = [
        OUTPUT_ROOT / "patch_bank" / "rgb",
        OUTPUT_ROOT / "patch_bank" / "tir",
        OUTPUT_ROOT / "background_pool" / "rgb",
        OUTPUT_ROOT / "background_pool" / "tir",
        OUTPUT_ROOT / "stats"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def classify_size(area):
    """增加 Tiny 分类，更适合红外小目标"""
    if area < 16 ** 2:
        return "tiny"  # 小于 16x16 像素的目标
    elif area < 32 ** 2:
        return "small"
    elif area < 96 ** 2:
        return "medium"
    else:
        return "large"


def extract_backgrounds(img_rgb, img_tir, anns, flat_name):
    """滑动窗口负采样提取纯净背景"""
    h, w = img_rgb.shape[:2]
    exclusion_mask = np.zeros((h, w), dtype=np.uint8)

    # 1. 构建禁区 (Exclusion Zone)
    for ann in anns:
        bx, by, bw, bh = map(int, ann['bbox'])
        x1, y1 = max(0, bx - EXCLUSION_MARGIN), max(0, by - EXCLUSION_MARGIN)
        x2, y2 = min(w, bx + bw + EXCLUSION_MARGIN), min(h, by + bh + EXCLUSION_MARGIN)
        exclusion_mask[y1:y2, x1:x2] = 255

    bg_samples = []
    attempts = 0
    # 2. 尝试采样
    while len(bg_samples) < BG_PER_IMG and attempts < 50:
        attempts += 1
        if h < BG_CROP_SIZE or w < BG_CROP_SIZE: break
        top = np.random.randint(0, h - BG_CROP_SIZE)
        left = np.random.randint(0, w - BG_CROP_SIZE)

        # 检查禁区交集
        if np.sum(exclusion_mask[top:top + BG_CROP_SIZE, left:left + BG_CROP_SIZE]) == 0:
            crop_rgb = img_rgb[top:top + BG_CROP_SIZE, left:left + BG_CROP_SIZE]
            crop_tir = img_tir[top:top + BG_CROP_SIZE, left:left + BG_CROP_SIZE]
            # 信息量过滤 (简单的方差检查，防止纯黑图)
            if np.var(crop_rgb) > 50 and np.var(crop_tir) > 50:
                save_name = f"{flat_name}_bg_{len(bg_samples)}.png"
                bg_samples.append((save_name, crop_rgb, crop_tir))
    return bg_samples


def main():
    setup_dirs()

    # 标注文件路径 (请确保路径正确)
    ann_file_rgb = DATA_ROOT / "raw" / "dataset" / "annotations_coco" / "instances_00_train2017.json"
    ann_file_tir = DATA_ROOT / "raw" / "dataset" / "annotations_coco" / "instances_01_train2017.json"

    if not ann_file_rgb.exists() or not ann_file_tir.exists():
        print(f"❌ 错误: 找不到标注文件 {ann_file_rgb} 或 {ann_file_tir}")
        return

    with open(ann_file_rgb, 'r') as f:
        coco_rgb = json.load(f)

    with open(ann_file_tir, 'r') as f:
        coco_tir = json.load(f)

    # 建立索引
    img_dict_rgb = {img['id']: img for img in coco_rgb['images']}
    img_dict_tir = {img['id']: img for img in coco_tir['images']}

    img_to_anns_rgb = {}
    for ann in coco_rgb['annotations']:
        img_id = ann['image_id']
        img_to_anns_rgb.setdefault(img_id, []).append(ann)

    img_to_anns_tir = {}
    for ann in coco_tir['annotations']:
        img_id = ann['image_id']
        img_to_anns_tir.setdefault(img_id, []).append(ann)

    categories = {cat['id']: cat['name'] for cat in coco_rgb['categories']}
    class_stats = {name: {'count': 0, 'tiny': 0, 'small': 0, 'medium': 0, 'large': 0} for name in categories.values()}

    missing_pairs = []
    total_patches = 0
    total_bgs = 0

    print(f"🚀 开始处理 {len(img_dict_rgb)} 张图像对...")

    for img_id, img_info in tqdm(img_dict_rgb.items()):
        # file_name 为 "DJI_0032_3/00/00060.jpg"
        file_name = img_info['file_name']

        # 1. 构造 RGB 绝对路径
        rgb_path = RAW_ROOT / file_name

        # 2. 构造 TIR 绝对路径
        # 逻辑：将路径中的 /00/ 替换为 /01/
        tir_path = Path(str(rgb_path).replace("/00/", "/01/").replace("\\00\\", "\\01\\"))

        # 3. 检查配对是否存在
        if not rgb_path.exists() or not tir_path.exists():
            missing_pairs.append(file_name)
            continue

        img_rgb = cv2.imread(str(rgb_path))
        img_tir = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)

        if img_rgb is None or img_tir is None:
            missing_pairs.append(file_name)
            continue

        # 扁平化名称，用于保存 (例如 DJI_0032_3_00_00060)
        flat_name = file_name.replace("/", "_").replace("\\", "_")
        anns_rgb = img_to_anns_rgb.get(img_id, [])
        anns_tir = img_to_anns_tir.get(img_id, [])

        # 提取背景
        bgs = extract_backgrounds(img_rgb, img_tir, anns_rgb, flat_name)
        for bg_name, b_rgb, b_tir in bgs:
            cv2.imwrite(str(OUTPUT_ROOT / "background_pool/rgb" / bg_name), b_rgb)
            cv2.imwrite(str(OUTPUT_ROOT / "background_pool/tir" / bg_name), b_tir)
            total_bgs += 1

        # 提取前景
        for i, ann in enumerate(anns_rgb):
            cat_name = categories[ann['category_id']]
            x, y, w, h = map(int, ann['bbox'])
            if w < MIN_PATCH_SIZE or h < MIN_PATCH_SIZE: continue

            x, y = max(0, x), max(0, y)
            patch_rgb = img_rgb[y:y + h, x:x + w]
            patch_tir = img_tir[y:y + h, x:x + w]

            if patch_rgb.size == 0 or patch_tir.size == 0: continue

            save_name = f"{cat_name}_{flat_name}_{i}.png"
            cat_dir_rgb = OUTPUT_ROOT / "patch_bank/rgb" / cat_name
            cat_dir_tir = OUTPUT_ROOT / "patch_bank/tir" / cat_name
            cat_dir_rgb.mkdir(exist_ok=True);
            cat_dir_tir.mkdir(exist_ok=True)

            cv2.imwrite(str(cat_dir_rgb / save_name), patch_rgb)
            cv2.imwrite(str(cat_dir_tir / save_name), patch_tir)

            size_cls = classify_size(w * h)
            class_stats[cat_name]['count'] += 1
            class_stats[cat_name][size_cls] += 1
            total_patches += 1

    # 导出统计
    save_stats_csv(class_stats)
    if missing_pairs:
        with open(OUTPUT_ROOT / "stats" / "missing_pairs.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for m in missing_pairs: writer.writerow([m])

    print(f"\n✅ 完成！提取背景: {total_bgs}, 目标: {total_patches}")


def save_stats_csv(class_stats):
    with open(OUTPUT_ROOT / "stats" / "class_stats.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Total", "Tiny", "Small", "Medium", "Large"])
        for cls, s in class_stats.items():
            writer.writerow([cls, s['count'], s['tiny'], s['small'], s['medium'], s['large']])


if __name__ == "__main__":
    main()
