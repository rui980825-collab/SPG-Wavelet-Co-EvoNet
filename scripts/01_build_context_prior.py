# scripts/01_build_context_prior.py
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import yaml

from coevonet.synthesis.constraints import scene_label_roi_8class, SCENES_8


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def read_rgb_from_raw(raw_root: Path, file_name: str):
    """
    file_name example: DJI_0022_1/00/00000.jpg
    raw_root example: data/raw/dataset/images
    """
    img_path = raw_root / file_name
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_coco(coco_path: Path):
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in coco.get("images", [])}
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns = coco.get("annotations", [])
    return imgs, cats, anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=str, required=True, help="instances_00_train2017.json (RGB side)")
    ap.add_argument("--raw_root", type=str, required=True, help="data/raw/dataset/images (the folder containing DJI_xxx/00/xxx.jpg)")
    ap.add_argument("--out_yaml", type=str, default="configs/context_rules.yaml")

    ap.add_argument("--context_ratio", type=float, default=2.5, help="ROI size relative to bbox (>=1)")
    ap.add_argument("--min_prob", type=float, default=0.12, help="min P(scene|class) to keep")
    ap.add_argument("--topk", type=int, default=3, help="keep at most top-k scenes per class")
    ap.add_argument("--max_samples_per_class", type=int, default=50000, help="cap samples per class for speed")

    ap.add_argument("--skip_unknown", action="store_true", help="do not count 'unknown' scene")
    args = ap.parse_args()

    coco_path = Path(args.coco)
    raw_root = Path(args.raw_root)
    out_yaml = Path(args.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO not found: {coco_path}")
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    imgs, cats, anns = load_coco(coco_path)

    # count[class][scene]
    counts = {cls: defaultdict(int) for cls in set(cats.values())}
    total_cls = defaultdict(int)

    cap = int(args.max_samples_per_class)
    used_cls = defaultdict(int)

    ratio = max(1.0, float(args.context_ratio))
    skip_unknown = bool(args.skip_unknown)

    missing_imgs = 0

    for a in anns:
        img_id = a.get("image_id")
        cat_id = a.get("category_id")
        if img_id not in imgs or cat_id not in cats:
            continue

        cls = cats[cat_id]
        if used_cls[cls] >= cap:
            continue

        file_name = imgs[img_id].get("file_name")
        if not file_name:
            continue

        rgb = read_rgb_from_raw(raw_root, file_name)
        if rgb is None:
            missing_imgs += 1
            continue

        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        x, y, w, h = float(x), float(y), float(w), float(h)
        if w < 2 or h < 2:
            continue

        H, W = rgb.shape[:2]

        # context crop around bbox
        cx = x + w * 0.5
        cy = y + h * 0.5
        ww = w * ratio
        hh = h * ratio

        x1 = clamp(int(round(cx - ww * 0.5)), 0, W - 1)
        y1 = clamp(int(round(cy - hh * 0.5)), 0, H - 1)
        x2 = clamp(int(round(cx + ww * 0.5)), 1, W)
        y2 = clamp(int(round(cy + hh * 0.5)), 1, H)

        if x2 <= x1 + 3 or y2 <= y1 + 3:
            continue

        roi = rgb[y1:y2, x1:x2]
        scene = scene_label_roi_8class(roi)
        if skip_unknown and scene == "unknown":
            continue
        if scene == "unknown":
            # 默认：unknown 不参与先验（保持规则干净）
            continue

        counts[cls][scene] += 1
        total_cls[cls] += 1
        used_cls[cls] += 1

    # build probs -> rules
    rules = {}
    stats = {}

    for cls, sc_dict in counts.items():
        denom = max(1, int(total_cls[cls]))

        # scene probs
        probs = []
        for sc in SCENES_8:
            p = float(sc_dict.get(sc, 0)) / float(denom)
            probs.append((sc, p, int(sc_dict.get(sc, 0))))

        probs.sort(key=lambda t: t[1], reverse=True)

        keep = [(sc, p, n) for (sc, p, n) in probs if p >= float(args.min_prob)]
        keep = keep[: int(args.topk)]
        rules[cls] = [sc for (sc, _p, _n) in keep] if keep else []

        stats[cls] = {
            "total_used": int(denom),
            "kept": [{"scene": sc, "prob": round(p, 4), "count": n} for (sc, p, n) in keep],
            "all_top8": [{"scene": sc, "prob": round(p, 4), "count": n} for (sc, p, n) in probs[:8]],
        }

    out = {
        "scenes": SCENES_8,
        "class_to_context": rules,
        "build_args": {
            "coco": str(coco_path),
            "raw_root": str(raw_root),
            "context_ratio": float(args.context_ratio),
            "min_prob": float(args.min_prob),
            "topk": int(args.topk),
            "max_samples_per_class": int(args.max_samples_per_class),
            "skip_unknown": bool(args.skip_unknown),
        },
        "meta": {
            "missing_images": int(missing_imgs),
            "note": "Rules learned from ROI-level scene heuristics on RGB train set.",
        },
        "stats": stats,
    }

    out_yaml.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"✅ wrote: {out_yaml}")
    print(f"   missing_images={missing_imgs}")
    print("   keys: scenes, class_to_context, stats")


if __name__ == "__main__":
    main()
