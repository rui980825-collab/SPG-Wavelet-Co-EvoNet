# coevonet/data/patch_bank.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Tuple, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _imread_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _imread_tir(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _resize(img: np.ndarray, size: int, is_rgb: bool) -> np.ndarray:
    if is_rgb:
        im = Image.fromarray(img, mode="RGB").resize((size, size), resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)
    im = Image.fromarray(img, mode="L").resize((size, size), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


class PatchBankDataset(Dataset):
    """
    Step1 产物：
      patch_bank/
        rgb/<cls>/*.png
        tir/<cls>/*.png

    配对规则：rgb 相对路径映射到 tir 下同名文件。

    新增（为快速验证/推理服务）：
      - max_pairs: 只加载前 N 对样本，避免 rglob 扫全库导致 infer 卡死
      - class_filter: 只从指定类别子目录加载（例如 ["ship"]），仍然传 patch_bank_root 根目录
      - seed: 当 class_filter 生效时，可以做“随机抽样”而不是只取字典序前 N 个（可选）
    """
    def __init__(
        self,
        patch_bank_root: Path,
        img_size: int = 64,
        max_pairs: Optional[int] = None,
        class_filter: Optional[Sequence[str]] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.root = Path(patch_bank_root)
        self.img_size = int(img_size)

        self.rgb_root = self.root / "rgb"
        self.tir_root = self.root / "tir"

        if not self.rgb_root.exists():
            raise FileNotFoundError(f"RGB root not found: {self.rgb_root}")
        if not self.tir_root.exists():
            raise FileNotFoundError(f"TIR root not found: {self.tir_root}")

        self.max_pairs = None if max_pairs is None else int(max_pairs)
        self.class_filter = None if class_filter is None else [str(c) for c in class_filter]
        self.seed = int(seed)

        # --- collect rgb files ---
        # 默认（训练）行为：全量扫描
        # 推理/快速验证：可以通过 max_pairs / class_filter 抽样
        rgb_files: List[Path] = []

        if self.class_filter is None or len(self.class_filter) == 0:
            # 全库：rglob 仍然会扫描很多文件，所以只有在 max_pairs 很小的时候才建议用于推理
            # 这里仍然保留原逻辑，以保证 Step2 训练不受影响
            rgb_files = sorted(self.rgb_root.rglob("*.png"))
        else:
            # 只扫描指定类目录：显著减少文件枚举量
            for cls in self.class_filter:
                cls_dir = self.rgb_root / cls
                if not cls_dir.exists():
                    raise FileNotFoundError(f"class dir not found under rgb/: {cls_dir}")
                rgb_files.extend(sorted(cls_dir.rglob("*.png")))

        if len(rgb_files) == 0:
            raise RuntimeError(f"No RGB patches found under: {self.rgb_root} (class_filter={self.class_filter})")

        # 如果 max_pairs 很小且文件很多，建议随机抽样，而不是拿字典序前 N 个（避免偏置）
        if self.max_pairs is not None and len(rgb_files) > self.max_pairs:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(rgb_files), size=self.max_pairs, replace=False)
            rgb_files = [rgb_files[i] for i in idx.tolist()]
            # 为了可复现，排序一下（固定输出顺序）
            rgb_files = sorted(rgb_files)

        # --- pair rgb/tir ---
        pairs: List[Tuple[Path, Path]] = []
        for rp in rgb_files:
            rel = rp.relative_to(self.rgb_root)
            tp = self.tir_root / rel
            if tp.exists():
                pairs.append((rp, tp))

        if len(pairs) == 0:
            raise RuntimeError("No paired RGB/TIR patches found. Check your patch_bank output.")

        self.pairs = pairs

        if verbose:
            extra = []
            if self.max_pairs is not None:
                extra.append(f"max_pairs={self.max_pairs}")
            if self.class_filter is not None:
                extra.append(f"class_filter={self.class_filter}")
            extra_s = (" " + " ".join(extra)) if extra else ""
            print(f"[PatchBankDataset] pairs={len(self.pairs)} img_size={self.img_size} root={self.root}{extra_s}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rgb_path, tir_path = self.pairs[idx]

        rgb = _imread_rgb(rgb_path)     # H,W,3
        tir = _imread_tir(tir_path)     # H,W

        rgb = _resize(rgb, self.img_size, is_rgb=True)
        tir = _resize(tir, self.img_size, is_rgb=False)

        rgb = (rgb.astype(np.float32) / 255.0)
        tir = (tir.astype(np.float32) / 255.0)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)   # 3,H,W
        tir_t = torch.from_numpy(tir).unsqueeze(0)       # 1,H,W
        return rgb_t, tir_t
