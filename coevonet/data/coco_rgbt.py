from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms.functional import pil_to_tensor


def _resolve_path(path_like: str | Path, project_root: Optional[Path] = None) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    if project_root is not None:
        return (project_root / p).resolve()
    return p.resolve()


def _xywh_to_xyxy(x: float, y: float, w: float, h: float, width: int, height: int) -> Optional[List[float]]:
    if w <= 0 or h <= 0:
        return None
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), x1 + float(w))
    y2 = min(float(height), y1 + float(h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


class RepeatDataset(Dataset):
    def __init__(self, dataset: Dataset, repeat: int):
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.dataset = dataset
        self.repeat = int(repeat)

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx: int):
        base_idx = int(idx) % len(self.dataset)
        return self.dataset[base_idx]


class CocoRGBTDataset(Dataset):
    """
    COCO-style dataset for paired RGB/TIR detection.
    - RGB annotations are read from ann_file.
    - TIR file path is inferred from RGB file_name:
      1) same relative file_name under tir_root
      2) if contains '/00/' or '\\00\\', replace with '/01/' variant.
    """

    def __init__(
        self,
        ann_file: str | Path,
        rgb_root: str | Path,
        tir_root: str | Path,
        project_root: Optional[Path] = None,
        filter_empty: bool = True,
        label_by_name: Optional[Dict[str, int]] = None,
    ):
        self.ann_file = _resolve_path(ann_file, project_root)
        self.rgb_root = _resolve_path(rgb_root, project_root)
        self.tir_root = _resolve_path(tir_root, project_root)
        self.filter_empty = bool(filter_empty)

        if not self.ann_file.exists():
            raise FileNotFoundError(f"ann_file not found: {self.ann_file}")
        if not self.rgb_root.exists():
            raise FileNotFoundError(f"rgb_root not found: {self.rgb_root}")
        if not self.tir_root.exists():
            raise FileNotFoundError(f"tir_root not found: {self.tir_root}")

        coco = json.loads(self.ann_file.read_text(encoding="utf-8"))
        self.images: List[Dict] = coco.get("images", [])
        self.annotations: List[Dict] = coco.get("annotations", [])
        self.categories: List[Dict] = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
        self.cat_names: List[str] = [str(c["name"]) for c in self.categories]

        if label_by_name is None:
            self.cat_id_to_label = {int(c["id"]): i + 1 for i, c in enumerate(self.categories)}
        else:
            self.cat_id_to_label = {}
            for c in self.categories:
                name = str(c["name"])
                if name not in label_by_name:
                    raise ValueError(f"category '{name}' missing from label_by_name in {self.ann_file}")
                self.cat_id_to_label[int(c["id"])] = int(label_by_name[name])
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}
        self.cat_signature = [(int(c["id"]), str(c["name"])) for c in self.categories]

        anns_by_img: Dict[int, List[Dict]] = defaultdict(list)
        for ann in self.annotations:
            anns_by_img[int(ann["image_id"])].append(ann)
        self._anns_by_img = anns_by_img

        kept = []
        for im in self.images:
            img_id = int(im["id"])
            if self.filter_empty and len(self._anns_by_img.get(img_id, [])) == 0:
                continue
            kept.append(im)
        self._items = kept

    def _resolve_tir_rel_path(self, file_name: str) -> str:
        rel = str(file_name).replace("\\", "/")
        cand = [rel]
        if "/00/" in rel:
            cand.append(rel.replace("/00/", "/01/"))
        for cr in cand:
            if (self.tir_root / cr).exists():
                return cr
        # last fallback: use converted path even if missing for clear error message
        if "/00/" in rel:
            return rel.replace("/00/", "/01/")
        return rel

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        im = self._items[idx]
        img_id = int(im["id"])
        width = int(im["width"])
        height = int(im["height"])
        rgb_rel = str(im["file_name"]).replace("\\", "/")
        tir_rel = self._resolve_tir_rel_path(rgb_rel)

        rgb_path = self.rgb_root / rgb_rel
        tir_path = self.tir_root / tir_rel
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not tir_path.exists():
            raise FileNotFoundError(f"TIR image not found: {tir_path}")

        rgb = Image.open(rgb_path).convert("RGB")
        tir = Image.open(tir_path).convert("L")

        rgb_t = pil_to_tensor(rgb).float() / 255.0  # [3,H,W]
        tir_t = pil_to_tensor(tir).float() / 255.0  # [1,H,W]
        image = torch.cat([rgb_t, tir_t], dim=0)    # [4,H,W]

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in self._anns_by_img.get(img_id, []):
            cat_id = int(ann["category_id"])
            if cat_id not in self.cat_id_to_label:
                continue
            b = ann.get("bbox", None)
            if b is None or len(b) != 4:
                continue
            xyxy = _xywh_to_xyxy(float(b[0]), float(b[1]), float(b[2]), float(b[3]), width, height)
            if xyxy is None:
                continue
            boxes.append(xyxy)
            labels.append(self.cat_id_to_label[cat_id])
            areas.append(float(ann.get("area", b[2] * b[3])))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }
        return image, target


def detection_collate_fn(batch):
    return tuple(zip(*batch))


def build_concat_train_dataset(
    dataset_cfgs: Sequence[Dict],
    project_root: Optional[Path] = None,
    filter_empty: bool = True,
) -> Tuple[Dataset, List[str]]:
    datasets: List[Dataset] = []
    name_lists: List[List[str]] = []
    names: List[str] = []
    label_by_name: Optional[Dict[str, int]] = None

    for idx, cfg in enumerate(dataset_cfgs):
        if not isinstance(cfg, dict):
            raise ValueError(f"dataset config must be mapping, got: {type(cfg)}")
        name = str(cfg.get("name", f"dataset{len(datasets)}"))
        ds = CocoRGBTDataset(
            ann_file=cfg["ann_file"],
            rgb_root=cfg["rgb_root"],
            tir_root=cfg["tir_root"],
            project_root=project_root,
            filter_empty=bool(cfg.get("filter_empty", filter_empty)),
            label_by_name=label_by_name,
        )
        if idx == 0:
            base_names = list(ds.cat_names)
            label_by_name = {n: i + 1 for i, n in enumerate(base_names)}
        else:
            if sorted(ds.cat_names) != sorted(base_names):
                raise ValueError(f"category name mismatch between datasets '{names[0]}' and '{name}'")

        repeat = int(cfg.get("repeat", 1))
        if repeat > 1:
            ds = RepeatDataset(ds, repeat=repeat)
        datasets.append(ds)
        if hasattr(ds, "cat_names"):
            name_lists.append(ds.cat_names)
        elif hasattr(ds, "dataset") and hasattr(ds.dataset, "cat_names"):
            name_lists.append(ds.dataset.cat_names)
        else:
            raise ValueError(f"dataset '{name}' does not expose cat_names")
        names.append(name)

    if len(datasets) == 0:
        raise ValueError("at least one train dataset is required")

    ref = name_lists[0]
    for i in range(1, len(name_lists)):
        if sorted(name_lists[i]) != sorted(ref):
            raise ValueError(f"category mismatch between datasets '{names[0]}' and '{names[i]}'")

    if len(datasets) == 1:
        return datasets[0], ref
    return ConcatDataset(datasets), ref
