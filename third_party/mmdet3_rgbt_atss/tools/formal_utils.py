from __future__ import annotations

import copy
import csv
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig_mmdet3")))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONFIG_PATHS = {
    "real_only": ROOT / "configs" / "dualstream_atss_r50_fpn_1x_coevonet_real_only.py",
    "phasec_synth0p25x": ROOT / "configs" / "dualstream_atss_r50_fpn_1x_coevonet_phasec_synth0p25x.py",
    "phasec_density_synth0p25x": ROOT / "configs" / "dualstream_atss_r50_fpn_1x_coevonet_phasec_density_synth0p25x.py",
}

SETTING_LABELS = {
    "dualstream_atss_r50_fpn_1x_coevonet_real_only": "Real-only",
    "dualstream_atss_r50_fpn_1x_coevonet_phasec_synth0p25x": "Real + Phase-C 0.25x",
    "dualstream_atss_r50_fpn_1x_coevonet_phasec_density_synth0p25x": "Real + Phase-C density 0.25x",
}

METRIC_KEY_MAP = {
    "AP": ("bbox_mAP", "coco/bbox_mAP"),
    "AP50": ("bbox_mAP_50", "coco/bbox_mAP_50"),
    "AP75": ("bbox_mAP_75", "coco/bbox_mAP_75"),
    "AP_small": ("bbox_mAP_s", "coco/bbox_mAP_s"),
}


def bootstrap_runtime() -> None:
    from rgbt_atss_runtime import bootstrap_rgbt_atss_runtime

    bootstrap_rgbt_atss_runtime()


def config_to_setting_label(config_name: str) -> str:
    stem = Path(config_name).stem
    return SETTING_LABELS.get(stem, stem)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def read_json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def reduce_loss_dict(losses: dict) -> tuple[dict, torch.Tensor]:
    reduced = {}
    total = None
    for key, value in losses.items():
        if isinstance(value, (list, tuple)):
            scalar = sum(v.mean() for v in value)
        else:
            scalar = value.mean()
        reduced[key] = float(scalar.detach().cpu())
        total = scalar if total is None else total + scalar
    return reduced, total


def resolve_pipeline(dataset_cfg):
    current = dataset_cfg
    while True:
        if hasattr(current, "pipeline"):
            return copy.deepcopy(current.pipeline)
        if hasattr(current, "dataset"):
            current = current.dataset
            continue
        if hasattr(current, "datasets") and len(current.datasets) > 0:
            current = current.datasets[0]
            continue
        raise AttributeError("Unable to resolve pipeline from train dataset config.")


def build_val_loss_dataloader(cfg: Config):
    val_loader_cfg = copy.deepcopy(cfg.val_dataloader)
    dataset_cfg = val_loader_cfg.dataset
    dataset_cfg.test_mode = False
    dataset_cfg.pipeline = resolve_pipeline(cfg.train_dataloader.dataset)
    return Runner.build_dataloader(val_loader_cfg)


def normalize_eval_metrics(metrics: dict | None) -> dict:
    metrics = metrics or {}
    out = {}
    for out_key, candidates in METRIC_KEY_MAP.items():
        value = 0.0
        for key in candidates:
            if key in metrics:
                value = float(metrics[key])
                break
        out[out_key] = value
    return out


def safe_number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def format_mean_std(values: Iterable[float]) -> str:
    mean, std = mean_std(values)
    return f"{mean:.4f} ± {std:.4f}"


def infer_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_train_rows_from_scalars(scalars_rows: list[dict]) -> list[dict]:
    out = []
    for row in scalars_rows:
        if "loss" in row or "loss_cls" in row or "lr" in row:
            out.append(dict(row))
    return out


def extract_eval_rows_from_scalars(scalars_rows: list[dict]) -> list[dict]:
    out = []
    for row in scalars_rows:
        if any(k in row for k in ("bbox_mAP", "coco/bbox_mAP", "bbox_mAP_50", "coco/bbox_mAP_50")):
            normalized = normalize_eval_metrics(row)
            out.append({**dict(row), **normalized})
    return out
