from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from formal_utils import (
    bootstrap_runtime,
    config_to_setting_label,
    infer_device,
    normalize_eval_metrics,
    set_seed,
)

from mmdet.registry import METRICS, MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AP_tiny/AP_et/AR_tiny/AR_et for dual-stream ATSS runs"
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--work-dir", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def patch_cfg(cfg: Config, args) -> Config:
    cfg = cfg.copy()
    cfg.work_dir = args.work_dir
    cfg.launcher = "none"
    # Avoid re-loading ImageNet weights before the checkpoint restore.
    if "backbone" in cfg.model and "pretrained_path" in cfg.model.backbone:
        cfg.model.backbone.pretrained_path = None
    return cfg


def load_model_checkpoint(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    model.load_state_dict(state_dict, strict=True)


def resolve_bbox_json(pred_dir: Path) -> Path:
    candidates = sorted(pred_dir.rglob("*.bbox.json"))
    if not candidates:
        raise FileNotFoundError(f"No bbox json found under {pred_dir}")
    return candidates[-1]


def run_eval_and_export(cfg: Config, model, work_dir: Path, max_eval_batches: int | None) -> tuple[dict, Path]:
    eval_loader = Runner.build_dataloader(cfg.val_dataloader)
    pred_dir = work_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    outfile_prefix = str((pred_dir / "results").resolve())
    metric = METRICS.build(dict(cfg.val_evaluator, outfile_prefix=outfile_prefix))
    metric.dataset_meta = getattr(eval_loader.dataset, "metainfo", {})

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_loader, start=1):
            if max_eval_batches is not None and step > max_eval_batches:
                break
            processed = model.data_preprocessor(batch, training=False)
            preds = model(processed["inputs"], processed["data_samples"], mode="predict")
            metric.process(batch, [pred.to_dict() for pred in preds])

    standard_metrics = normalize_eval_metrics(metric.compute_metrics(metric.results))
    bbox_json = resolve_bbox_json(pred_dir)
    return standard_metrics, bbox_json


def eval_area_range(ann_file: str, pred_json: Path, area_max: float) -> dict:
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(str(pred_json))
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.params.areaRng = [[0.0, float(area_max)]]
    evaluator.params.areaRngLbl = ["all"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR": float(evaluator.stats[8]),
    }


def count_gt_upto_area(ann_file: str, area_max: float) -> int:
    ann = json.loads(Path(ann_file).read_text(encoding="utf-8"))
    count = 0
    for item in ann.get("annotations", []):
        bbox = item.get("bbox", None)
        if not bbox or len(bbox) < 4:
            continue
        area = float(bbox[2]) * float(bbox[3])
        if area < area_max:
            count += 1
    return count


def main():
    args = parse_args()
    bootstrap_runtime()
    set_seed(args.seed)
    device = infer_device(args.device)

    cfg = patch_cfg(Config.fromfile(args.config), args)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model = MODELS.build(cfg.model).to(device)
    load_model_checkpoint(model, Path(args.checkpoint))

    standard_metrics, pred_json = run_eval_and_export(cfg, model, work_dir, args.max_eval_batches)

    ann_file = cfg.val_evaluator["ann_file"] if isinstance(cfg.val_evaluator, dict) else cfg.val_evaluator.ann_file
    tiny_metrics = eval_area_range(ann_file, pred_json, area_max=16.0 * 16.0)
    et_metrics = eval_area_range(ann_file, pred_json, area_max=8.0 * 8.0)

    summary = {
        "setting": config_to_setting_label(args.config),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "work_dir": str(work_dir.resolve()),
        "seed": args.seed,
        "prediction_json": str(pred_json.resolve()),
        "standard_AP": standard_metrics["AP"],
        "standard_AP50": standard_metrics["AP50"],
        "standard_AP75": standard_metrics["AP75"],
        "standard_AP_small": standard_metrics["AP_small"],
        "AP_tiny": tiny_metrics["AP"],
        "AP50_tiny": tiny_metrics["AP50"],
        "AP75_tiny": tiny_metrics["AP75"],
        "AR_tiny": tiny_metrics["AR"],
        "AP_et": et_metrics["AP"],
        "AP50_et": et_metrics["AP50"],
        "AP75_et": et_metrics["AP75"],
        "AR_et": et_metrics["AR"],
        "num_gt_tiny": count_gt_upto_area(ann_file, area_max=16.0 * 16.0),
        "num_gt_et": count_gt_upto_area(ann_file, area_max=8.0 * 8.0),
    }

    summary_path = work_dir / "tiny_et_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("summary=", summary_path)
    print("prediction_json=", pred_json)
    print("AP_tiny=", summary["AP_tiny"])
    print("AR_tiny=", summary["AR_tiny"])
    print("AP_et=", summary["AP_et"])
    print("AR_et=", summary["AR_et"])


if __name__ == "__main__":
    main()
