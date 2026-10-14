from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig_mmdet3")))

from mmdet.registry import METRICS, MODELS
from mmdet.utils import register_all_modules

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny sanity training for dual-stream ATSS")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--work-dir", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-iters", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=5)
    parser.add_argument("--sanity-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def main():
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "eval").mkdir(parents=True, exist_ok=True)

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)

    train_loader = Runner.build_dataloader(cfg.train_dataloader)
    val_loader = Runner.build_dataloader(cfg.val_dataloader)
    val_loss_loader = build_val_loss_dataloader(cfg)

    model = MODELS.build(cfg.model)
    optimizer_cfg = cfg.optim_wrapper.optimizer if hasattr(cfg.optim_wrapper, "optimizer") else cfg.optim_wrapper["optimizer"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(args.sanity_lr),
        momentum=float(optimizer_cfg.get("momentum", 0.9)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )

    train_rows = []
    train_iter = cycle(train_loader)
    model.train()
    for step in range(1, args.max_train_iters + 1):
        batch = next(train_iter)
        processed = model.data_preprocessor(batch, training=True)
        losses = model(processed["inputs"], processed["data_samples"], mode="loss")
        reduced, total_loss = reduce_loss_dict(losses)
        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        train_rows.append({"iter": step, "loss_total": float(total_loss.detach().cpu()), **reduced})

    ckpt_path = work_dir / "latest_sanity.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": args.seed,
            "config": str(Path(args.config).resolve()),
            "train_iters": args.max_train_iters,
        },
        ckpt_path,
    )

    metric = METRICS.build(
        dict(
            cfg.val_evaluator,
            outfile_prefix=str((work_dir / "eval" / "results").resolve()),
        )
    )
    metric.dataset_meta = getattr(val_loader.dataset, "metainfo", {})
    model.eval()
    if getattr(model, "test_cfg", None) is not None and "score_thr" in model.test_cfg:
        model.test_cfg.score_thr = 0.0
    val_loss_rows = []
    with torch.no_grad():
        for step, batch in enumerate(val_loss_loader, start=1):
            if step > args.max_val_batches:
                break
            processed = model.data_preprocessor(batch, training=False)
            loss_dict = model(processed["inputs"], processed["data_samples"], mode="loss")
            reduced_loss, total_val_loss = reduce_loss_dict(loss_dict)
            val_loss_rows.append(
                {"iter": step, "loss_total": float(total_val_loss.detach().cpu()), **reduced_loss}
            )
        for step, batch in enumerate(val_loader, start=1):
            if step > args.max_val_batches:
                break
            processed = model.data_preprocessor(batch, training=False)
            preds = model(processed["inputs"], processed["data_samples"], mode="predict")
            metric.process(batch, [pred.to_dict() for pred in preds])

    eval_metrics = metric.compute_metrics(metric.results)
    if not eval_metrics:
        eval_metrics = {
            "coco/bbox_mAP": 0.0,
            "coco/bbox_mAP_50": 0.0,
            "coco/bbox_mAP_75": 0.0,
            "coco/bbox_mAP_s": 0.0,
        }

    train_csv = work_dir / "train_metrics.csv"
    val_loss_csv = work_dir / "val_loss_metrics.csv"
    eval_csv = work_dir / "eval_metrics.csv"
    write_csv(train_csv, train_rows)
    write_csv(val_loss_csv, val_loss_rows)
    write_csv(eval_csv, [eval_metrics])

    summary = {
        "config": str(Path(args.config).resolve()),
        "work_dir": str(work_dir.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "seed": args.seed,
        "max_train_iters": args.max_train_iters,
        "max_val_batches": args.max_val_batches,
        "sanity_lr": args.sanity_lr,
        "grad_clip": args.grad_clip,
        "train_last": train_rows[-1] if train_rows else {},
        "val_loss_last": val_loss_rows[-1] if val_loss_rows else {},
        "eval_metrics": eval_metrics,
        "train_csv": str(train_csv.resolve()),
        "val_loss_csv": str(val_loss_csv.resolve()),
        "eval_csv": str(eval_csv.resolve()),
    }
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("work_dir=", work_dir)
    print("checkpoint=", ckpt_path)
    print("train_last=", summary["train_last"])
    print("val_loss_last=", summary["val_loss_last"])
    print("eval_metrics=", eval_metrics)
    print("summary=", summary_path)


if __name__ == "__main__":
    main()
