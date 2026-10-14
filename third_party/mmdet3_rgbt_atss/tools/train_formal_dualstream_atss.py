from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from formal_utils import (
    ROOT,
    bootstrap_runtime,
    build_val_loss_dataloader,
    infer_device,
    normalize_eval_metrics,
    read_json_lines,
    reduce_loss_dict,
    safe_number,
    set_seed,
    write_csv,
    config_to_setting_label,
)

from mmdet.registry import METRICS, MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="Formal trainer for dual-stream ATSS-R50-FPN")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--work-dir", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--load-from", type=str, default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def patch_cfg(cfg: Config, args) -> Config:
    cfg = cfg.copy()
    cfg.work_dir = args.work_dir
    cfg.launcher = "none"
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    if args.max_epochs is not None:
        cfg.train_cfg.max_epochs = int(args.max_epochs)
    if args.pretrained_path:
        cfg.model.backbone.pretrained_path = str(Path(args.pretrained_path).resolve())
    if args.load_from:
        cfg.load_from = str(Path(args.load_from).resolve())
        cfg.resume = False
    return cfg


def copy_checkpoint(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.resolve())


def resolve_resume_path(work_dir: Path, resume_arg: str | None) -> Path | None:
    if not resume_arg:
        return None
    if resume_arg == "auto":
        candidate = work_dir / "latest_formal.pth"
        return candidate if candidate.exists() else None
    candidate = Path(resume_arg)
    return candidate if candidate.exists() else None


def build_optimizer(model, optimizer_cfg: dict):
    optimizer_type = optimizer_cfg.get("type", "SGD")
    kwargs = dict(optimizer_cfg)
    kwargs.pop("type", None)
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), **kwargs)
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer type for formal script: {optimizer_type}")


def run_val_loss(model, cfg: Config, max_val_batches: int | None) -> dict:
    val_loader = build_val_loss_dataloader(cfg)
    model.eval()
    rows = []
    with torch.no_grad():
        for step, batch in enumerate(val_loader, start=1):
            if max_val_batches is not None and step > max_val_batches:
                break
            processed = model.data_preprocessor(batch, training=False)
            loss_dict = model(processed["inputs"], processed["data_samples"], mode="loss")
            reduced = {}
            total = None
            for key, value in loss_dict.items():
                scalar = sum(v.mean() for v in value) if isinstance(value, (list, tuple)) else value.mean()
                reduced[key] = float(scalar.detach().cpu())
                total = scalar if total is None else total + scalar
            rows.append({"iter": step, "loss_total": float(total.detach().cpu()), **reduced})
    model.train()
    if not rows:
        return {}
    summary = {"num_batches": len(rows)}
    for key in rows[0].keys():
        if key == "iter":
            continue
        valid = [float(row[key]) for row in rows if safe_number(row[key]) is not None]
        summary[key] = float(sum(valid) / len(valid)) if valid else None
    return summary


def run_eval(model, cfg: Config, max_eval_batches: int | None, work_dir: Path, epoch: int) -> dict:
    eval_loader = Runner.build_dataloader(cfg.val_dataloader)
    metric = METRICS.build(
        dict(
            cfg.val_evaluator,
            outfile_prefix=str((work_dir / "eval" / f"epoch_{epoch:02d}" / "results").resolve()),
        )
    )
    metric.dataset_meta = getattr(eval_loader.dataset, "metainfo", {})
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_loader, start=1):
            if max_eval_batches is not None and step > max_eval_batches:
                break
            processed = model.data_preprocessor(batch, training=False)
            preds = model(processed["inputs"], processed["data_samples"], mode="predict")
            metric.process(batch, [pred.to_dict() for pred in preds])
    metrics = metric.compute_metrics(metric.results)
    return normalize_eval_metrics(metrics)


def load_model_checkpoint(model, optimizer, checkpoint_path: Path, strict_model: bool) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    model.load_state_dict(state_dict, strict=strict_model)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


def run_manual_micro_formal(cfg: Config, args, device: torch.device, work_dir: Path) -> dict:
    train_loader = Runner.build_dataloader(cfg.train_dataloader)
    model = MODELS.build(cfg.model).to(device)
    optimizer_cfg = cfg.optim_wrapper.optimizer if hasattr(cfg.optim_wrapper, "optimizer") else cfg.optim_wrapper["optimizer"]
    optimizer = build_optimizer(model, optimizer_cfg)

    resume_path = resolve_resume_path(work_dir, args.resume)
    if resume_path is not None:
        load_model_checkpoint(model, optimizer, resume_path, strict_model=True)
    elif args.load_from:
        load_model_checkpoint(model, None, Path(args.load_from), strict_model=False)

    model.train()
    train_rows = []
    train_iter = iter(train_loader)
    for step in range(1, int(args.max_train_batches) + 1):
        batch = next(train_iter)
        processed = model.data_preprocessor(batch, training=True)
        losses = model(processed["inputs"], processed["data_samples"], mode="loss")
        reduced, total_loss = reduce_loss_dict(losses)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_rows.append({"epoch": 1, "iter": step, "loss_total": float(total_loss.detach().cpu()), **reduced})

    latest_payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 1,
        "seed": args.seed,
        "config": str(Path(args.config).resolve()),
    }
    latest_path = work_dir / "latest_formal.pth"
    torch.save(latest_payload, latest_path)

    val_loss_summary = run_val_loss(model, cfg, args.max_val_batches)
    eval_row = {"epoch": 1, "step": int(args.max_train_batches), **run_eval(model, cfg, args.max_eval_batches, work_dir, epoch=1)}
    best_path = None
    if eval_row["AP50"] >= 0.0:
        best_path = work_dir / "best_ap50.pth"
        torch.save(latest_payload, best_path)

    return {
        "train_rows": train_rows,
        "val_loss_summary": val_loss_summary,
        "eval_rows": [eval_row],
        "latest_checkpoint": str(latest_path.resolve()),
        "best_checkpoint": str(best_path.resolve()) if best_path else None,
        "scalars_json": None,
    }


def main():
    args = parse_args()
    bootstrap_runtime()
    set_seed(args.seed)
    device = infer_device(args.device)

    cfg = patch_cfg(Config.fromfile(args.config), args)
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_path(work_dir, args.resume)
    if resume_path is not None:
        cfg.resume = True
        cfg.load_from = str(resume_path.resolve())

    if args.max_train_batches is None:
        runner = Runner.from_cfg(cfg)
        runner.model.to(device)
        runner.train()
        scalars_path = Path(runner.work_dir) / runner.timestamp / "vis_data" / "scalars.json"
        scalar_rows = read_json_lines(scalars_path)
        train_rows = [{k: safe_number(v) if isinstance(v, (int, float)) else v for k, v in dict(row).items()} for row in scalar_rows if "loss" in row or "loss_cls" in row or "lr" in row]
        eval_rows = [dict(row) for row in scalar_rows if any(k in row for k in ("bbox_mAP", "coco/bbox_mAP", "bbox_mAP_50", "coco/bbox_mAP_50"))]
        val_loss_summary = run_val_loss(runner.model, cfg, args.max_val_batches)
        latest_candidates = sorted(Path(runner.work_dir).glob("latest*.pth"))
        best_candidates = sorted(Path(runner.work_dir).glob("best*.pth"))
        latest_src = latest_candidates[-1] if latest_candidates else None
        best_src = None
        for candidate in best_candidates:
            if "bbox_mAP_50" in candidate.name or "coco_bbox_mAP_50" in candidate.name:
                best_src = candidate
                break
        if best_src is None and best_candidates:
            best_src = best_candidates[-1]
        latest_dst = work_dir / "latest_formal.pth"
        best_dst = work_dir / "best_ap50.pth"
        latest_ckpt = copy_checkpoint(latest_src, latest_dst)
        best_ckpt = copy_checkpoint(best_src, best_dst)
        normalized_eval_rows = []
        for row in eval_rows:
            normalized = normalize_eval_metrics(row)
            normalized_eval_rows.append({"epoch": row.get("epoch"), "step": row.get("step"), **normalized})
        scalars_json = str(scalars_path.resolve()) if scalars_path.exists() else None
    else:
        manual = run_manual_micro_formal(cfg, args, device, work_dir)
        train_rows = manual["train_rows"]
        val_loss_summary = manual["val_loss_summary"]
        normalized_eval_rows = manual["eval_rows"]
        latest_ckpt = manual["latest_checkpoint"]
        best_ckpt = manual["best_checkpoint"]
        scalars_json = manual["scalars_json"]

    train_csv = work_dir / "train_metrics.csv"
    val_csv = work_dir / "val_metrics.csv"
    eval_csv = work_dir / "eval_metrics.csv"
    write_csv(train_csv, train_rows)
    write_csv(val_csv, [val_loss_summary] if val_loss_summary else [], fieldnames=list(val_loss_summary.keys()) if val_loss_summary else ["num_batches", "loss_total"])
    write_csv(eval_csv, normalized_eval_rows, fieldnames=["epoch", "step", "AP", "AP50", "AP75", "AP_small"])

    best_row = max(normalized_eval_rows, key=lambda row: row["AP50"]) if normalized_eval_rows else {"epoch": None, "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0}
    final_row = normalized_eval_rows[-1] if normalized_eval_rows else {"epoch": None, "AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0}

    summary = {
        "config_name": Path(args.config).name,
        "config": str(Path(args.config).resolve()),
        "setting": config_to_setting_label(args.config),
        "seed": args.seed,
        "device": str(device),
        "work_dir": str(work_dir.resolve()),
        "pretrained_path": str(Path(args.pretrained_path).resolve()) if args.pretrained_path else None,
        "resume_from": str(resume_path.resolve()) if resume_path else None,
        "load_from": str(Path(args.load_from).resolve()) if args.load_from else None,
        "max_train_batches": args.max_train_batches,
        "best_epoch": best_row.get("epoch"),
        "best_AP": best_row.get("AP", 0.0),
        "best_AP50": best_row.get("AP50", 0.0),
        "best_AP75": best_row.get("AP75", 0.0),
        "best_AP_small": best_row.get("AP_small", 0.0),
        "final_AP": final_row.get("AP", 0.0),
        "final_AP50": final_row.get("AP50", 0.0),
        "final_AP75": final_row.get("AP75", 0.0),
        "final_AP_small": final_row.get("AP_small", 0.0),
        "latest_checkpoint": latest_ckpt,
        "best_checkpoint": best_ckpt,
        "train_csv": str(train_csv.resolve()),
        "val_csv": str(val_csv.resolve()),
        "eval_csv": str(eval_csv.resolve()),
        "scalars_json": scalars_json,
        "train_last": train_rows[-1] if train_rows else {},
        "val_last": val_loss_summary,
        "eval_last": final_row,
    }
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("work_dir=", work_dir)
    print("latest_formal=", latest_ckpt)
    print("best_ap50=", best_ckpt)
    print("summary=", summary_path)
    print("best_AP50=", summary["best_AP50"])
    print("final_AP50=", summary["final_AP50"])


if __name__ == "__main__":
    main()
