from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig_mmdet3")))


DEFAULT_CONFIGS = {
    "real_only": ROOT / "configs" / "dualstream_atss_r50_fpn_1x_coevonet_real_only.py",
    "phasec_synth0p25x": ROOT / "configs" / "dualstream_atss_r50_fpn_1x_coevonet_phasec_synth0p25x.py",
    "phasec_density_synth0p25x": ROOT / "configs" / "dualstream_atss_r50_fpn_1x_coevonet_phasec_density_synth0p25x.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tiny sanity matrix for dual-stream ATSS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-iters", type=int, default=4)
    parser.add_argument("--max-val-batches", type=int, default=5)
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs/mmdet3_rgbt_atss_sanity",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    python_exe = Path(sys.executable).resolve()
    train_script = ROOT / "tools" / "train_sanity_dualstream_atss.py"

    rows = []
    for run_name, config_path in DEFAULT_CONFIGS.items():
        work_dir = runs_root / f"{run_name}_seed{args.seed}"
        cmd = [
            str(python_exe),
            str(train_script),
            "--config",
            str(config_path),
            "--work-dir",
            str(work_dir),
            "--seed",
            str(args.seed),
            "--max-train-iters",
            str(args.max_train_iters),
            "--max-val-batches",
            str(args.max_val_batches),
        ]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
        eval_metrics = summary.get("eval_metrics", {})
        ap = eval_metrics.get("bbox_mAP", eval_metrics.get("coco/bbox_mAP"))
        ap50 = eval_metrics.get("bbox_mAP_50", eval_metrics.get("coco/bbox_mAP_50"))
        ap75 = eval_metrics.get("bbox_mAP_75", eval_metrics.get("coco/bbox_mAP_75"))
        aps = eval_metrics.get("bbox_mAP_s", eval_metrics.get("coco/bbox_mAP_s"))
        rows.append(
            {
                "run_name": run_name,
                "seed": args.seed,
                "config": summary["config"],
                "work_dir": summary["work_dir"],
                "train_loss": summary.get("train_last", {}).get("loss_total"),
                "val_loss": summary.get("val_loss_last", {}).get("loss_total"),
                "AP": ap,
                "AP50": ap50,
                "AP75": ap75,
                "AP_small": aps,
            }
        )

    out_csv = runs_root / "sanity_matrix_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("summary_csv=", out_csv)


if __name__ == "__main__":
    main()
