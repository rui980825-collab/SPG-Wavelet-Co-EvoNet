from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from formal_utils import CONFIG_PATHS, ROOT, config_to_setting_label


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 3-seed formal matrix for dual-stream ATSS")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--runs-root", type=str, default="runs/mmdet3_rgbt_atss_formal")
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    python_exe = Path(sys.executable).resolve()
    train_script = ROOT / "tools" / "train_formal_dualstream_atss.py"

    rows = []
    for run_name, config_path in CONFIG_PATHS.items():
        for seed in seeds:
            work_dir = runs_root / run_name / f"seed{seed}"
            cmd = [
                str(python_exe),
                str(train_script),
                "--config",
                str(config_path),
                "--work-dir",
                str(work_dir),
                "--seed",
                str(seed),
            ]
            if args.pretrained_path:
                cmd.extend(["--pretrained-path", str(args.pretrained_path)])
            if args.max_epochs is not None:
                cmd.extend(["--max-epochs", str(args.max_epochs)])
            if args.max_train_batches is not None:
                cmd.extend(["--max-train-batches", str(args.max_train_batches)])
            if args.max_val_batches is not None:
                cmd.extend(["--max-val-batches", str(args.max_val_batches)])

            env = os.environ.copy()
            env.setdefault("MPLCONFIGDIR", ".mplconfig_mmdet3")
            print("running:", " ".join(cmd))
            subprocess.run(cmd, cwd=str(ROOT.parent.parent), env=env, check=True)

            summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_name": run_name,
                    "setting": config_to_setting_label(summary["config_name"]),
                    "seed": seed,
                    "config": summary["config"],
                    "work_dir": summary["work_dir"],
                    "best_epoch": summary["best_epoch"],
                    "best_AP": summary["best_AP"],
                    "best_AP50": summary["best_AP50"],
                    "best_AP75": summary["best_AP75"],
                    "best_AP_small": summary["best_AP_small"],
                    "final_AP": summary["final_AP"],
                    "final_AP50": summary["final_AP50"],
                    "final_AP75": summary["final_AP75"],
                    "final_AP_small": summary["final_AP_small"],
                }
            )

    out_csv = runs_root / "formal_matrix_runs.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "run_name", "setting", "seed", "config", "work_dir",
            "best_epoch", "best_AP", "best_AP50", "best_AP75", "best_AP_small",
            "final_AP", "final_AP50", "final_AP75", "final_AP_small",
        ])
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    print("formal_matrix_runs=", out_csv)


if __name__ == "__main__":
    main()
