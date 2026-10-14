from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from formal_utils import format_mean_std, mean_std


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize formal dual-stream ATSS runs")
    parser.add_argument("--runs-root", required=True, type=str)
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir) if args.out_dir else runs_root
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    for summary_path in runs_root.rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        detail_rows.append(
            {
                "Setting": summary.get("setting"),
                "seed": summary.get("seed"),
                "config_name": summary.get("config_name"),
                "work_dir": summary.get("work_dir"),
                "best_epoch": summary.get("best_epoch"),
                "best_AP": summary.get("best_AP", 0.0),
                "best_AP50": summary.get("best_AP50", 0.0),
                "best_AP75": summary.get("best_AP75", 0.0),
                "best_AP_small": summary.get("best_AP_small", 0.0),
                "final_AP": summary.get("final_AP", 0.0),
                "final_AP50": summary.get("final_AP50", 0.0),
                "final_AP75": summary.get("final_AP75", 0.0),
                "final_AP_small": summary.get("final_AP_small", 0.0),
                "best_checkpoint": summary.get("best_checkpoint"),
                "latest_checkpoint": summary.get("latest_checkpoint"),
            }
        )

    detail_rows.sort(key=lambda row: (row["Setting"] or "", row["seed"] or -1))
    detail_csv = out_dir / "formal_matrix_detail.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()) if detail_rows else [
            "Setting", "seed", "config_name", "work_dir", "best_epoch",
            "best_AP", "best_AP50", "best_AP75", "best_AP_small",
            "final_AP", "final_AP50", "final_AP75", "final_AP_small",
            "best_checkpoint", "latest_checkpoint",
        ])
        writer.writeheader()
        if detail_rows:
            writer.writerows(detail_rows)

    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[row["Setting"]].append(row)

    agg_rows = []
    for setting, rows in grouped.items():
        ap_values = [float(row["best_AP"]) for row in rows]
        ap50_values = [float(row["best_AP50"]) for row in rows]
        ap75_values = [float(row["best_AP75"]) for row in rows]
        aps_values = [float(row["best_AP_small"]) for row in rows]
        ap_mean, ap_std = mean_std(ap_values)
        ap50_mean, ap50_std = mean_std(ap50_values)
        ap75_mean, ap75_std = mean_std(ap75_values)
        aps_mean, aps_std = mean_std(aps_values)
        agg_rows.append(
            {
                "Setting": setting,
                "num_seeds": len(rows),
                "AP_mean": f"{ap_mean:.6f}",
                "AP_std": f"{ap_std:.6f}",
                "AP50_mean": f"{ap50_mean:.6f}",
                "AP50_std": f"{ap50_std:.6f}",
                "AP75_mean": f"{ap75_mean:.6f}",
                "AP75_std": f"{ap75_std:.6f}",
                "AP_small_mean": f"{aps_mean:.6f}",
                "AP_small_std": f"{aps_std:.6f}",
                "AP mean ± std": format_mean_std(ap_values),
                "AP50 mean ± std": format_mean_std(ap50_values),
                "AP75 mean ± std": format_mean_std(ap75_values),
                "AP_small mean ± std": format_mean_std(aps_values),
            }
        )

    agg_csv = out_dir / "formal_matrix_agg.csv"
    with agg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()) if agg_rows else [
            "Setting", "num_seeds",
            "AP_mean", "AP_std", "AP50_mean", "AP50_std", "AP75_mean", "AP75_std", "AP_small_mean", "AP_small_std",
            "AP mean ± std", "AP50 mean ± std", "AP75 mean ± std", "AP_small mean ± std",
        ])
        writer.writeheader()
        if agg_rows:
            writer.writerows(agg_rows)

    table_md = out_dir / "table3_detector_utility.md"
    lines = [
        "| Setting | AP | AP50 | AP75 | AP_small |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            f"| {row['Setting']} | {row['AP mean ± std']} | {row['AP50 mean ± std']} | {row['AP75 mean ± std']} | {row['AP_small mean ± std']} |"
        )
    table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("detail_csv=", detail_csv)
    print("agg_csv=", agg_csv)
    print("table_md=", table_md)


if __name__ == "__main__":
    main()
