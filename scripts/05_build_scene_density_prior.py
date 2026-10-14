#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


DEFAULT_BUCKETS = [
    "sparse:1:4",
    "medium:5:9",
    "dense:10:19",
    "crowded:20:999",
]


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (Path.cwd() / pp).resolve()


def _parse_bucket(spec: str) -> Tuple[str, int, int]:
    parts = str(spec).split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid bucket spec: {spec} (expected name:min:max)")
    name = str(parts[0]).strip()
    mn = int(parts[1])
    mx = int(parts[2])
    if mx < mn:
        raise ValueError(f"invalid bucket range: {spec}")
    return name, mn, mx


def _count_stats(values: List[int]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0}
    vals = sorted(int(v) for v in values)
    n = len(vals)

    def q(frac: float) -> float:
        idx = int(round((n - 1) * float(frac)))
        idx = max(0, min(n - 1, idx))
        return float(vals[idx])

    return {
        "mean": float(sum(vals) / max(1, n)),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": int(vals[-1]),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build scene-density prior from a COCO annotation file.")
    ap.add_argument("--ann-file", required=True, help="COCO annotation json.")
    ap.add_argument("--out-yaml", required=True, help="Output YAML path for density prior.")
    ap.add_argument("--out-hist-csv", default="", help="Optional per-image object-count histogram CSV.")
    ap.add_argument("--dataset-name", default="", help="Optional dataset name in output metadata.")
    ap.add_argument(
        "--buckets",
        nargs="+",
        default=DEFAULT_BUCKETS,
        help="Density buckets, each in name:min:max format.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ann_file = _resolve_path(args.ann_file)
    out_yaml = _resolve_path(args.out_yaml)
    out_hist_csv = _resolve_path(args.out_hist_csv) if str(args.out_hist_csv).strip() else Path("")

    obj = json.loads(ann_file.read_text(encoding="utf-8"))
    image_ids = [int(im["id"]) for im in obj.get("images", [])]
    img_counts: Dict[int, int] = {img_id: 0 for img_id in image_ids}
    for ann in obj.get("annotations", []):
        img_id = int(ann["image_id"])
        img_counts[img_id] = int(img_counts.get(img_id, 0)) + 1

    per_image_counts = [int(img_counts[img_id]) for img_id in image_ids]
    non_empty_counts = [v for v in per_image_counts if int(v) > 0]
    stats = _count_stats(per_image_counts)
    empty_rate = float(sum(1 for v in per_image_counts if int(v) == 0) / max(1, len(per_image_counts)))

    buckets = [_parse_bucket(spec) for spec in args.buckets]
    density_modes = {}
    for name, mn, mx in buckets:
        matched = [v for v in per_image_counts if mn <= int(v) <= mx]
        matched_non_empty = [v for v in non_empty_counts if mn <= int(v) <= mx]
        density_modes[name] = {
            "min_objects": int(mn),
            "max_objects": int(mx),
            "image_count": int(len(matched)),
            "image_fraction": float(len(matched) / max(1, len(per_image_counts))),
            "non_empty_fraction": float(len(matched_non_empty) / max(1, len(non_empty_counts))),
            "mean_objects": float(sum(matched) / max(1, len(matched))) if matched else 0.0,
            "weight": float(len(matched_non_empty) / max(1, len(non_empty_counts))),
        }

    output = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": str(args.dataset_name or ann_file.stem),
        "source_annotation": str(ann_file),
        "image_count": int(len(per_image_counts)),
        "non_empty_image_count": int(len(non_empty_counts)),
        "empty_rate": float(empty_rate),
        "object_count_stats": stats,
        "density_modes": density_modes,
        "default_sampling": {
            "enabled": True,
            "allow_empty": False,
            "modes": {
                str(name): {
                    "min_objects": int(rec["min_objects"]),
                    "max_objects": int(rec["max_objects"]),
                    "weight": float(rec["weight"]),
                }
                for name, rec in density_modes.items()
            },
        },
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if out_hist_csv:
        hist_rows = []
        hist: Dict[int, int] = {}
        for v in per_image_counts:
            hist[int(v)] = int(hist.get(int(v), 0)) + 1
        for object_count in sorted(hist.keys()):
            image_count = int(hist[object_count])
            hist_rows.append(
                {
                    "object_count": int(object_count),
                    "image_count": image_count,
                    "fraction": float(image_count / max(1, len(per_image_counts))),
                }
            )
        out_hist_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_hist_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["object_count", "image_count", "fraction"])
            w.writeheader()
            for row in hist_rows:
                w.writerow(row)

    print(f"[ok] ann_file={ann_file}")
    print(f"[ok] out_yaml={out_yaml}")
    if out_hist_csv:
        print(f"[ok] out_hist_csv={out_hist_csv}")
    print(
        f"[ok] image_count={len(per_image_counts)} mean={stats['mean']:.3f} "
        f"p50={stats['p50']:.1f} p90={stats['p90']:.1f} empty_rate={empty_rate:.4f}"
    )
    for name, rec in density_modes.items():
        print(
            f"     {name}: range=[{rec['min_objects']},{rec['max_objects']}] "
            f"image_fraction={rec['image_fraction']:.4f} non_empty_fraction={rec['non_empty_fraction']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
