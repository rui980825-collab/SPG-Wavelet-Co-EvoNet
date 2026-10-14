#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (Path.cwd() / pp).resolve()


def _to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _iter_run_dirs(runs_root: Path) -> Iterable[Path]:
    for p in sorted([x for x in runs_root.iterdir() if x.is_dir()], key=lambda t: t.name):
        if (p / "meta" / "synthesis_summary.json").exists() and (p / "tables" / "sample_manifest.csv").exists():
            yield p


def _count_bad(rows: List[Dict[str, str]], cls_name: str, bad_ctx: set) -> int:
    n = 0
    for r in rows:
        if r.get("status") != "accepted":
            continue
        if r.get("class") != cls_name:
            continue
        if r.get("ctx_label") in bad_ctx:
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch summarize synthesis runs under one root directory.")
    p.add_argument("--runs-root", required=True, help="Run root directory, e.g. runs/synth/.../v25lite_phaseC_train_tau027")
    p.add_argument("--out-csv", default=None, help="Output CSV path. Default: <runs-root>/run_summary.csv")
    p.add_argument("--max-trials-th", type=float, default=0.10, help="Pass threshold for max_trials_exceeded_rate.")
    p.add_argument("--wavelet-p50-th", type=float, default=1.0, help="Pass threshold for wavelet_gain_p50.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = _resolve_path(args.runs_root)
    out_csv = _resolve_path(args.out_csv) if args.out_csv else (runs_root / "run_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ship_bad_ctx = {"airport", "bridge", "city road", "country road", "playground"}
    plane_bad_ctx = {"sea", "lake", "bridge", "city road", "country road", "playground"}

    rows_out: List[Dict[str, str]] = []
    for run_dir in _iter_run_dirs(runs_root):
        run_id = run_dir.name
        summary = json.loads((run_dir / "meta" / "synthesis_summary.json").read_text(encoding="utf-8"))
        manifest = list(csv.DictReader((run_dir / "tables" / "sample_manifest.csv").open("r", encoding="utf-8", newline="")))

        stats = summary.get("stats", {})
        seed = summary.get("seed", {}).get("effective_seed", "")

        accept_rate = _to_float(stats.get("accept_rate"))
        max_trials_exceeded_rate = _to_float(stats.get("max_trials_exceeded_rate"))
        reject_photometric = _to_int(stats.get("reject_photometric"))
        wavelet_gain_p50 = _to_float(stats.get("wavelet_gain_p50"))
        tir_edge_energy_gain_mean = _to_float(stats.get("tir_edge_energy_gain_mean"))
        ship_bad = _count_bad(manifest, "ship", ship_bad_ctx)
        plane_bad = _count_bad(manifest, "plane", plane_bad_ctx)

        pass_semantic = int(ship_bad == 0 and plane_bad == 0)
        pass_trials = int(max_trials_exceeded_rate < args.max_trials_th)
        pass_photometric = int(reject_photometric > 0)
        pass_wavelet = int(wavelet_gain_p50 > args.wavelet_p50_th)
        pass_all = int(pass_semantic and pass_trials and pass_photometric and pass_wavelet)

        rows_out.append(
            {
                "run_id": run_id,
                "seed": str(seed),
                "accept_rate": f"{accept_rate:.6f}",
                "max_trials_exceeded_rate": f"{max_trials_exceeded_rate:.6f}",
                "reject_photometric": str(reject_photometric),
                "ship_bad": str(ship_bad),
                "plane_bad": str(plane_bad),
                "wavelet_gain_p50": f"{wavelet_gain_p50:.6f}",
                "tir_edge_energy_gain_mean": f"{tir_edge_energy_gain_mean:.6f}",
                "pass_semantic": str(pass_semantic),
                "pass_trials": str(pass_trials),
                "pass_photometric": str(pass_photometric),
                "pass_wavelet": str(pass_wavelet),
                "pass_all": str(pass_all),
            }
        )

    fieldnames = [
        "run_id",
        "seed",
        "accept_rate",
        "max_trials_exceeded_rate",
        "reject_photometric",
        "ship_bad",
        "plane_bad",
        "wavelet_gain_p50",
        "tir_edge_energy_gain_mean",
        "pass_semantic",
        "pass_trials",
        "pass_photometric",
        "pass_wavelet",
        "pass_all",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    pass_count = sum(1 for r in rows_out if r["pass_all"] == "1")
    total = len(rows_out)
    print(f"[summary] runs_root={runs_root}")
    print(f"[summary] runs={total} pass_all={pass_count}/{total}")
    print(f"[summary] out_csv={out_csv}")
    for r in rows_out:
        print(
            f"run={r['run_id']} seed={r['seed']} pass_all={r['pass_all']} "
            f"ship_bad={r['ship_bad']} plane_bad={r['plane_bad']} "
            f"max_trials={r['max_trials_exceeded_rate']} wavelet_p50={r['wavelet_gain_p50']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

