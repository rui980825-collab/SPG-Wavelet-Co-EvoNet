#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict, List


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (Path.cwd() / pp).resolve()


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _find_ann_file(run_dir: Path) -> Path:
    ann_dir = run_dir / "annotations"
    p = ann_dir / "instances_synth.json"
    if p.exists():
        return p
    cands = sorted(ann_dir.glob("*.json"))
    if not cands:
        raise FileNotFoundError(f"no annotation json under {ann_dir}")
    return cands[0]


def _discover_runs(runs_root: Path) -> List[Path]:
    return sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda t: t.name)


def _parse_run_ids_arg(arg: str) -> List[str]:
    return [x.strip() for x in arg.split(",") if x.strip()]


def _load_run_ids_file(path: Path) -> List[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge multiple synthesis COCO runs into one dataset.")
    p.add_argument("--runs-root", required=True, help="Root path containing run_id subdirectories.")
    p.add_argument("--out-root", required=True, help="Output merged dataset root.")
    p.add_argument("--run-ids", default="", help="Comma-separated run ids. Empty means all runs under --runs-root.")
    p.add_argument("--run-ids-file", default="", help="Optional text file listing run ids, one per line.")
    p.add_argument("--annotation-name", default="instances_synth_merged.json", help="Merged annotation filename.")
    p.add_argument("--copy-images", action="store_true", help="Copy images/rgb and images/tir into out-root.")
    p.add_argument("--image-prefix", default="run_id", choices=["run_id", "none"], help="Prefix strategy to avoid collisions.")
    return p.parse_args()


def _normalize_categories(cats: List[dict]) -> List[dict]:
    return sorted(cats, key=lambda x: int(x.get("id", 0)))


def _copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    runs_root = _resolve_path(args.runs_root)
    out_root = _resolve_path(args.out_root)
    out_ann_dir = out_root / "annotations"
    out_img_rgb = out_root / "images" / "rgb"
    out_img_tir = out_root / "images" / "tir"
    out_meta = out_root / "meta"
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        out_img_rgb.mkdir(parents=True, exist_ok=True)
        out_img_tir.mkdir(parents=True, exist_ok=True)

    selected_ids = []
    if args.run_ids:
        selected_ids.extend(_parse_run_ids_arg(args.run_ids))
    if args.run_ids_file:
        selected_ids.extend(_load_run_ids_file(_resolve_path(args.run_ids_file)))
    selected_set = set(selected_ids)

    run_dirs = _discover_runs(runs_root)
    if selected_set:
        run_dirs = [p for p in run_dirs if p.name in selected_set]
    if not run_dirs:
        raise RuntimeError("no runs selected")

    merged_images: List[dict] = []
    merged_annotations: List[dict] = []
    merged_categories: List[dict] = []
    merged_licenses: List[dict] = []
    source_rows: List[Dict[str, str]] = []

    next_image_id = 1
    next_ann_id = 1
    cat_signature = None

    for run_dir in run_dirs:
        run_id = run_dir.name
        ann_path = _find_ann_file(run_dir)
        data = _load_json(ann_path)
        images = data.get("images", [])
        anns = data.get("annotations", [])
        cats = _normalize_categories(data.get("categories", []))
        lics = data.get("licenses", [])

        sig = [(int(c.get("id", 0)), str(c.get("name", ""))) for c in cats]
        if cat_signature is None:
            cat_signature = sig
            merged_categories = cats
            merged_licenses = lics
        elif sig != cat_signature:
            raise ValueError(f"category mismatch in run={run_id}")

        img_id_map: Dict[int, int] = {}
        copied_rgb = 0
        copied_tir = 0
        for img in images:
            old_id = int(img["id"])
            old_name = str(img["file_name"])
            if args.image_prefix == "run_id":
                new_name = f"{run_id}_{old_name}"
            else:
                new_name = old_name

            new_img = dict(img)
            new_img["id"] = next_image_id
            new_img["file_name"] = new_name
            merged_images.append(new_img)
            img_id_map[old_id] = next_image_id
            next_image_id += 1

            if args.copy_images:
                src_rgb = run_dir / "images" / "rgb" / old_name
                src_tir = run_dir / "images" / "tir" / old_name
                dst_rgb = out_img_rgb / new_name
                dst_tir = out_img_tir / new_name
                if src_rgb.exists():
                    _copy_if_exists(src_rgb, dst_rgb)
                    copied_rgb += 1
                _copy_if_exists(src_tir, dst_tir)
                if src_tir.exists():
                    copied_tir += 1

        for ann in anns:
            old_img_id = int(ann["image_id"])
            if old_img_id not in img_id_map:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = img_id_map[old_img_id]
            merged_annotations.append(new_ann)
            next_ann_id += 1

        source_rows.append(
            {
                "run_id": run_id,
                "images": str(len(images)),
                "annotations": str(len(anns)),
                "copied_rgb": str(copied_rgb),
                "copied_tir": str(copied_tir),
                "annotation_file": str(ann_path),
            }
        )
        print(f"[merge] run={run_id} images={len(images)} anns={len(anns)} copied_rgb={copied_rgb} copied_tir={copied_tir}")

    merged = {
        "info": {
            "description": "Merged synthetic dataset",
            "version": "v1",
            "date_created": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_runs_root": str(runs_root),
            "source_runs": [p.name for p in run_dirs],
        },
        "licenses": merged_licenses,
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories,
    }
    out_ann = out_ann_dir / args.annotation_name
    out_ann.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    src_csv = out_meta / "merged_source_runs.csv"
    with src_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["run_id", "images", "annotations", "copied_rgb", "copied_tir", "annotation_file"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in source_rows:
            w.writerow(r)

    print(f"[done] merged_runs={len(run_dirs)} images={len(merged_images)} anns={len(merged_annotations)}")
    print(f"[done] annotation={out_ann}")
    print(f"[done] source_csv={src_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

