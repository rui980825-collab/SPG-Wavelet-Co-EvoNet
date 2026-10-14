#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _cat_maps(cats: List[dict]) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_name = {}
    name_to_id = {}
    for c in cats:
        cid = int(c["id"])
        name = str(c["name"])
        if cid in id_to_name:
            raise ValueError(f"duplicate category id: {cid}")
        if name in name_to_id:
            raise ValueError(f"duplicate category name: {name}")
        id_to_name[cid] = name
        name_to_id[name] = cid
    return id_to_name, name_to_id


def _default_out_path(in_path: Path, suffix: str) -> Path:
    return in_path.with_name(f"{in_path.stem}{suffix}{in_path.suffix}")


def _align_one(
    synth_ann_path: Path,
    real_cats: List[dict],
    real_name_to_id: Dict[str, int],
    suffix: str,
    out_dir: Path | None,
    inplace: bool,
    strict: bool,
) -> Path:
    obj = _load_json(synth_ann_path)
    cats = obj.get("categories", [])
    anns = obj.get("annotations", [])
    if not isinstance(cats, list) or not isinstance(anns, list):
        raise ValueError(f"invalid COCO schema in {synth_ann_path}")

    synth_id_to_name, _ = _cat_maps(cats)
    missing = sorted({n for n in synth_id_to_name.values() if n not in real_name_to_id})
    if missing:
        msg = f"{synth_ann_path}: category names not in real categories: {missing}"
        if strict:
            raise ValueError(msg)
        print(f"[warn] {msg}")

    old_to_new: Dict[int, int] = {}
    for old_id, name in synth_id_to_name.items():
        if name in real_name_to_id:
            old_to_new[old_id] = int(real_name_to_id[name])

    remap_fail = 0
    for ann in anns:
        old = int(ann["category_id"])
        if old not in old_to_new:
            remap_fail += 1
            continue
        ann["category_id"] = int(old_to_new[old])

    if remap_fail > 0 and strict:
        raise ValueError(f"{synth_ann_path}: {remap_fail} annotations could not be remapped")
    if remap_fail > 0:
        print(f"[warn] {synth_ann_path}: {remap_fail} annotations not remapped")

    # Use real categories as canonical output categories for maximum external compatibility.
    obj["categories"] = real_cats

    if inplace:
        out_path = synth_ann_path
    else:
        if out_dir is not None:
            out_path = out_dir / f"{synth_ann_path.stem}{suffix}{synth_ann_path.suffix}"
        else:
            out_path = _default_out_path(synth_ann_path, suffix=suffix)
    _save_json(out_path, obj)

    print(f"[ok] aligned: {synth_ann_path} -> {out_path}")
    print(f"     remap_cat_ids: {len(old_to_new)} categories, remap_fail={remap_fail}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Align synth COCO category_id to real COCO category_id by category name."
    )
    p.add_argument("--real-ann", required=True, help="Real COCO annotation JSON (canonical category ids).")
    p.add_argument(
        "--synth-ann",
        required=True,
        nargs="+",
        help="One or more synth COCO annotation JSON paths to align.",
    )
    p.add_argument("--suffix", default="_realcat", help="Output filename suffix when not inplace.")
    p.add_argument("--out-dir", default="", help="Optional output directory for aligned JSON files.")
    p.add_argument("--inplace", action="store_true", help="Overwrite synth annotations in-place.")
    p.add_argument("--strict", action="store_true", help="Fail on any unmapped category or annotation.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    real_ann = _resolve_path(args.real_ann)
    synth_anns = [_resolve_path(x) for x in args.synth_ann]
    out_dir = _resolve_path(args.out_dir) if args.out_dir else None

    real_obj = _load_json(real_ann)
    real_cats = sorted(real_obj.get("categories", []), key=lambda c: int(c["id"]))
    if len(real_cats) == 0:
        raise ValueError(f"real ann has empty categories: {real_ann}")
    _, real_name_to_id = _cat_maps(real_cats)

    print(f"[real] {real_ann}")
    print(f"[real] categories={[(int(c['id']), str(c['name'])) for c in real_cats]}")

    out_paths = []
    for p in synth_anns:
        out_paths.append(
            _align_one(
                synth_ann_path=p,
                real_cats=real_cats,
                real_name_to_id=real_name_to_id,
                suffix=str(args.suffix),
                out_dir=out_dir,
                inplace=bool(args.inplace),
                strict=bool(args.strict),
            )
        )

    print("[done] aligned_files:")
    for p in out_paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

