#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (PROJECT_ROOT / pp).resolve()


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return data


def _save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run synthesis shards without hand-editing YAML. "
        "Writes temporary per-shard configs and launches scripts/20_synthesize_dataset.py."
    )
    p.add_argument("--template-config", required=True, help="Base YAML config path.")
    p.add_argument("--shard-start", type=int, required=True, help="Inclusive shard start.")
    p.add_argument("--shard-end", type=int, required=True, help="Inclusive shard end.")
    p.add_argument("--tmp-dir", default="configs/tmp_phaseC_shards", help="Temporary config output directory.")
    p.add_argument("--python-exe", default=sys.executable, help="Python executable used to run synthesis.")
    p.add_argument("--synth-script", default="scripts/20_synthesize_dataset.py", help="Synthesis entry script.")
    p.add_argument("--seed-base", type=int, default=None, help="Override seed_base in emitted configs.")
    p.add_argument("--n-images", type=int, default=None, help="Override n_images in emitted configs.")
    p.add_argument("--out-root", default=None, help="Override out_root in emitted configs.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue remaining shards after a failure.")
    p.add_argument("--keep-tmp", action="store_true", help="Keep generated temporary YAML files.")
    p.add_argument("--dry-run", action="store_true", help="Only print commands, do not execute.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.shard_end < args.shard_start:
        raise ValueError("shard-end must be >= shard-start")

    template_path = _resolve_path(args.template_config)
    tmp_dir = _resolve_path(args.tmp_dir)
    python_exe = _resolve_path(args.python_exe) if args.python_exe != sys.executable else Path(args.python_exe)
    synth_script = _resolve_path(args.synth_script)

    base_cfg = _load_yaml(template_path)
    if str(base_cfg.get("seed_mode", "")).lower() != "sweep":
        print(f"[warn] template seed_mode={base_cfg.get('seed_mode')}, forcing sweep for shard runs.")

    ok = []
    failed = []
    total = args.shard_end - args.shard_start + 1
    print(
        f"[run] template={template_path} shards=[{args.shard_start}..{args.shard_end}] total={total} "
        f"tmp_dir={tmp_dir}"
    )

    for idx, shard in enumerate(range(args.shard_start, args.shard_end + 1), start=1):
        cfg = copy.deepcopy(base_cfg)
        cfg["seed_mode"] = "sweep"
        cfg["seed_shard"] = int(shard)
        if args.seed_base is not None:
            cfg["seed_base"] = int(args.seed_base)
        if args.n_images is not None:
            cfg["n_images"] = int(args.n_images)
        if args.out_root is not None:
            cfg["out_root"] = str(args.out_root)

        cfg_path = tmp_dir / f"{template_path.stem}_shard{shard}.yaml"
        _save_yaml(cfg_path, cfg)

        cmd = [str(python_exe), str(synth_script), "--config", str(cfg_path)]
        print(f"[{idx}/{total}] shard={shard} cmd={' '.join(cmd)}")
        if args.dry_run:
            if not args.keep_tmp:
                cfg_path.unlink(missing_ok=True)
            continue

        t0 = time.time()
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        dt = time.time() - t0
        if res.returncode == 0:
            ok.append(shard)
            print(f"[ok] shard={shard} elapsed={dt:.1f}s")
            if not args.keep_tmp:
                cfg_path.unlink(missing_ok=True)
        else:
            failed.append(shard)
            print(f"[fail] shard={shard} code={res.returncode} elapsed={dt:.1f}s")
            if not args.continue_on_error:
                break

    print(f"[done] ok={len(ok)} failed={len(failed)} ok_shards={ok} failed_shards={failed}")
    return 0 if len(failed) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
