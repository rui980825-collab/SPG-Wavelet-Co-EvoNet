# scripts/02_build_prototype_bank.py
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.amp import autocast

# Ensure project root is importable when running "python scripts/xx.py".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coevonet.models.generator import SiameseWKANAE


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_runtime_device(requested_device: str) -> torch.device:
    req = str(requested_device).strip().lower()
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA requested but not available. Fallback to CPU.")
            return torch.device("cpu")
        try:
            major, minor = torch.cuda.get_device_capability(0)
            sm = f"sm_{major}{minor}"
            archs = set(torch.cuda.get_arch_list())
            if len(archs) > 0 and sm not in archs:
                print(f"[warn] Torch CUDA arch list does not include {sm} (supported={sorted(archs)}). Fallback to CPU.")
                return torch.device("cpu")
        except Exception as e:
            print(f"[warn] CUDA capability check failed ({e}). Keep requested CUDA.")
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _size_bucket_from_hw(h: int, w: int) -> str:
    m = int(max(h, w))
    if m <= 12:
        return "tiny"
    if m <= 24:
        return "small"
    if m <= 48:
        return "medium"
    return "large"


def list_patch_pairs(patch_bank_root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    rgb_root = patch_bank_root / "rgb"
    tir_root = patch_bank_root / "tir"
    if not rgb_root.exists() or not tir_root.exists():
        raise FileNotFoundError(f"patch_bank_root must contain rgb/ and tir/. Got: {patch_bank_root}")

    out: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for cls_dir in sorted([p for p in rgb_root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        rgb_files = list(cls_dir.rglob("*.png")) + list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.jpeg"))
        for rp in rgb_files:
            rel = rp.relative_to(rgb_root)
            tp = tir_root / rel
            if tp.exists():
                out[cls].append((rp, tp))
    return out


def read_patch_pair(rgb_path: Path, tir_path: Path, img_size: int) -> Tuple[np.ndarray, np.ndarray, str]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    tir = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or tir is None:
        raise RuntimeError(f"failed to read pair: {rgb_path}, {tir_path}")
    h, w = rgb.shape[:2]
    size_bucket = _size_bucket_from_hw(h, w)

    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tir = cv2.resize(tir, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb, tir, size_bucket


def extract_feature_color_stats(rgb_u8: np.ndarray, tir_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    tir = tir_u8.astype(np.float32) / 255.0
    feat = np.concatenate(
        [
            rgb.mean(axis=(0, 1)),
            rgb.std(axis=(0, 1)),
            np.asarray([tir.mean(), tir.std()], dtype=np.float32),
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def extract_feature_generator(
    model: SiameseWKANAE,
    device: torch.device,
    use_amp: bool,
    rgb_list: List[np.ndarray],
    tir_list: List[np.ndarray],
) -> np.ndarray:
    rgb_np = np.stack(rgb_list, axis=0).astype(np.float32) / 255.0
    tir_np = np.stack(tir_list, axis=0).astype(np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).to(device)
    tir_t = torch.from_numpy(tir_np).unsqueeze(1).to(device)
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=use_amp):
            fr = model.stem_rgb(rgb_t)
            ft = model.stem_tir(tir_t)
            zr = model.encoder(fr)
            zt = model.encoder(ft)
            z = 0.5 * (zr + zt)
    return z.float().cpu().numpy().astype(np.float32)


def kmeans_np(x: np.ndarray, k: int, iters: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    if n < 1:
        raise RuntimeError("kmeans got empty input")
    if k <= 1:
        c = _normalize_rows(np.mean(x, axis=0, keepdims=True).astype(np.float32))
        labels = np.zeros((n,), dtype=np.int32)
        return c, labels

    rng = np.random.default_rng(seed)
    kk = int(min(k, n))
    init_idx = rng.choice(n, size=kk, replace=False)
    centers = x[init_idx].copy()
    centers = _normalize_rows(centers)

    labels = np.zeros((n,), dtype=np.int32)
    for _ in range(int(max(1, iters))):
        sim = x @ centers.T
        labels = np.argmax(sim, axis=1).astype(np.int32)
        new_c = np.zeros_like(centers)
        for j in range(kk):
            m = x[labels == j]
            if m.shape[0] == 0:
                new_c[j] = x[int(rng.integers(0, n))]
            else:
                new_c[j] = np.mean(m, axis=0)
        centers = _normalize_rows(new_c.astype(np.float32))

    return centers, labels


def write_confuse_csv(path: Path, classes: List[str], mat: np.ndarray):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class"] + classes)
        for i, cls in enumerate(classes):
            row = [cls] + [f"{float(mat[i, j]):.6f}" for j in range(len(classes))]
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_bank_root", type=str, default="data/patches/patch_bank")
    ap.add_argument("--out_root", type=str, default="runs/prototypes/v25lite")
    ap.add_argument("--append_run_id", action="store_true")

    ap.add_argument("--feature_mode", type=str, default="generator_encoder", choices=["generator_encoder", "color_stats"])
    ap.add_argument("--generator_ckpt", type=str, default="runs/gen/uav_rgbir/20251226_160415/checkpoints/best.pth")
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--latent_ch", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--proto_k", type=int, default=1)
    ap.add_argument("--kmeans_iters", type=int, default=20)
    ap.add_argument("--max_pairs_per_class", type=int, default=6000)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    patch_bank_root = Path(args.patch_bank_root)
    out_root = Path(args.out_root)
    run_dir = (out_root / now_id()) if bool(args.append_run_id) else out_root
    ensure_dir(run_dir)
    ensure_dir(run_dir / "meta")

    set_seed(int(args.seed))
    device = _resolve_runtime_device(args.device)
    use_amp = bool(args.amp) and (device.type == "cuda")

    class_pairs = list_patch_pairs(patch_bank_root)
    classes = sorted(class_pairs.keys())
    if len(classes) == 0:
        raise RuntimeError("No classes found in patch bank.")

    model = None
    if args.feature_mode == "generator_encoder":
        ckpt = Path(args.generator_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"generator_ckpt not found: {ckpt}")
        model = SiameseWKANAE(base_ch=int(args.base_ch), latent_ch=int(args.latent_ch)).to(device)
        state = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()

    print(f"[proto] classes={len(classes)} feature_mode={args.feature_mode} img_size={args.img_size}")
    print(f"[proto] device={device.type} amp={use_amp} out_dir={run_dir}")

    class_to_prototypes = {}
    class_to_patch_features = {}
    class_to_patch_rgb_rel = {}
    class_to_patch_tir_rel = {}
    class_to_size_bucket = {}
    class_to_patch_proto_id = {}
    class_to_counts = {}
    proto_stats = {"classes": {}, "feature_mode": args.feature_mode, "proto_k": int(args.proto_k), "img_size": int(args.img_size)}

    for ci, cls in enumerate(classes):
        pairs = list(class_pairs[cls])
        if int(args.max_pairs_per_class) > 0 and len(pairs) > int(args.max_pairs_per_class):
            rng = np.random.default_rng(int(args.seed) + ci)
            idx = rng.choice(len(pairs), size=int(args.max_pairs_per_class), replace=False)
            pairs = [pairs[int(i)] for i in idx.tolist()]
            pairs = sorted(pairs, key=lambda x: str(x[0]))

        rgb_rel = []
        tir_rel = []
        buckets = []
        feats_all = []

        if args.feature_mode == "generator_encoder":
            rgb_batch: List[np.ndarray] = []
            tir_batch: List[np.ndarray] = []
            meta_batch: List[Tuple[str, str, str]] = []
            for rp, tp in pairs:
                rgb_u8, tir_u8, bkt = read_patch_pair(rp, tp, int(args.img_size))
                rgb_batch.append(rgb_u8)
                tir_batch.append(tir_u8)
                meta_batch.append((rp.relative_to(patch_bank_root).as_posix(), tp.relative_to(patch_bank_root).as_posix(), bkt))

                if len(rgb_batch) >= int(args.batch_size):
                    z = extract_feature_generator(model, device, use_amp, rgb_batch, tir_batch)
                    feats_all.append(z)
                    for rr, tt, bb in meta_batch:
                        rgb_rel.append(rr)
                        tir_rel.append(tt)
                        buckets.append(bb)
                    rgb_batch, tir_batch, meta_batch = [], [], []

            if len(rgb_batch) > 0:
                z = extract_feature_generator(model, device, use_amp, rgb_batch, tir_batch)
                feats_all.append(z)
                for rr, tt, bb in meta_batch:
                    rgb_rel.append(rr)
                    tir_rel.append(tt)
                    buckets.append(bb)
        else:
            vecs = []
            for rp, tp in pairs:
                rgb_u8, tir_u8, bkt = read_patch_pair(rp, tp, int(args.img_size))
                v = extract_feature_color_stats(rgb_u8, tir_u8)
                vecs.append(v)
                rgb_rel.append(rp.relative_to(patch_bank_root).as_posix())
                tir_rel.append(tp.relative_to(patch_bank_root).as_posix())
                buckets.append(bkt)
            feats_all = [np.stack(vecs, axis=0).astype(np.float32)] if len(vecs) > 0 else []

        if len(feats_all) == 0:
            raise RuntimeError(f"class {cls} produced no features.")
        feats = np.concatenate(feats_all, axis=0).astype(np.float32)
        feats = _normalize_rows(feats)
        n, d = feats.shape

        if n < 1:
            raise RuntimeError(f"class {cls} has zero samples after preprocessing.")
        k = int(max(1, min(int(args.proto_k), n)))
        protos, labels = kmeans_np(feats, k=k, iters=int(args.kmeans_iters), seed=int(args.seed) + ci)
        protos = _normalize_rows(protos.astype(np.float32))

        cluster_ratio = []
        for j in range(k):
            cluster_ratio.append(float((labels == j).mean()))
        variance = float(np.mean(np.var(feats, axis=0)))

        class_to_prototypes[cls] = protos.astype(np.float32)
        class_to_patch_features[cls] = feats.astype(np.float32)
        class_to_patch_rgb_rel[cls] = rgb_rel
        class_to_patch_tir_rel[cls] = tir_rel
        class_to_size_bucket[cls] = buckets
        class_to_patch_proto_id[cls] = labels.astype(np.int32)
        class_to_counts[cls] = int(n)

        proto_stats["classes"][cls] = {
            "n_samples": int(n),
            "feature_dim": int(d),
            "proto_k": int(k),
            "cluster_ratio": [float(x) for x in cluster_ratio],
            "feature_variance_mean": float(variance),
            "size_bucket_hist": {
                "tiny": int(sum(1 for x in buckets if x == "tiny")),
                "small": int(sum(1 for x in buckets if x == "small")),
                "medium": int(sum(1 for x in buckets if x == "medium")),
                "large": int(sum(1 for x in buckets if x == "large")),
            },
        }
        print(f"[proto] class={cls} n={n} k={k} var={variance:.6f}")

    for cls in classes:
        if cls not in class_to_prototypes:
            raise RuntimeError(f"class missing prototype: {cls}")

    class_mean = []
    for cls in classes:
        p = np.asarray(class_to_prototypes[cls], dtype=np.float32)
        cm = np.mean(p, axis=0, keepdims=True).astype(np.float32)
        cm = _normalize_rows(cm)[0]
        class_mean.append(cm)
    class_mean = np.stack(class_mean, axis=0).astype(np.float32)
    confuse = class_mean @ class_mean.T

    bank = {
        "version": "v25lite_proto_bank_v1",
        "feature_mode": args.feature_mode,
        "feature_dim": int(class_mean.shape[1]),
        "proto_k": int(args.proto_k),
        "img_size": int(args.img_size),
        "classes": classes,
        "class_to_prototypes": class_to_prototypes,
        "class_to_patch_features": class_to_patch_features,
        "class_to_patch_rgb_rel": class_to_patch_rgb_rel,
        "class_to_patch_tir_rel": class_to_patch_tir_rel,
        "class_to_size_bucket": class_to_size_bucket,
        "class_to_patch_proto_id": class_to_patch_proto_id,
        "class_to_counts": class_to_counts,
        "class_confuse_matrix": confuse.astype(np.float32),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patch_bank_root": str(patch_bank_root),
        "generator_ckpt": str(args.generator_ckpt) if args.feature_mode == "generator_encoder" else "",
    }

    np.save(run_dir / "prototypes.npy", bank, allow_pickle=True)
    (run_dir / "proto_stats.json").write_text(json.dumps(proto_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_confuse_csv(run_dir / "proto_confuse.csv", classes, confuse)

    meta = {
        "run_dir": str(run_dir),
        "classes": classes,
        "feature_mode": args.feature_mode,
        "proto_k": int(args.proto_k),
        "img_size": int(args.img_size),
        "max_pairs_per_class": int(args.max_pairs_per_class),
        "device": str(device),
        "amp": bool(use_amp),
        "seed": int(args.seed),
        "outputs": {
            "prototypes_npy": str(run_dir / "prototypes.npy"),
            "proto_stats_json": str(run_dir / "proto_stats.json"),
            "proto_confuse_csv": str(run_dir / "proto_confuse.csv"),
        },
    }
    (run_dir / "meta" / "build_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] prototype bank built")
    print(f"     run_dir={run_dir}")
    print(f"     prototypes={run_dir / 'prototypes.npy'}")
    print(f"     stats={run_dir / 'proto_stats.json'}")
    print(f"     confuse={run_dir / 'proto_confuse.csv'}")


if __name__ == "__main__":
    main()
