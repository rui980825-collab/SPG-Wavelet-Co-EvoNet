# scripts/20_synthesize_dataset.py
# -*- coding: utf-8 -*-

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import yaml
from torch.amp import autocast

# Ensure project root is importable when running "python scripts/xx.py".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coevonet.models.generator import SiameseWKANAE
from coevonet.synthesis.constraints import (
    ConstraintConfig,
    semantic_gate,
    photometric_match_details,
    placement_score,
)
from coevonet.synthesis.wavelet_refiner import wavelet_refine_tir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_id():
    return time.strftime("%Y%m%d_%H%M%S")


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_img_bgr(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_img_gray(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def save_img_bgr(path: Path, img_bgr: np.ndarray, ext: str = "jpg", quality: int = 95):
    ensure_dir(path.parent)
    if ext.lower() in ["jpg", "jpeg"]:
        cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        cv2.imwrite(str(path), img_bgr)


def save_img_gray(path: Path, img_u8: np.ndarray, ext: str = "jpg", quality: int = 95):
    ensure_dir(path.parent)
    if ext.lower() in ["jpg", "jpeg"]:
        cv2.imwrite(str(path), img_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    else:
        cv2.imwrite(str(path), img_u8)


def write_csv(path: Path, header: List[str], rows: List[List]):
    ensure_dir(path.parent)
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_json(path: Path, obj):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_local_path(path_like: str) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _load_yaml_or_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config/prior not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
    else:
        obj = yaml.safe_load(text)
    return obj or {}


def _normalize_density_modes(raw_modes, fallback_min: int, fallback_max: int, allow_empty: bool) -> List[dict]:
    out: List[dict] = []
    if not isinstance(raw_modes, dict):
        return out
    for name, rec in raw_modes.items():
        mn = fallback_min
        mx = fallback_max
        weight = 1.0
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            mn, mx = int(rec[0]), int(rec[1])
        elif isinstance(rec, dict):
            rr = rec.get("range", None)
            if isinstance(rr, (list, tuple)) and len(rr) >= 2:
                mn, mx = int(rr[0]), int(rr[1])
            else:
                mn = int(rec.get("min_objects", rec.get("min", fallback_min)))
                mx = int(rec.get("max_objects", rec.get("max", fallback_max)))
            weight = float(
                rec.get(
                    "weight",
                    rec.get(
                        "non_empty_fraction",
                        rec.get("sampling_weight", rec.get("image_fraction", 1.0)),
                    ),
                )
            )
        mn = int(max(0, mn))
        mx = int(max(mn, mx))
        if (not allow_empty) and (mx <= 0):
            continue
        if (not allow_empty) and (mn <= 0):
            mn = 1
        out.append(
            {
                "name": str(name),
                "min_objects": int(mn),
                "max_objects": int(mx),
                "weight": float(max(0.0, weight)),
            }
        )
    return out


def _resolve_scene_density_config(cfg: dict, obj_min: int, obj_max: int) -> dict:
    scfg = cfg.get("scene_density", {})
    if not isinstance(scfg, dict):
        scfg = {}
    enabled = bool(scfg.get("enabled", False))
    allow_empty = bool(scfg.get("allow_empty", False))
    prior_path_str = str(scfg.get("prior_path", "")).strip()

    prior_obj = {}
    if prior_path_str:
        prior_obj = _load_yaml_or_json(_resolve_local_path(prior_path_str))

    prior_modes = {}
    if isinstance(prior_obj.get("default_sampling"), dict):
        prior_modes = prior_obj.get("default_sampling", {}).get("modes", {}) or {}
    if (not prior_modes) and isinstance(prior_obj.get("density_modes"), dict):
        prior_modes = prior_obj.get("density_modes", {})

    modes = _normalize_density_modes(prior_modes, obj_min, obj_max, allow_empty)
    merged = {m["name"]: dict(m) for m in modes}
    raw_mode_overrides = scfg.get("modes", {})
    if isinstance(raw_mode_overrides, dict):
        for name, rec in raw_mode_overrides.items():
            base = dict(
                merged.get(
                    str(name),
                    {
                        "name": str(name),
                        "min_objects": int(max(0 if allow_empty else 1, obj_min)),
                        "max_objects": int(max(obj_max, obj_min)),
                        "weight": 1.0,
                    },
                )
            )
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                mn, mx = int(rec[0]), int(rec[1])
                base["min_objects"] = int(max(0, mn))
                base["max_objects"] = int(max(base["min_objects"], mx))
            elif isinstance(rec, dict):
                rr = rec.get("range", None)
                if isinstance(rr, (list, tuple)) and len(rr) >= 2:
                    mn, mx = int(rr[0]), int(rr[1])
                    base["min_objects"] = int(max(0, mn))
                    base["max_objects"] = int(max(base["min_objects"], mx))
                else:
                    if "min_objects" in rec or "min" in rec:
                        base["min_objects"] = int(max(0, rec.get("min_objects", rec.get("min", base["min_objects"]))))
                    if "max_objects" in rec or "max" in rec:
                        base["max_objects"] = int(
                            max(base["min_objects"], rec.get("max_objects", rec.get("max", base["max_objects"])))
                        )
                if any(k in rec for k in ["weight", "non_empty_fraction", "sampling_weight", "image_fraction"]):
                    base["weight"] = float(
                        rec.get(
                            "weight",
                            rec.get(
                                "non_empty_fraction",
                                rec.get("sampling_weight", rec.get("image_fraction", base["weight"])),
                            ),
                        )
                    )
            if (not allow_empty) and int(base["max_objects"]) <= 0:
                continue
            if (not allow_empty) and int(base["min_objects"]) <= 0:
                base["min_objects"] = 1
            merged[str(name)] = base
    modes = list(merged.values())

    mode_weights = scfg.get("mode_weights", {})
    if isinstance(mode_weights, dict):
        for rec in modes:
            if rec["name"] in mode_weights:
                rec["weight"] = float(max(0.0, mode_weights[rec["name"]]))

    if len(modes) == 0:
        modes = [
            {
                "name": "uniform",
                "min_objects": int(max(0 if allow_empty else 1, obj_min)),
                "max_objects": int(max(obj_max, obj_min)),
                "weight": 1.0,
            }
        ]
        enabled = False

    weight_sum = float(sum(float(m["weight"]) for m in modes))
    if weight_sum <= 0.0:
        for rec in modes:
            rec["weight"] = 1.0
        weight_sum = float(len(modes))

    for rec in modes:
        rec["weight"] = float(rec["weight"]) / float(max(weight_sum, 1e-8))

    return {
        "enabled": bool(enabled),
        "allow_empty": bool(allow_empty),
        "prior_path": str(_resolve_local_path(prior_path_str)) if prior_path_str else "",
        "source_name": str(prior_obj.get("dataset_name", prior_obj.get("source_annotation", ""))),
        "modes": modes,
    }


def _sample_scene_density(scene_density_cfg: dict) -> Tuple[str, int]:
    modes = scene_density_cfg.get("modes", [])
    if len(modes) == 0:
        return "uniform", 0
    weights = [float(m.get("weight", 1.0)) for m in modes]
    idx = random.choices(list(range(len(modes))), weights=weights, k=1)[0]
    rec = modes[int(idx)]
    mn = int(rec.get("min_objects", 1))
    mx = int(rec.get("max_objects", mn))
    if mx < mn:
        mx = mn
    n_obj = random.randint(mn, mx)
    return str(rec.get("name", "uniform")), int(n_obj)


def _basic_count_stats(values: List[int]) -> dict:
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": int(np.max(arr)),
    }


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return float(inter) / float(max(1, ua))


def make_feather_alpha(h: int, w: int, feather: int) -> np.ndarray:
    if feather <= 0:
        return np.ones((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    d_left = xx
    d_right = (w - 1) - xx
    d_top = yy
    d_bottom = (h - 1) - yy
    d = np.minimum(np.minimum(d_left, d_right), np.minimum(d_top, d_bottom)).astype(np.float32)
    return np.clip(d / float(feather), 0.0, 1.0)


def weighted_choice(items: List[str], weights: List[float]) -> str:
    s = float(sum(weights))
    if s <= 0:
        return random.choice(items)
    r = random.random() * s
    acc = 0.0
    for it, w in zip(items, weights):
        acc += float(w)
        if r <= acc:
            return it
    return items[-1]


def paste_bgr(canvas_bgr: np.ndarray, patch_bgr: np.ndarray, x: int, y: int, alpha: np.ndarray):
    ph, pw = patch_bgr.shape[:2]
    roi = canvas_bgr[y : y + ph, x : x + pw].astype(np.float32)
    p = patch_bgr.astype(np.float32)
    a = alpha[..., None]
    out = roi * (1.0 - a) + p * a
    canvas_bgr[y : y + ph, x : x + pw] = np.clip(out, 0, 255).astype(np.uint8)


def paste_gray(canvas_u8: np.ndarray, patch_u8: np.ndarray, x: int, y: int, alpha: np.ndarray):
    ph, pw = patch_u8.shape[:2]
    roi = canvas_u8[y : y + ph, x : x + pw].astype(np.float32)
    p = patch_u8.astype(np.float32)
    out = roi * (1.0 - alpha) + p * alpha
    canvas_u8[y : y + ph, x : x + pw] = np.clip(out, 0, 255).astype(np.uint8)


def resize_patch(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


@dataclass
class PatchItem:
    cls_name: str
    rgb_path: Path
    tir_path: Path


@dataclass
class BgItem:
    rgb_path: Path
    tir_path: Path


def list_bg_pairs(bg_root: Path) -> List[BgItem]:
    rgb_dir = bg_root / "rgb"
    tir_dir = bg_root / "tir"
    if not rgb_dir.exists() or not tir_dir.exists():
        raise FileNotFoundError(f"background_pool_root must contain rgb/ and tir/. Got: {bg_root}")
    rgb_files = list(rgb_dir.rglob("*.png")) + list(rgb_dir.rglob("*.jpg")) + list(rgb_dir.rglob("*.jpeg"))
    out = []
    for rp in rgb_files:
        tp = tir_dir / rp.name
        if tp.exists():
            out.append(BgItem(rgb_path=rp, tir_path=tp))
    return out


def list_patch_pairs(patch_bank_root: Path) -> List[PatchItem]:
    rgb_root = patch_bank_root / "rgb"
    tir_root = patch_bank_root / "tir"
    if not rgb_root.exists() or not tir_root.exists():
        raise FileNotFoundError(f"patch_bank_root must contain rgb/ and tir/. Got: {patch_bank_root}")

    out: List[PatchItem] = []
    for cls_dir in sorted([p for p in rgb_root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        rgb_files = list(cls_dir.rglob("*.png")) + list(cls_dir.rglob("*.jpg")) + list(cls_dir.rglob("*.jpeg"))
        for rp in rgb_files:
            rel = rp.relative_to(rgb_root)
            tp = tir_root / rel
            if tp.exists():
                out.append(PatchItem(cls_name=cls, rgb_path=rp, tir_path=tp))
    return out


def build_class_index(patches: List[PatchItem]) -> Tuple[List[str], Dict[str, List[PatchItem]]]:
    cls_to_items: Dict[str, List[PatchItem]] = {}
    for it in patches:
        cls_to_items.setdefault(it.cls_name, []).append(it)
    return sorted(cls_to_items.keys()), cls_to_items


def load_class_weights(cfg: dict, classes: List[str]) -> List[float]:
    w = cfg.get("class_weights", None)
    if not isinstance(w, dict):
        return [1.0] * len(classes)
    return [float(w.get(c, 1.0)) for c in classes]


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = float(max(1e-6, temperature))
    z = x / t
    z = z - np.max(z)
    e = np.exp(z)
    s = float(np.sum(e))
    if s <= 1e-12:
        return np.ones_like(x, dtype=np.float64) / max(1, x.size)
    return e / s


def _size_bucket_from_hw(h: int, w: int) -> str:
    m = int(max(h, w))
    if m <= 12:
        return "tiny"
    if m <= 24:
        return "small"
    if m <= 48:
        return "medium"
    return "large"


def _load_prototype_runtime(
    proto_bank_path: Path,
    patch_bank_root: Path,
    classes: List[str],
    cls_to_items: Dict[str, List[PatchItem]],
) -> Optional[dict]:
    if not proto_bank_path.exists():
        print(f"[warn] proto_bank not found: {proto_bank_path}. fallback to random patch sampling.")
        return None
    bank_obj = np.load(str(proto_bank_path), allow_pickle=True)
    bank = bank_obj.item() if isinstance(bank_obj, np.ndarray) and bank_obj.shape == () else bank_obj
    if not isinstance(bank, dict):
        print("[warn] proto_bank format invalid. fallback to random patch sampling.")
        return None

    rgb_lookup: Dict[str, PatchItem] = {}
    for cls in classes:
        for it in cls_to_items.get(cls, []):
            rel = it.rgb_path.relative_to(patch_bank_root).as_posix()
            rgb_lookup[rel] = it

    class_to_proto = bank.get("class_to_prototypes", {})
    class_to_feat = bank.get("class_to_patch_features", {})
    class_to_rgb_rel = bank.get("class_to_patch_rgb_rel", {})
    class_to_bucket = bank.get("class_to_size_bucket", {})
    class_to_proto_ids = bank.get("class_to_patch_proto_id", {})
    class_to_counts = bank.get("class_to_counts", {})
    if not isinstance(class_to_proto, dict) or not isinstance(class_to_feat, dict) or not isinstance(class_to_rgb_rel, dict):
        print("[warn] proto_bank missing required keys. fallback to random patch sampling.")
        return None

    runtime = {"classes": classes, "class_data": {}, "class_mean": {}, "class_counts": {}}
    missing = []
    for cls in classes:
        p = np.asarray(class_to_proto.get(cls, []), dtype=np.float32)
        f = np.asarray(class_to_feat.get(cls, []), dtype=np.float32)
        rels = list(class_to_rgb_rel.get(cls, []))
        buckets = list(class_to_bucket.get(cls, []))
        proto_ids = list(class_to_proto_ids.get(cls, []))

        if p.ndim != 2 or p.shape[0] < 1 or f.ndim != 2 or f.shape[0] < 1 or len(rels) != int(f.shape[0]):
            missing.append(cls)
            continue

        keep_items: List[PatchItem] = []
        keep_feat = []
        keep_bucket = []
        keep_proto_id = []
        for i, rel in enumerate(rels):
            key = str(rel).replace("\\", "/")
            it = rgb_lookup.get(key, None)
            if it is None:
                continue
            keep_items.append(it)
            keep_feat.append(f[i])
            b = buckets[i] if i < len(buckets) else _size_bucket_from_hw(16, 16)
            keep_bucket.append(str(b))
            pid = int(proto_ids[i]) if i < len(proto_ids) else 0
            keep_proto_id.append(pid)

        if len(keep_items) == 0:
            missing.append(cls)
            continue

        feats = _normalize_rows(np.asarray(keep_feat, dtype=np.float32))
        protos = _normalize_rows(np.asarray(p, dtype=np.float32))
        cmean = _normalize_rows(np.mean(protos, axis=0, keepdims=True).astype(np.float32))[0]
        runtime["class_mean"][cls] = cmean
        runtime["class_counts"][cls] = int(class_to_counts.get(cls, len(keep_items)))

        bucket_freq: Dict[str, int] = {}
        for b in keep_bucket:
            bucket_freq[b] = bucket_freq.get(b, 0) + 1
        bucket_rarity = np.asarray([1.0 / max(1, bucket_freq[b]) for b in keep_bucket], dtype=np.float32)
        bucket_rarity = bucket_rarity / float(np.max(bucket_rarity)) if bucket_rarity.size > 0 else bucket_rarity

        runtime["class_data"][cls] = {
            "items": keep_items,
            "features": feats,
            "prototypes": protos,
            "size_bucket": keep_bucket,
            "bucket_rarity": bucket_rarity,
            "proto_ids": np.asarray(keep_proto_id, dtype=np.int32),
            "confuse": np.zeros((feats.shape[0],), dtype=np.float32),
        }

    if len(runtime["class_data"]) == 0:
        print("[warn] proto_bank has no valid class data after path match. fallback to random patch sampling.")
        return None
    if len(missing) > 0:
        print(f"[warn] proto classes unavailable and will fallback random: {missing}")

    valid_cls = list(runtime["class_data"].keys())
    means = np.stack([runtime["class_mean"][c] for c in valid_cls], axis=0)
    for cls in valid_cls:
        cd = runtime["class_data"][cls]
        feats = cd["features"]
        sims_all = feats @ means.T
        j = valid_cls.index(cls)
        if sims_all.shape[1] <= 1:
            confuse = np.zeros((feats.shape[0],), dtype=np.float32)
        else:
            other = np.delete(sims_all, j, axis=1)
            confuse = np.max(other, axis=1).astype(np.float32)
            confuse = np.clip((confuse + 1.0) * 0.5, 0.0, 1.0)
        cd["confuse"] = confuse

    print(f"[proto] loaded bank={proto_bank_path} usable_classes={len(runtime['class_data'])}/{len(classes)}")
    return runtime


def _choose_patch_with_prototype(
    cls: str,
    cls_to_items: Dict[str, List[PatchItem]],
    proto_rt: Optional[dict],
    proto_temperature: float,
    beta_confuse: float,
    alpha_imbalance: float,
    proto_candidate_pool: int,
) -> Tuple[PatchItem, Dict[str, float]]:
    fallback = random.choice(cls_to_items[cls])
    meta = {"proto_id": -1, "proto_sim": 0.0, "sampling_weight": 0.0, "candidate_pool_size": 0}
    if proto_rt is None:
        return fallback, meta

    cd = proto_rt["class_data"].get(cls, None)
    if cd is None:
        return fallback, meta

    feats = cd["features"]
    protos = cd["prototypes"]
    items = cd["items"]
    if feats.shape[0] == 0 or protos.shape[0] == 0:
        return fallback, meta

    if protos.shape[0] > 1:
        proto_id = int(np.random.randint(0, protos.shape[0]))
    else:
        proto_id = 0
    pvec = protos[proto_id]
    sims = (feats @ pvec).astype(np.float64)

    n = int(sims.shape[0])
    pool = int(max(1, min(n, int(proto_candidate_pool))))
    if pool < n:
        idx = np.argpartition(sims, -pool)[-pool:]
    else:
        idx = np.arange(n, dtype=np.int64)

    ss = sims[idx]
    w = _softmax(ss, temperature=float(proto_temperature))
    if abs(float(beta_confuse)) > 1e-8:
        w = w * (1.0 + float(beta_confuse) * cd["confuse"][idx].astype(np.float64))
    if abs(float(alpha_imbalance)) > 1e-8:
        rarity = np.power(np.maximum(1e-6, cd["bucket_rarity"][idx].astype(np.float64)), float(alpha_imbalance))
        w = w * rarity
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        w = np.ones_like(w) / max(1, w.size)
    else:
        w = w / w_sum

    choose_j = int(np.random.choice(np.arange(idx.shape[0]), p=w))
    real_i = int(idx[choose_j])
    it = items[real_i]
    meta = {
        "proto_id": int(proto_id),
        "proto_sim": float(sims[real_i]),
        "sampling_weight": float(w[choose_j]),
        "candidate_pool_size": int(pool),
    }
    return it, meta


def repair_tir_by_generator(
    gen: SiameseWKANAE,
    device: torch.device,
    use_amp: bool,
    canvas_bgr: np.ndarray,
    canvas_tir: np.ndarray,
    placed: List[Tuple[int, int, int, int, int]],
    gen_patch_size: int,
    gen_context: int,
):
    h, w = canvas_tir.shape[:2]
    tir_out = canvas_tir.copy()
    for (x, y, bw, bh, _cat) in placed:
        x1 = clamp(x - gen_context, 0, w - 1)
        y1 = clamp(y - gen_context, 0, h - 1)
        x2 = clamp(x + bw + gen_context, 0, w)
        y2 = clamp(y + bh + gen_context, 0, h)

        crop_bgr = canvas_bgr[y1:y2, x1:x2]
        crop_tir = tir_out[y1:y2, x1:x2]
        if crop_bgr.size == 0 or crop_tir.size == 0:
            continue

        ps = int(gen_patch_size)
        crop_bgr_rs = cv2.resize(crop_bgr, (ps, ps), interpolation=cv2.INTER_LINEAR)
        crop_tir_rs = cv2.resize(crop_tir, (ps, ps), interpolation=cv2.INTER_LINEAR)
        crop_rgb_rs = cv2.cvtColor(crop_bgr_rs, cv2.COLOR_BGR2RGB)

        rgb_t = torch.from_numpy(crop_rgb_rs.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        tir_t = torch.from_numpy(crop_tir_rs.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                _rgb_hat, tir_hat = gen(rgb_t, tir_t)

        tir_patch = tir_hat[0, 0].float().clamp(0, 1).cpu().numpy()
        tir_patch_u8 = (tir_patch * 255.0 + 0.5).astype(np.uint8)
        tir_patch_back = cv2.resize(tir_patch_u8, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

        bx1, by1 = x, y
        bx2, by2 = x + bw, y + bh
        rx1, ry1 = bx1 - x1, by1 - y1
        rx2, ry2 = bx2 - x1, by2 - y1
        rx1 = clamp(rx1, 0, tir_patch_back.shape[1])
        rx2 = clamp(rx2, 0, tir_patch_back.shape[1])
        ry1 = clamp(ry1, 0, tir_patch_back.shape[0])
        ry2 = clamp(ry2, 0, tir_patch_back.shape[0])
        tir_out[by1:by2, bx1:bx2] = tir_patch_back[ry1:ry2, rx1:rx2]
    return tir_out


def _get_nested(cfg: dict, path: List[str], default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _cfg_to_constraint_config(cfg: dict) -> Tuple[ConstraintConfig, dict, dict]:
    c = cfg.get("constraints", {}) or {}
    sw = cfg.get("switches", {}) or {}
    th = cfg.get("thresholds", {}) or {}
    ccfg = ConstraintConfig()

    use_semantic = bool(cfg.get("use_semantic_gate", sw.get("use_semantic_gate", c.get("enable_semantic_gate", True))))
    use_photometric = bool(cfg.get("use_photometric", sw.get("use_photometric", c.get("enable_photometric", True))))
    use_context_place = bool(
        cfg.get("use_context_place", sw.get("use_context_place", c.get("enable_context_aware", True)))
    )
    use_prototype_sampling = bool(cfg.get("use_prototype_sampling", sw.get("use_prototype_sampling", False)))
    use_wavelet_refiner = bool(cfg.get("use_wavelet_refiner", sw.get("use_wavelet_refiner", False)))
    use_legal_region_sampling = bool(cfg.get("use_legal_region_sampling", sw.get("use_legal_region_sampling", True)))

    ccfg.enable_semantic_gate = use_semantic
    ccfg.enable_photometric = use_photometric
    ccfg.enable_context_aware = use_context_place
    ccfg.allow_unknown_scene = bool(c.get("allow_unknown_scene", cfg.get("allow_unknown_scene", True)))
    ccfg.class_to_context = c.get("class_to_context", cfg.get("class_to_context", None))
    ccfg.context_ratio = float(c.get("context_ratio", cfg.get("context_ratio", 2.5)))
    ccfg.tau_ctx = float(c.get("tau_ctx", th.get("tau_ctx", cfg.get("tau_ctx", 0.55))))
    ccfg.hard_prior_tau = float(c.get("hard_prior_tau", th.get("hard_prior_tau", cfg.get("hard_prior_tau", 0.30))))
    ccfg.score_place_min = float(c.get("score_place_min", th.get("score_place_min", cfg.get("score_place_min", 0.60))))

    ccfg.lab_match_strength = float(c.get("lab_match_strength", th.get("lab_match_strength", cfg.get("lab_match_strength", 1.0))))
    ccfg.max_l_gap = float(c.get("max_l_gap", th.get("max_l_gap", cfg.get("max_l_gap", 40.0))))
    ccfg.max_ab_gap = float(c.get("max_ab_gap", th.get("max_ab_gap", cfg.get("max_ab_gap", 35.0))))
    ccfg.max_l_gap_pre = float(c.get("max_l_gap_pre", th.get("max_l_gap_pre", cfg.get("max_l_gap_pre", 140.0))))
    ccfg.max_ab_gap_pre = float(c.get("max_ab_gap_pre", th.get("max_ab_gap_pre", cfg.get("max_ab_gap_pre", 45.0))))
    ccfg.day_v_threshold = float(c.get("day_v_threshold", th.get("day_v_threshold", cfg.get("day_v_threshold", 112.0))))
    ccfg.night_v_threshold = float(c.get("night_v_threshold", th.get("night_v_threshold", cfg.get("night_v_threshold", 80.0))))
    ccfg.max_l_gap_day = float(c.get("max_l_gap_day", th.get("max_l_gap_day", cfg.get("max_l_gap_day", 36.0))))
    ccfg.max_ab_gap_day = float(c.get("max_ab_gap_day", th.get("max_ab_gap_day", cfg.get("max_ab_gap_day", 30.0))))
    ccfg.max_l_gap_night = float(c.get("max_l_gap_night", th.get("max_l_gap_night", cfg.get("max_l_gap_night", 48.0))))
    ccfg.max_ab_gap_night = float(c.get("max_ab_gap_night", th.get("max_ab_gap_night", cfg.get("max_ab_gap_night", 40.0))))
    ccfg.max_l_gap_twilight = float(c.get("max_l_gap_twilight", th.get("max_l_gap_twilight", cfg.get("max_l_gap_twilight", 42.0))))
    ccfg.max_ab_gap_twilight = float(c.get("max_ab_gap_twilight", th.get("max_ab_gap_twilight", cfg.get("max_ab_gap_twilight", 35.0))))
    ccfg.max_edge_density = float(c.get("max_edge_density", cfg.get("max_edge_density", ccfg.max_edge_density)))
    ccfg.max_texture = float(c.get("max_texture", cfg.get("max_texture", ccfg.max_texture)))
    ccfg.feather = int(c.get("feather", cfg.get("feather", ccfg.feather)))

    switches = {
        "use_semantic_gate": use_semantic,
        "use_photometric": use_photometric,
        "use_context_place": use_context_place,
        "use_prototype_sampling": use_prototype_sampling,
        "use_wavelet_refiner": use_wavelet_refiner,
        "use_legal_region_sampling": use_legal_region_sampling,
    }
    thresholds = {
        "tau_ctx": float(ccfg.tau_ctx),
        "hard_prior_tau": float(ccfg.hard_prior_tau),
        "score_place_min": float(ccfg.score_place_min),
        "day_v_threshold": float(ccfg.day_v_threshold),
        "night_v_threshold": float(ccfg.night_v_threshold),
        "max_l_gap_day": float(ccfg.max_l_gap_day),
        "max_ab_gap_day": float(ccfg.max_ab_gap_day),
        "max_l_gap_pre": float(ccfg.max_l_gap_pre),
        "max_ab_gap_pre": float(ccfg.max_ab_gap_pre),
        "max_l_gap_night": float(ccfg.max_l_gap_night),
        "max_ab_gap_night": float(ccfg.max_ab_gap_night),
        "max_l_gap_twilight": float(ccfg.max_l_gap_twilight),
        "max_ab_gap_twilight": float(ccfg.max_ab_gap_twilight),
    }
    return ccfg, switches, thresholds


def _dominant_reason(rej_iou: int, rej_semantic: int, rej_context: int, rej_photo: int) -> str:
    pairs = [
        ("reject_iou", rej_iou),
        ("reject_semantic", rej_semantic),
        ("reject_context_place", rej_context),
        ("reject_photometric", rej_photo),
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    if pairs[0][1] <= 0:
        return "no_valid_trial"
    return pairs[0][0]


_HARD_SEMANTIC_CLASSES = {"ship", "boat", "plane", "aircraft", "airplane", "helicopter"}


def _build_legal_semantic_positions(
    canvas_bgr: np.ndarray,
    cls_name: str,
    box_w: int,
    box_h: int,
    ccfg: ConstraintConfig,
    context_rules: dict,
    probe_trials: int = 480,
    max_pool: int = 160,
) -> List[Tuple[int, int, str, float]]:
    h, w = canvas_bgr.shape[:2]
    if box_w >= w or box_h >= h:
        return []
    out: List[Tuple[int, int, str, float]] = []
    for _ in range(int(max(1, probe_trials))):
        x = random.randint(0, w - box_w)
        y = random.randint(0, h - box_h)
        roi_bgr = canvas_bgr[y : y + box_h, x : x + box_w]
        if roi_bgr.size == 0:
            continue
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        ctx_label, ctx_conf, ok = semantic_gate(
            cls_name=cls_name,
            roi_rgb_u8=roi_rgb,
            bbox_xywh=(x, y, box_w, box_h),
            context_rules=context_rules,
            cfg=ccfg,
        )
        if ok:
            out.append((x, y, ctx_label, float(ctx_conf)))
            if len(out) >= int(max_pool):
                break
    random.shuffle(out)
    return out


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
            # If current GPU arch is not in this torch build, CUDA kernels will fail at runtime.
            if len(archs) > 0 and sm not in archs:
                print(f"[warn] Torch CUDA arch list does not include {sm} (supported={sorted(archs)}). Fallback to CPU.")
                return torch.device("cpu")
        except Exception as e:
            print(f"[warn] CUDA capability check failed ({e}). Keep requested CUDA.")
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/synthesis_v25lite.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seed_mode = str(cfg.get("seed_mode", "fixed")).strip().lower()
    base_seed = int(cfg.get("seed", cfg.get("seed_base", 42)))
    if seed_mode == "sweep":
        shard_id = int(cfg.get("seed_shard", cfg.get("shard_id", 0)))
        seed = int(base_seed + shard_id)
    else:
        seed = int(base_seed)
    set_seed(seed)

    device = _resolve_runtime_device(str(cfg.get("device", "cuda")))
    use_amp = bool(cfg.get("amp", True)) and (device.type == "cuda")

    patch_bank_root = Path(cfg["patch_bank_root"])
    bg_root = Path(cfg["background_pool_root"])
    out_root = Path(cfg["out_root"])
    context_rules_path = Path(cfg.get("context_rules_path", "configs/context_rules.yaml"))

    append_run_id = bool(cfg.get("append_run_id", True))
    run_id = now_id()
    run_dir = (out_root / run_id) if append_run_id else out_root

    rgb_out_dir = run_dir / "images" / "rgb"
    tir_out_dir = run_dir / "images" / "tir"
    ann_dir = run_dir / "annotations"
    prev_dir = run_dir / "previews"
    tab_dir = run_dir / "tables"
    meta_dir = run_dir / "meta"
    cfg_dir = run_dir / "configs"
    for d in [rgb_out_dir, tir_out_dir, ann_dir, prev_dir, tab_dir, meta_dir, cfg_dir]:
        ensure_dir(d)

    (cfg_dir / cfg_path.name).write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    mode = str(cfg.get("mode", "translate"))
    n_images = int(cfg.get("n_images", 2000))
    canvas_w = int(cfg.get("canvas_w", 640))
    canvas_h = int(cfg.get("canvas_h", 512))
    obj_min, obj_max = cfg.get("objects_per_image", [1, 8])
    obj_min, obj_max = int(obj_min), int(obj_max)
    scene_density_cfg = _resolve_scene_density_config(cfg, obj_min=obj_min, obj_max=obj_max)
    max_iou = float(cfg.get("max_iou", 0.10))
    max_trials = int(_get_nested(cfg, ["thresholds", "max_trials"], cfg.get("max_trials", cfg.get("max_place_trials", 50))))
    scale_min, scale_max = cfg.get("scale_range", [0.7, 1.4])
    scale_min, scale_max = float(scale_min), float(scale_max)

    feather_cfg = int(cfg.get("feather", 4))
    gen_patch_size = int(cfg.get("gen_patch_size", 64))
    gen_context = int(cfg.get("gen_context", 8))
    image_ext = str(cfg.get("image_ext", "jpg")).lower()
    jpeg_quality = int(cfg.get("jpeg_quality", 95))
    save_preview_every = int(cfg.get("save_preview_every", 0))
    refiner_cfg = {
        "wavelet_refiner_steps": int(cfg.get("wavelet_refiner_steps", 2)),
        "wavelet_name": str(cfg.get("wavelet_name", "haar")),
        "wavelet_refiner_mode": str(cfg.get("wavelet_refiner_mode", "analytic")),
        "wavelet_refiner_strength": float(cfg.get("wavelet_refiner_strength", 0.8)),
    }

    ccfg, switches, thresholds = _cfg_to_constraint_config(cfg)
    blend_feather = int(feather_cfg if "feather" in cfg else ccfg.feather)

    context_rules = {}
    if context_rules_path.exists():
        context_rules = yaml.safe_load(context_rules_path.read_text(encoding="utf-8")) or {}
    elif isinstance(ccfg.class_to_context, dict):
        context_rules = {"class_to_context": ccfg.class_to_context}

    bg_pairs = list_bg_pairs(bg_root)
    patches = list_patch_pairs(patch_bank_root)
    classes, cls_to_items = build_class_index(patches)
    cls_weights = load_class_weights(cfg, classes)

    proto_bank_path = Path(cfg.get("proto_bank_path", "runs/prototypes/latest/prototypes.npy"))
    proto_temperature = float(cfg.get("proto_temperature", 0.07))
    beta_confuse = float(cfg.get("beta_confuse", 1.0))
    alpha_imbalance = float(cfg.get("alpha_imbalance", 1.0))
    proto_candidate_pool = int(cfg.get("proto_candidate_pool", 256))
    proto_rt = None
    if switches["use_prototype_sampling"]:
        proto_rt = _load_prototype_runtime(
            proto_bank_path=proto_bank_path,
            patch_bank_root=patch_bank_root,
            classes=classes,
            cls_to_items=cls_to_items,
        )

    # class-level long-tail compensation (Phase-B): upweight classes with fewer prototype samples.
    if switches["use_prototype_sampling"] and proto_rt is not None and abs(alpha_imbalance) > 1e-8:
        counts = [int(proto_rt["class_counts"].get(c, max(1, len(cls_to_items.get(c, []))))) for c in classes]
        cmax = max(1, max(counts))
        cls_weights = [
            float(w) * float((float(cmax) / float(max(1, cnt))) ** float(alpha_imbalance))
            for w, cnt in zip(cls_weights, counts)
        ]

    print(f"[Step3] mode={mode} n_images={n_images} run_dir={run_dir}")
    print(f"        classes={len(classes)} bg_pairs={len(bg_pairs)} patches_total={len(patches)}")
    print(
        f"        switches: semantic={switches['use_semantic_gate']} photometric={switches['use_photometric']} "
        f"context_place={switches['use_context_place']} prototype={switches['use_prototype_sampling']} "
        f"wavelet_refiner={switches['use_wavelet_refiner']} legal_region={switches['use_legal_region_sampling']}"
    )
    print(
        f"        thresholds: tau_ctx={thresholds['tau_ctx']:.2f} score_place_min={thresholds['score_place_min']:.2f} "
        f"max_trials={max_trials}"
    )
    print(f"        runtime: device={device.type} amp={use_amp} seed_mode={seed_mode} seed={seed}")
    if scene_density_cfg["enabled"]:
        density_desc = ", ".join(
            [
                f"{m['name']}[{m['min_objects']},{m['max_objects']}]={m['weight']:.3f}"
                for m in scene_density_cfg["modes"]
            ]
        )
        print(f"        scene_density: enabled prior={scene_density_cfg['prior_path'] or 'inline'}")
        print(f"        scene_density_modes: {density_desc}")
    if switches["use_prototype_sampling"]:
        print(
            f"        proto: enabled={proto_rt is not None} bank={proto_bank_path} "
            f"temp={proto_temperature} beta_confuse={beta_confuse} alpha_imbalance={alpha_imbalance}"
        )
    if switches["use_wavelet_refiner"]:
        print(
            f"        wavelet_refiner: mode={refiner_cfg['wavelet_refiner_mode']} "
            f"steps={refiner_cfg['wavelet_refiner_steps']} strength={refiner_cfg['wavelet_refiner_strength']} "
            f"wavelet={refiner_cfg['wavelet_name']}"
        )

    if len(bg_pairs) == 0:
        raise RuntimeError("No background pairs found. Check background_pool_root.")
    if len(patches) == 0:
        raise RuntimeError("No patch pairs found. Check patch_bank_root.")

    gen = None
    if mode == "translate":
        ckpt = Path(cfg["generator_ckpt"])
        if not ckpt.exists():
            raise FileNotFoundError(f"generator_ckpt not found: {ckpt}")
        gen = SiameseWKANAE(base_ch=32, latent_ch=256).to(device)
        sd = torch.load(str(ckpt), map_location="cpu")
        gen.load_state_dict(sd, strict=True)
        gen.eval()

    coco = {
        "info": {
            "description": "Co-EvoNet Synthetic Dataset",
            "version": "2.5-lite",
            "year": 2026,
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "run_id": run_id,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    cat_id_map = {c: i + 1 for i, c in enumerate(classes)}
    coco["categories"] = [{"id": cat_id_map[c], "name": c, "supercategory": "object"} for c in classes]

    total_trials = 0
    reject_iou = 0
    reject_semantic = 0
    reject_context = 0
    reject_photometric = 0
    max_trials_exceeded = 0
    accepted_targets = 0
    total_targets = 0

    sample_manifest_rows = []
    photometric_rows = []
    placement_rows = []
    semantic_trial_rows = []
    trials_per_target = []
    feature_history = {"grad_energy": [], "texture_var": [], "boundary_dist": []}
    proto_sim_accepted: List[float] = []
    per_class_accept_count: Dict[str, int] = {c: 0 for c in classes}
    sampled_density_mode_counts: Dict[str, int] = {str(m["name"]): 0 for m in scene_density_cfg["modes"]}
    sampled_target_budgets: List[int] = []
    accepted_target_counts_per_image: List[int] = []
    image_density_rows: List[List] = []
    wavelet_gain_accepted: List[float] = []
    wavelet_energy_before_accepted: List[float] = []
    wavelet_energy_after_accepted: List[float] = []
    wavelet_edge_gain_accepted: List[float] = []

    ann_id = 1
    sample_id = 0

    for img_idx in range(n_images):
        bg = random.choice(bg_pairs)
        bg_bgr = read_img_bgr(bg.rgb_path)
        bg_tir = read_img_gray(bg.tir_path)
        if bg_bgr is None or bg_tir is None:
            continue

        bg_bgr = cv2.resize(bg_bgr, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
        bg_tir = cv2.resize(bg_tir, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)

        canvas_bgr = bg_bgr.copy()
        canvas_tir = bg_tir.copy()
        density_mode, n_obj = _sample_scene_density(scene_density_cfg)
        sampled_density_mode_counts[density_mode] = int(sampled_density_mode_counts.get(density_mode, 0)) + 1
        sampled_target_budgets.append(int(n_obj))
        placed: List[Tuple[int, int, int, int, int]] = []
        anns_for_img = []

        for obj_idx in range(1, n_obj + 1):
            total_targets += 1
            sample_id += 1
            cls = weighted_choice(classes, cls_weights)
            if switches["use_prototype_sampling"]:
                it, proto_meta = _choose_patch_with_prototype(
                    cls=cls,
                    cls_to_items=cls_to_items,
                    proto_rt=proto_rt,
                    proto_temperature=proto_temperature,
                    beta_confuse=beta_confuse,
                    alpha_imbalance=alpha_imbalance,
                    proto_candidate_pool=proto_candidate_pool,
                )
            else:
                it = random.choice(cls_to_items[cls])
                proto_meta = {"proto_id": -1, "proto_sim": 0.0, "sampling_weight": 0.0, "candidate_pool_size": 0}

            p_bgr = read_img_bgr(it.rgb_path)
            p_tir = read_img_gray(it.tir_path)
            if p_bgr is None or p_tir is None:
                sample_manifest_rows.append(
                    [
                        sample_id, img_idx + 1, obj_idx, cls,
                        density_mode, int(n_obj),
                        int(proto_meta["proto_id"]), float(proto_meta["proto_sim"]), float(proto_meta["sampling_weight"]), int(proto_meta["candidate_pool_size"]),
                        str(it.rgb_path), str(it.tir_path), str(bg.rgb_path), str(bg.tir_path),
                        "dropped", "patch_read_failed", 0, max_trials, 0, 0, 0, 0, 0, "", 0.0, 0.0, 0.0, -1, -1, -1, -1,
                        0, "none", 0, 0.0, 0.0, 1.0,
                    ]
                )
                continue

            ph, pw = p_bgr.shape[:2]
            if ph < 2 or pw < 2:
                sample_manifest_rows.append(
                    [
                        sample_id, img_idx + 1, obj_idx, cls,
                        density_mode, int(n_obj),
                        int(proto_meta["proto_id"]), float(proto_meta["proto_sim"]), float(proto_meta["sampling_weight"]), int(proto_meta["candidate_pool_size"]),
                        str(it.rgb_path), str(it.tir_path), str(bg.rgb_path), str(bg.tir_path),
                        "dropped", "patch_too_small", 0, max_trials, 0, 0, 0, 0, 0, "", 0.0, 0.0, 0.0, -1, -1, -1, -1,
                        0, "none", 0, 0.0, 0.0, 1.0,
                    ]
                )
                continue

            s = random.uniform(scale_min, scale_max)
            new_w = max(2, int(round(pw * s)))
            new_h = max(2, int(round(ph * s)))
            if new_w >= canvas_w or new_h >= canvas_h:
                sample_manifest_rows.append(
                    [
                        sample_id, img_idx + 1, obj_idx, cls,
                        density_mode, int(n_obj),
                        int(proto_meta["proto_id"]), float(proto_meta["proto_sim"]), float(proto_meta["sampling_weight"]), int(proto_meta["candidate_pool_size"]),
                        str(it.rgb_path), str(it.tir_path), str(bg.rgb_path), str(bg.tir_path),
                        "dropped", "patch_scaled_out_of_canvas", 0, max_trials, 0, 0, 0, 0, 0, "", 0.0, 0.0, 0.0, -1, -1, -1, -1,
                        0, "none", 0, 0.0, 0.0, 1.0,
                    ]
                )
                continue

            p_bgr_rs = resize_patch(p_bgr, new_w, new_h)
            p_tir_rs = resize_patch(p_tir, new_w, new_h)
            p_rgb_rs = cv2.cvtColor(p_bgr_rs, cv2.COLOR_BGR2RGB)
            alpha = make_feather_alpha(new_h, new_w, blend_feather)

            obj_rej_iou = 0
            obj_rej_sem = 0
            obj_rej_ctx = 0
            obj_rej_photo = 0
            trials_used = 0
            accepted = False
            final_ctx_label = ""
            final_ctx_conf = 0.0
            final_phi_gap = 0.0
            final_score_place = 0.0
            final_x = final_y = -1
            final_wavelet_enabled = 0
            final_wavelet_mode = "none"
            final_wavelet_steps = 0
            final_wavelet_energy_before = 0.0
            final_wavelet_energy_after = 0.0
            final_wavelet_gain_ratio = 1.0
            legal_positions: List[Tuple[int, int, str, float]] = []
            cls_key = str(cls).strip().lower()
            if (
                switches["use_semantic_gate"]
                and switches.get("use_legal_region_sampling", True)
                and (cls_key in _HARD_SEMANTIC_CLASSES)
            ):
                legal_positions = _build_legal_semantic_positions(
                    canvas_bgr=canvas_bgr,
                    cls_name=cls,
                    box_w=new_w,
                    box_h=new_h,
                    ccfg=ccfg,
                    context_rules=context_rules,
                    probe_trials=int(cfg.get("legal_region_probe_trials", 480)),
                    max_pool=int(cfg.get("legal_region_pool_size", 160)),
                )

            for trial_id in range(1, max_trials + 1):
                trials_used = trial_id
                total_trials += 1

                pre_sem: Optional[Tuple[str, float, bool]] = None
                if len(legal_positions) > 0:
                    x, y, pre_label, pre_conf = legal_positions.pop()
                    pre_sem = (pre_label, float(pre_conf), True)
                else:
                    x = random.randint(0, canvas_w - new_w)
                    y = random.randint(0, canvas_h - new_h)
                bb = (x, y, new_w, new_h)

                too_much = False
                for (px, py, pw2, ph2, _cid) in placed:
                    if bbox_iou(bb, (px, py, pw2, ph2)) > max_iou:
                        too_much = True
                        break
                if too_much:
                    reject_iou += 1
                    obj_rej_iou += 1
                    continue

                roi_bgr = canvas_bgr[y : y + new_h, x : x + new_w]
                if roi_bgr.size == 0:
                    continue
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

                if switches["use_semantic_gate"]:
                    if pre_sem is not None:
                        ctx_label, ctx_conf, sem_ok = pre_sem
                    else:
                        ctx_label, ctx_conf, sem_ok = semantic_gate(
                            cls_name=cls,
                            roi_rgb_u8=roi_rgb,
                            bbox_xywh=bb,
                            context_rules=context_rules,
                            cfg=ccfg,
                        )
                else:
                    ctx_label, ctx_conf, sem_ok = "disabled", 1.0, True
                semantic_trial_rows.append([img_idx + 1, obj_idx, cls, ctx_label, float(ctx_conf), int(sem_ok)])
                if not sem_ok:
                    reject_semantic += 1
                    obj_rej_sem += 1
                    continue

                if switches["use_context_place"]:
                    score_place, score_stats = placement_score(
                        roi_rgb_u8=roi_rgb,
                        bbox_xywh=bb,
                        canvas_w=canvas_w,
                        canvas_h=canvas_h,
                        feature_history=feature_history,
                    )
                    feature_history["grad_energy"].append(float(score_stats["grad_energy"]))
                    feature_history["texture_var"].append(float(score_stats["texture_var"]))
                    feature_history["boundary_dist"].append(float(score_stats["boundary_dist"]))
                    score_ok = score_place >= float(ccfg.score_place_min)
                else:
                    score_place = 1.0
                    score_stats = {
                        "grad_energy": 0.0,
                        "texture_var": 0.0,
                        "boundary_dist": 0.0,
                        "quality_grad": 1.0,
                        "quality_texture": 1.0,
                        "quality_boundary": 1.0,
                    }
                    score_ok = True

                placement_rows.append(
                    [
                        img_idx + 1,
                        obj_idx,
                        cls,
                        trial_id,
                        float(score_place),
                        float(ccfg.score_place_min),
                        int(score_ok),
                        float(score_stats["grad_energy"]),
                        float(score_stats["texture_var"]),
                        float(score_stats["boundary_dist"]),
                        float(score_stats["quality_grad"]),
                        float(score_stats["quality_texture"]),
                        float(score_stats["quality_boundary"]),
                        x,
                        y,
                        new_w,
                        new_h,
                    ]
                )
                if not score_ok:
                    reject_context += 1
                    obj_rej_ctx += 1
                    continue

                if switches["use_photometric"]:
                    p_rgb_adj, p_stats, photo_ok = photometric_match_details(p_rgb_rs, roi_rgb, ccfg)
                    phi_gap = float(p_stats["phi_gap"])
                else:
                    p_rgb_adj = p_rgb_rs
                    p_stats = {
                        "l_gap_before": 0.0,
                        "ab_gap_before": 0.0,
                        "l_gap_after": 0.0,
                        "ab_gap_after": 0.0,
                        "max_l_gap": 0.0,
                        "max_ab_gap": 0.0,
                        "phi_gap": 0.0,
                        "day_phase": "disabled",
                    }
                    photo_ok = True
                    phi_gap = 0.0

                photometric_rows.append(
                    [
                        img_idx + 1,
                        obj_idx,
                        cls,
                        trial_id,
                        float(p_stats["l_gap_before"]),
                        float(p_stats["ab_gap_before"]),
                        float(p_stats["l_gap_after"]),
                        float(p_stats["ab_gap_after"]),
                        float(p_stats["max_l_gap"]),
                        float(p_stats["max_ab_gap"]),
                        float(p_stats.get("max_l_gap_pre", 0.0)),
                        float(p_stats.get("max_ab_gap_pre", 0.0)),
                        float(phi_gap),
                        str(p_stats["day_phase"]),
                        int(bool(p_stats.get("precheck_ok", True))),
                        int(photo_ok),
                        x,
                        y,
                        new_w,
                        new_h,
                    ]
                )
                if not photo_ok:
                    reject_photometric += 1
                    obj_rej_photo += 1
                    continue

                p_bgr_adj = cv2.cvtColor(p_rgb_adj, cv2.COLOR_RGB2BGR)
                tir_patch_final = p_tir_rs
                wstats = {
                    "wavelet_refiner_enabled": 0,
                    "wavelet_mode": "none",
                    "wavelet_steps": 0,
                    "wavelet_energy_before": 0.0,
                    "wavelet_energy_after": 0.0,
                    "wavelet_gain_ratio": 1.0,
                    "tir_edge_energy_gain": 1.0,
                }
                if switches["use_wavelet_refiner"]:
                    mask_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                    tir_patch_final, wstats = wavelet_refine_tir(
                        rgb_patch_u8=p_rgb_adj,
                        tir_patch_u8=p_tir_rs,
                        mask_u8=mask_u8,
                        cfg=refiner_cfg,
                    )

                paste_bgr(canvas_bgr, p_bgr_adj, x, y, alpha)
                paste_gray(canvas_tir, tir_patch_final, x, y, alpha)

                cat_id = cat_id_map[cls]
                placed.append((x, y, new_w, new_h, cat_id))
                anns_for_img.append(
                    {
                        "id": None,
                        "image_id": img_idx + 1,
                        "category_id": cat_id,
                        "bbox": [int(x), int(y), int(new_w), int(new_h)],
                        "area": int(new_w * new_h),
                        "iscrowd": 0,
                    }
                )
                accepted_targets += 1
                per_class_accept_count[cls] = int(per_class_accept_count.get(cls, 0)) + 1
                if int(proto_meta.get("proto_id", -1)) >= 0:
                    proto_sim_accepted.append(float(proto_meta.get("proto_sim", 0.0)))
                accepted = True
                final_ctx_label = ctx_label
                final_ctx_conf = float(ctx_conf)
                final_phi_gap = float(phi_gap)
                final_score_place = float(score_place)
                final_x, final_y = int(x), int(y)
                final_wavelet_enabled = int(wstats.get("wavelet_refiner_enabled", 0))
                final_wavelet_mode = str(wstats.get("wavelet_mode", "none"))
                final_wavelet_steps = int(wstats.get("wavelet_steps", 0))
                final_wavelet_energy_before = float(wstats.get("wavelet_energy_before", 0.0))
                final_wavelet_energy_after = float(wstats.get("wavelet_energy_after", 0.0))
                final_wavelet_gain_ratio = float(wstats.get("wavelet_gain_ratio", 1.0))
                if final_wavelet_enabled:
                    wavelet_gain_accepted.append(float(final_wavelet_gain_ratio))
                    wavelet_energy_before_accepted.append(float(final_wavelet_energy_before))
                    wavelet_energy_after_accepted.append(float(final_wavelet_energy_after))
                    wavelet_edge_gain_accepted.append(float(wstats.get("tir_edge_energy_gain", 1.0)))
                break

            trials_per_target.append(int(trials_used))
            exceeded = int((not accepted) and (trials_used >= max_trials))
            if exceeded:
                max_trials_exceeded += 1
            status = "accepted" if accepted else "dropped"
            reason = "accepted" if accepted else _dominant_reason(obj_rej_iou, obj_rej_sem, obj_rej_ctx, obj_rej_photo)

            sample_manifest_rows.append(
                [
                    sample_id,
                    img_idx + 1,
                    obj_idx,
                    cls,
                    density_mode,
                    int(n_obj),
                    int(proto_meta["proto_id"]),
                    float(proto_meta["proto_sim"]),
                    float(proto_meta["sampling_weight"]),
                    int(proto_meta["candidate_pool_size"]),
                    str(it.rgb_path),
                    str(it.tir_path),
                    str(bg.rgb_path),
                    str(bg.tir_path),
                    status,
                    reason,
                    int(trials_used),
                    int(max_trials),
                    exceeded,
                    int(obj_rej_iou),
                    int(obj_rej_sem),
                    int(obj_rej_ctx),
                    int(obj_rej_photo),
                    final_ctx_label,
                    float(final_ctx_conf),
                    float(final_phi_gap),
                    float(final_score_place),
                    int(final_x),
                    int(final_y),
                    int(new_w),
                    int(new_h),
                    int(final_wavelet_enabled),
                    final_wavelet_mode,
                    int(final_wavelet_steps),
                    float(final_wavelet_energy_before),
                    float(final_wavelet_energy_after),
                    float(final_wavelet_gain_ratio),
                ]
            )

        if mode == "translate" and gen is not None and len(placed) > 0:
            canvas_tir = repair_tir_by_generator(
                gen=gen,
                device=device,
                use_amp=use_amp,
                canvas_bgr=canvas_bgr,
                canvas_tir=canvas_tir,
                placed=placed,
                gen_patch_size=gen_patch_size,
                gen_context=gen_context,
            )

        fname = f"{img_idx + 1:06d}.{image_ext}"
        save_img_bgr(rgb_out_dir / fname, canvas_bgr, ext=image_ext, quality=jpeg_quality)
        save_img_gray(tir_out_dir / fname, canvas_tir, ext=image_ext, quality=jpeg_quality)
        accepted_target_counts_per_image.append(int(len(anns_for_img)))
        image_density_rows.append(
            [
                int(img_idx + 1),
                str(density_mode),
                int(n_obj),
                int(len(anns_for_img)),
                float(len(anns_for_img) / max(1, n_obj)),
            ]
        )

        coco["images"].append({"id": img_idx + 1, "file_name": fname, "width": canvas_w, "height": canvas_h})
        for a in anns_for_img:
            a["id"] = ann_id
            ann_id += 1
            coco["annotations"].append(a)

        if save_preview_every > 0 and ((img_idx + 1) % save_preview_every == 0):
            vis = canvas_bgr.copy()
            for (x, y, w, h, cat_id) in placed:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(vis, str(cat_id), (x, max(0, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            save_img_bgr(prev_dir / f"preview_{img_idx + 1:06d}.jpg", vis, ext="jpg", quality=90)

        if (img_idx + 1) % 50 == 0:
            print(f"[{img_idx + 1}/{n_images}] saved {fname} | anns={len(anns_for_img)}")

    ann_path = ann_dir / "instances_synth.json"
    ann_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")

    write_csv(
        tab_dir / "sample_manifest.csv",
        [
            "sample_id",
            "image_id",
            "obj_idx",
            "class",
            "scene_density_mode",
            "scene_target_budget",
            "proto_id",
            "proto_sim",
            "sampling_weight",
            "candidate_pool_size",
            "patch_rgb_path",
            "patch_tir_path",
            "bg_rgb_path",
            "bg_tir_path",
            "status",
            "final_reason",
            "trials_used",
            "max_trials",
            "max_trials_exceeded",
            "reject_iou",
            "reject_semantic",
            "reject_context_place",
            "reject_photometric",
            "ctx_label",
            "ctx_conf",
            "phi_gap",
            "score_place",
            "x",
            "y",
            "w",
            "h",
            "wavelet_refiner_enabled",
            "wavelet_mode",
            "wavelet_steps",
            "wavelet_energy_before",
            "wavelet_energy_after",
            "wavelet_gain_ratio",
        ],
        sample_manifest_rows,
    )

    write_csv(
        tab_dir / "scene_density_stats.csv",
        [
            "image_id",
            "scene_density_mode",
            "scene_target_budget",
            "accepted_count",
            "accept_fill_ratio",
        ],
        image_density_rows,
    )

    rej_total = reject_iou + reject_semantic + reject_context + reject_photometric
    avg_trials = float(np.mean(trials_per_target)) if len(trials_per_target) > 0 else 0.0
    write_csv(
        tab_dir / "invalid_placement_rate.csv",
        [
            "total_targets",
            "accepted_targets",
            "accept_rate",
            "total_trials",
            "avg_trials_used",
            "max_trials",
            "max_trials_exceeded",
            "max_trials_exceeded_rate",
            "reject_iou",
            "reject_semantic",
            "reject_context_place",
            "reject_photometric",
            "total_reject",
        ],
        [
            [
                int(total_targets),
                int(accepted_targets),
                float(accepted_targets / max(1, total_targets)),
                int(total_trials),
                float(avg_trials),
                int(max_trials),
                int(max_trials_exceeded),
                float(max_trials_exceeded / max(1, total_targets)),
                int(reject_iou),
                int(reject_semantic),
                int(reject_context),
                int(reject_photometric),
                int(rej_total),
            ]
        ],
    )

    ctx_bins: Dict[str, Dict[str, float]] = {}
    for _img_id, _obj_id, _cls, ctx_label, ctx_conf, is_valid in semantic_trial_rows:
        rec = ctx_bins.setdefault(ctx_label, {"total": 0, "valid": 0, "invalid": 0, "conf_sum": 0.0})
        rec["total"] += 1
        rec["valid"] += int(is_valid)
        rec["invalid"] += int(not is_valid)
        rec["conf_sum"] += float(ctx_conf)
    ctx_hist_rows = []
    for k in sorted(ctx_bins.keys()):
        v = ctx_bins[k]
        ctx_hist_rows.append([k, int(v["total"]), int(v["valid"]), int(v["invalid"]), float(v["conf_sum"] / max(1, v["total"]))])
    write_csv(tab_dir / "ctx_hist.csv", ["ctx_label", "total", "valid", "invalid", "mean_conf"], ctx_hist_rows)

    write_csv(
        tab_dir / "photometric_gap.csv",
        [
            "image_id",
            "obj_idx",
            "class",
            "trial_id",
            "l_gap_before",
            "ab_gap_before",
            "l_gap_after",
            "ab_gap_after",
            "max_l_gap",
            "max_ab_gap",
            "max_l_gap_pre",
            "max_ab_gap_pre",
            "phi_gap",
            "day_phase",
            "precheck_ok",
            "is_acceptable",
            "x",
            "y",
            "w",
            "h",
        ],
        photometric_rows,
    )

    write_csv(
        tab_dir / "placement_score_stats.csv",
        [
            "image_id",
            "obj_idx",
            "class",
            "trial_id",
            "score_place",
            "score_place_min",
            "is_acceptable",
            "grad_energy",
            "texture_var",
            "boundary_dist",
            "quality_grad",
            "quality_texture",
            "quality_boundary",
            "x",
            "y",
            "w",
            "h",
        ],
        placement_rows,
    )

    if len(proto_sim_accepted) > 0:
        ps = np.asarray(proto_sim_accepted, dtype=np.float32)
        proto_sim_mean = float(np.mean(ps))
        proto_sim_p50 = float(np.quantile(ps, 0.50))
        proto_sim_p10 = float(np.quantile(ps, 0.10))
    else:
        proto_sim_mean = 0.0
        proto_sim_p50 = 0.0
        proto_sim_p10 = 0.0

    if len(wavelet_gain_accepted) > 0:
        wg = np.asarray(wavelet_gain_accepted, dtype=np.float32)
        wb = np.asarray(wavelet_energy_before_accepted, dtype=np.float32)
        wa = np.asarray(wavelet_energy_after_accepted, dtype=np.float32)
        we = np.asarray(wavelet_edge_gain_accepted, dtype=np.float32) if len(wavelet_edge_gain_accepted) > 0 else np.ones((1,), dtype=np.float32)
        wavelet_gain_mean = float(np.mean(wg))
        wavelet_gain_p50 = float(np.quantile(wg, 0.50))
        wavelet_gain_p10 = float(np.quantile(wg, 0.10))
        wavelet_energy_before_mean = float(np.mean(wb))
        wavelet_energy_after_mean = float(np.mean(wa))
        tir_edge_energy_gain_mean = float(np.mean(we))
    else:
        wavelet_gain_mean = 1.0
        wavelet_gain_p50 = 1.0
        wavelet_gain_p10 = 1.0
        wavelet_energy_before_mean = 0.0
        wavelet_energy_after_mean = 0.0
        tir_edge_energy_gain_mean = 1.0

    sampled_budget_stats = _basic_count_stats(sampled_target_budgets)
    accepted_per_image_stats = _basic_count_stats(accepted_target_counts_per_image)

    summary = {
        "run_id": run_id,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": str(cfg_path),
        "seed": {"mode": seed_mode, "base_seed": int(base_seed), "effective_seed": int(seed)},
        "switches": switches,
        "thresholds": {**thresholds, "max_trials": int(max_trials), "max_iou": float(max_iou)},
        "scene_density": {
            "enabled": bool(scene_density_cfg.get("enabled", False)),
            "allow_empty": bool(scene_density_cfg.get("allow_empty", False)),
            "prior_path": str(scene_density_cfg.get("prior_path", "")),
            "source_name": str(scene_density_cfg.get("source_name", "")),
            "modes": [
                {
                    "name": str(m["name"]),
                    "min_objects": int(m["min_objects"]),
                    "max_objects": int(m["max_objects"]),
                    "weight": float(m["weight"]),
                }
                for m in scene_density_cfg.get("modes", [])
            ],
        },
        "dataset": {
            "n_images": int(n_images),
            "canvas_w": int(canvas_w),
            "canvas_h": int(canvas_h),
            "mode": mode,
            "generator_ckpt": str(cfg.get("generator_ckpt", "")),
            "classes": classes,
        },
        "stats": {
            "total_targets": int(total_targets),
            "accepted_targets": int(accepted_targets),
            "accept_rate": float(accepted_targets / max(1, total_targets)),
            "total_trials": int(total_trials),
            "avg_trials_used": float(avg_trials),
            "max_trials_exceeded": int(max_trials_exceeded),
            "max_trials_exceeded_rate": float(max_trials_exceeded / max(1, total_targets)),
            "reject_iou": int(reject_iou),
            "reject_semantic": int(reject_semantic),
            "reject_context_place": int(reject_context),
            "reject_photometric": int(reject_photometric),
            "reject_total": int(rej_total),
            "images_written": int(len(coco["images"])),
            "annotations_written": int(len(coco["annotations"])),
            "proto_sampling_enabled": bool(switches.get("use_prototype_sampling", False) and (proto_rt is not None)),
            "proto_sim_mean": float(proto_sim_mean),
            "proto_sim_p50": float(proto_sim_p50),
            "proto_sim_p10": float(proto_sim_p10),
            "per_class_accept_count": {str(k): int(v) for k, v in per_class_accept_count.items()},
            "wavelet_enabled": bool(switches.get("use_wavelet_refiner", False)),
            "wavelet_gain_mean": float(wavelet_gain_mean),
            "wavelet_gain_p50": float(wavelet_gain_p50),
            "wavelet_gain_p10": float(wavelet_gain_p10),
            "wavelet_energy_before_mean": float(wavelet_energy_before_mean),
            "wavelet_energy_after_mean": float(wavelet_energy_after_mean),
            "tir_edge_energy_gain_mean": float(tir_edge_energy_gain_mean),
            "scene_density_enabled": bool(scene_density_cfg.get("enabled", False)),
            "scene_density_mode_counts": {str(k): int(v) for k, v in sampled_density_mode_counts.items()},
            "scene_target_budget_mean": float(sampled_budget_stats["mean"]),
            "scene_target_budget_p50": float(sampled_budget_stats["p50"]),
            "scene_target_budget_p90": float(sampled_budget_stats["p90"]),
            "scene_target_budget_max": int(sampled_budget_stats["max"]),
            "accepted_per_image_mean": float(accepted_per_image_stats["mean"]),
            "accepted_per_image_p50": float(accepted_per_image_stats["p50"]),
            "accepted_per_image_p90": float(accepted_per_image_stats["p90"]),
            "accepted_per_image_max": int(accepted_per_image_stats["max"]),
        },
        "outputs": {
            "coco_json": str(ann_path),
            "sample_manifest": str(tab_dir / "sample_manifest.csv"),
            "scene_density_stats": str(tab_dir / "scene_density_stats.csv"),
            "invalid_placement_rate": str(tab_dir / "invalid_placement_rate.csv"),
            "ctx_hist": str(tab_dir / "ctx_hist.csv"),
            "photometric_gap": str(tab_dir / "photometric_gap.csv"),
            "placement_score_stats": str(tab_dir / "placement_score_stats.csv"),
            "proto_bank_path": str(proto_bank_path) if switches.get("use_prototype_sampling", False) else "",
        },
    }
    save_json(meta_dir / "synthesis_summary.json", summary)
    save_json(meta_dir / "run_meta.json", summary)

    print(f"[OK] Step3 done. images={len(coco['images'])} anns={len(coco['annotations'])}")
    print(f"     run_dir={run_dir}")
    print(f"     coco={ann_path}")
    print(f"     summary={meta_dir / 'synthesis_summary.json'}")


if __name__ == "__main__":
    main()
