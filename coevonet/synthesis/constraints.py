# coevonet/synthesis/constraints.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np


SCENES_8 = [
    "sea",
    "lake",
    "bridge",
    "city road",
    "country road",
    "playground",
    "airport",
    "sky",
]


@dataclass
class ConstraintConfig:
    # semantic gate
    enable_semantic_gate: bool = True
    allow_unknown_scene: bool = True
    class_to_context: Optional[Dict[str, List[str]]] = None
    context_ratio: float = 2.5
    tau_ctx: float = 0.55
    hard_prior_tau: float = 0.30

    # photometric
    enable_photometric: bool = True
    lab_match_strength: float = 1.0
    max_l_gap: float = 40.0
    max_ab_gap: float = 35.0
    max_l_gap_pre: float = 140.0
    max_ab_gap_pre: float = 45.0
    day_v_threshold: float = 112.0
    night_v_threshold: float = 80.0
    max_l_gap_day: float = 36.0
    max_ab_gap_day: float = 30.0
    max_l_gap_night: float = 48.0
    max_ab_gap_night: float = 40.0
    max_l_gap_twilight: float = 42.0
    max_ab_gap_twilight: float = 35.0

    # context / placement
    enable_context_aware: bool = True
    max_edge_density: float = 0.22
    max_texture: float = 35.0
    score_place_min: float = 0.60

    # alpha feather
    feather: int = 2


def _ensure_3ch(rgb_u8: np.ndarray) -> np.ndarray:
    if rgb_u8.ndim == 2:
        return cv2.cvtColor(rgb_u8, cv2.COLOR_GRAY2RGB)
    return rgb_u8


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = x.std(axis=(0, 1), keepdims=False)
    s = np.maximum(s, 1.0)
    return s


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _norm01(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clip01((v - lo) / (hi - lo))


def expand_roi_xywh(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    ratio: float,
) -> Tuple[int, int, int, int]:
    ratio = float(max(1.0, ratio))
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    nw = max(2.0, w * ratio)
    nh = max(2.0, h * ratio)

    x0 = int(np.floor(cx - 0.5 * nw))
    y0 = int(np.floor(cy - 0.5 * nh))
    x1 = int(np.ceil(cx + 0.5 * nw))
    y1 = int(np.ceil(cy + 0.5 * nh))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def scene_scores_roi_8class(rgb_roi_u8: np.ndarray) -> Dict[str, float]:
    rgb = _ensure_3ch(rgb_roi_u8)
    if rgb.shape[0] < 8 or rgb.shape[1] < 8:
        return {k: 0.0 for k in SCENES_8}

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    mean_s = float(s.mean())
    mean_v = float(v.mean())

    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    r_m, g_m, b_m = float(r.mean()), float(g.mean()), float(b.mean())

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())
    texture = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    blue_bias = (b_m - r_m) / 255.0
    green_bias = (g_m - r_m) / 255.0

    scores = {k: 0.0 for k in SCENES_8}

    scores["sky"] = min(
        _norm01(mean_v, 125.0, 235.0),
        _norm01(90.0 - mean_s, 0.0, 90.0),
        _norm01(blue_bias, 0.02, 0.24),
    )
    scores["sea"] = min(
        _norm01(blue_bias, 0.03, 0.25),
        _norm01(mean_s, 20.0, 160.0),
        _norm01(mean_v, 70.0, 180.0),
        _norm01(0.18 - edge_density, 0.0, 0.18),
    )
    scores["lake"] = min(
        _norm01(blue_bias, 0.02, 0.20),
        _norm01(mean_s, 12.0, 140.0),
        _norm01(mean_v, 65.0, 170.0),
        _norm01(0.14 - edge_density, 0.0, 0.14),
        _norm01(70.0 - texture, 0.0, 70.0),
    )
    scores["bridge"] = min(
        _norm01(edge_density, 0.09, 0.33),
        _norm01(texture, 45.0, 220.0),
    )
    scores["city road"] = min(
        _norm01(edge_density, 0.07, 0.30),
        _norm01(texture, 25.0, 170.0),
        _norm01(140.0 - mean_s, 0.0, 140.0),
    )
    scores["country road"] = min(
        _norm01(edge_density, 0.06, 0.28),
        _norm01(texture, 28.0, 180.0),
        _norm01(green_bias, 0.02, 0.22),
    )
    scores["playground"] = min(
        _norm01(mean_s, 95.0, 220.0),
        _norm01(mean_v, 90.0, 220.0),
    )
    scores["airport"] = min(
        _norm01(mean_v, 120.0, 230.0),
        _norm01(110.0 - mean_s, 0.0, 110.0),
        _norm01(0.20 - edge_density, 0.0, 0.20),
    )

    ssum = float(sum(scores.values()))
    if ssum > 1e-6:
        scores = {k: float(v / ssum) for k, v in scores.items()}
    return scores


def scene_label_roi_8class(rgb_roi_u8: np.ndarray, min_conf: float = 0.15) -> str:
    scores = scene_scores_roi_8class(rgb_roi_u8)
    if not scores:
        return "unknown"
    label = max(scores, key=scores.get)
    conf = float(scores[label])
    if conf < float(min_conf):
        return "unknown"
    return label


def scene_label_with_conf(rgb_roi_u8: np.ndarray, min_conf: float = 0.15) -> Tuple[str, float]:
    scores = scene_scores_roi_8class(rgb_roi_u8)
    if not scores:
        return "unknown", 0.0
    label = max(scores, key=scores.get)
    conf = float(scores[label])
    if conf < float(min_conf):
        return "unknown", conf
    return label, conf


_HARD_CONTEXT_PRIORS = {
    "ship": {"sea", "lake"},
    "boat": {"sea", "lake"},
    "plane": {"sky", "airport"},
    "aircraft": {"sky", "airport"},
    "airplane": {"sky", "airport"},
    "helicopter": {"sky", "airport"},
}


def semantic_gate(
    cls_name: str,
    roi_rgb_u8: np.ndarray,
    bbox_xywh: Optional[Tuple[int, int, int, int]],
    context_rules: Optional[dict],
    cfg: ConstraintConfig,
) -> Tuple[str, float, bool]:
    label, conf = scene_label_with_conf(roi_rgb_u8)

    if not bool(cfg.enable_semantic_gate):
        return label, conf, True

    cls_key = str(cls_name).strip().lower()
    tau = float(cfg.tau_ctx)

    class_to_context = None
    if isinstance(context_rules, dict):
        class_to_context = context_rules.get("class_to_context", None)
    if not isinstance(class_to_context, dict):
        class_to_context = cfg.class_to_context if isinstance(cfg.class_to_context, dict) else {}

    allowed = class_to_context.get(cls_name, class_to_context.get(cls_key, None))
    allowed_set = set(str(x).strip().lower() for x in (allowed or []))
    label_key = str(label).strip().lower()

    if label_key == "unknown":
        return label, conf, bool(cfg.allow_unknown_scene)

    # hard priors only reject obvious impossible placements when confidence is high enough.
    hard_tau = float(min(tau, cfg.hard_prior_tau))
    if (cls_key in _HARD_CONTEXT_PRIORS) and (conf >= hard_tau):
        if label_key not in _HARD_CONTEXT_PRIORS[cls_key]:
            return label, conf, False

    if len(allowed_set) == 0:
        return label, conf, True

    if conf < tau:
        # low-confidence semantic prediction should not aggressively reject.
        return label, conf, True

    return label, conf, (label_key in allowed_set)


def semantic_gate_ok(
    cls_name: str,
    roi_rgb_u8: np.ndarray,
    cfg: ConstraintConfig,
) -> Tuple[bool, str]:
    label, _conf, ok = semantic_gate(
        cls_name=cls_name,
        roi_rgb_u8=roi_rgb_u8,
        bbox_xywh=None,
        context_rules=None,
        cfg=cfg,
    )
    return ok, label


def lab_match_patch_to_roi(
    patch_rgb_u8: np.ndarray,
    roi_rgb_u8: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6:
        return _ensure_3ch(patch_rgb_u8)

    patch = _ensure_3ch(patch_rgb_u8)
    roi = _ensure_3ch(roi_rgb_u8)

    patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)

    mp = patch_lab.mean(axis=(0, 1), keepdims=True)
    sp = _safe_std(patch_lab)
    mr = roi_lab.mean(axis=(0, 1), keepdims=True)
    sr = _safe_std(roi_lab)

    out = (patch_lab - mp) / sp.reshape(1, 1, 3) * sr.reshape(1, 1, 3) + mr
    out = np.clip(out, 0, 255)
    out = patch_lab * (1.0 - strength) + out * strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def photometric_gap(patch_rgb_u8: np.ndarray, roi_rgb_u8: np.ndarray) -> Tuple[float, float]:
    p = cv2.cvtColor(_ensure_3ch(patch_rgb_u8), cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(_ensure_3ch(roi_rgb_u8), cv2.COLOR_RGB2LAB).astype(np.float32)
    mp = p.mean(axis=(0, 1))
    mr = r.mean(axis=(0, 1))
    l_gap = float(abs(mp[0] - mr[0]))
    ab_gap = float(np.linalg.norm(mp[1:3] - mr[1:3]))
    return l_gap, ab_gap


def infer_day_phase(roi_rgb_u8: np.ndarray, cfg: ConstraintConfig) -> Tuple[str, float]:
    roi = _ensure_3ch(roi_rgb_u8)
    v_mean = float(cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)[..., 2].mean())
    if v_mean >= float(cfg.day_v_threshold):
        return "day", v_mean
    if v_mean <= float(cfg.night_v_threshold):
        return "night", v_mean
    return "twilight", v_mean


def _gaps_by_phase(phase: str, cfg: ConstraintConfig) -> Tuple[float, float]:
    if phase == "day":
        return float(cfg.max_l_gap_day), float(cfg.max_ab_gap_day)
    if phase == "night":
        return float(cfg.max_l_gap_night), float(cfg.max_ab_gap_night)
    return float(cfg.max_l_gap_twilight), float(cfg.max_ab_gap_twilight)


def photometric_match_details(
    patch_rgb_u8: np.ndarray,
    roi_rgb_u8: np.ndarray,
    cfg: ConstraintConfig,
) -> Tuple[np.ndarray, Dict[str, float], bool]:
    p0 = _ensure_3ch(patch_rgb_u8)
    r = _ensure_3ch(roi_rgb_u8)

    l0, ab0 = photometric_gap(p0, r)
    p1 = p0
    if bool(cfg.enable_photometric) and float(cfg.lab_match_strength) > 1e-6:
        p1 = lab_match_patch_to_roi(p0, r, strength=float(cfg.lab_match_strength))

    l1, ab1 = photometric_gap(p1, r)
    phase, v_mean = infer_day_phase(r, cfg)
    max_l, max_ab = _gaps_by_phase(phase, cfg)
    max_l_pre = max(1e-6, float(cfg.max_l_gap_pre))
    max_ab_pre = max(1e-6, float(cfg.max_ab_gap_pre))

    max_l = max(1e-6, float(max_l))
    max_ab = max(1e-6, float(max_ab))
    phi_gap = 0.5 * (l1 / max_l) + 0.5 * (ab1 / max_ab)
    pre_ok = (l0 <= max_l_pre) and (ab0 <= max_ab_pre)
    ok = pre_ok and (l1 <= max_l) and (ab1 <= max_ab)

    stats = {
        "l_gap_before": float(l0),
        "ab_gap_before": float(ab0),
        "l_gap_after": float(l1),
        "ab_gap_after": float(ab1),
        "max_l_gap": float(max_l),
        "max_ab_gap": float(max_ab),
        "max_l_gap_pre": float(max_l_pre),
        "max_ab_gap_pre": float(max_ab_pre),
        "precheck_ok": bool(pre_ok),
        "phi_gap": float(phi_gap),
        "roi_v_mean": float(v_mean),
        "day_phase": phase,
    }
    return p1, stats, bool(ok)


def photometric_match(
    patch_rgb_u8: np.ndarray,
    roi_rgb_u8: np.ndarray,
    cfg: ConstraintConfig,
) -> Tuple[np.ndarray, float, bool]:
    p, stats, ok = photometric_match_details(patch_rgb_u8, roi_rgb_u8, cfg)
    return p, float(stats["phi_gap"]), bool(ok)


def context_scores(roi_rgb_u8: np.ndarray) -> Tuple[float, float]:
    roi = _ensure_3ch(roi_rgb_u8)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())
    texture = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return edge_density, texture


def context_ok(
    roi_rgb_u8: np.ndarray,
    cfg: ConstraintConfig,
) -> Tuple[bool, float, float]:
    ed, tx = context_scores(roi_rgb_u8)
    if not cfg.enable_context_aware:
        return True, ed, tx
    ok = (ed <= float(cfg.max_edge_density)) and (tx <= float(cfg.max_texture))
    return ok, ed, tx


def _q_bounds(values: List[float], d_lo: float, d_hi: float, q_low: float, q_high: float) -> Tuple[float, float]:
    if len(values) < 20:
        return float(d_lo), float(d_hi)
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))
    if hi <= lo + 1e-6:
        return float(d_lo), float(d_hi)
    return lo, hi


def placement_score(
    roi_rgb_u8: np.ndarray,
    bbox_xywh: Tuple[int, int, int, int],
    canvas_w: int,
    canvas_h: int,
    feature_history: Optional[Dict[str, List[float]]] = None,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> Tuple[float, Dict[str, float]]:
    roi = _ensure_3ch(roi_rgb_u8)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = float(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y).mean())

    texture_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    x, y, w, h = bbox_xywh
    d_left = float(x)
    d_top = float(y)
    d_right = float(max(0, canvas_w - (x + w)))
    d_bottom = float(max(0, canvas_h - (y + h)))
    boundary_dist = min(d_left, d_top, d_right, d_bottom) / max(1.0, 0.5 * float(min(canvas_w, canvas_h)))
    boundary_dist = _clip01(boundary_dist)

    hist = feature_history or {}
    g_lo, g_hi = _q_bounds(hist.get("grad_energy", []), 25.0, 145.0, q_low, q_high)
    t_lo, t_hi = _q_bounds(hist.get("texture_var", []), 20.0, 260.0, q_low, q_high)
    b_lo, b_hi = _q_bounds(hist.get("boundary_dist", []), 0.03, 0.35, q_low, q_high)

    # low gradient/texture are favored for stable compositing; larger boundary distance is favored.
    q_grad = 1.0 - _norm01(grad_energy, g_lo, g_hi)
    q_tex = 1.0 - _norm01(texture_var, t_lo, t_hi)
    q_bnd = _norm01(boundary_dist, b_lo, b_hi)

    score = 0.40 * q_grad + 0.35 * q_tex + 0.25 * q_bnd
    score = _clip01(score)

    stats = {
        "grad_energy": float(grad_energy),
        "texture_var": float(texture_var),
        "boundary_dist": float(boundary_dist),
        "grad_q_lo": float(g_lo),
        "grad_q_hi": float(g_hi),
        "texture_q_lo": float(t_lo),
        "texture_q_hi": float(t_hi),
        "boundary_q_lo": float(b_lo),
        "boundary_q_hi": float(b_hi),
        "quality_grad": float(q_grad),
        "quality_texture": float(q_tex),
        "quality_boundary": float(q_bnd),
        "score_place": float(score),
    }
    return score, stats
