# coevonet/synthesis/wavelet_refiner.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _to_float01(img_u8: np.ndarray) -> np.ndarray:
    return np.clip(img_u8.astype(np.float32) / 255.0, 0.0, 1.0)


def _haar_dwt2(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = x.shape
    x00 = x[0:h:2, 0:w:2]
    x01 = x[0:h:2, 1:w:2]
    x10 = x[1:h:2, 0:w:2]
    x11 = x[1:h:2, 1:w:2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def _haar_idwt2(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray) -> np.ndarray:
    h2, w2 = ll.shape
    out = np.zeros((h2 * 2, w2 * 2), dtype=np.float32)
    out[0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    out[0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    out[1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    out[1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return out


def _hf_energy(lh: np.ndarray, hl: np.ndarray, hh: np.ndarray, mask_small: Optional[np.ndarray]) -> float:
    e = (lh * lh + hl * hl + hh * hh).astype(np.float32)
    if mask_small is None:
        return float(np.mean(e))
    w = np.clip(mask_small.astype(np.float32), 0.0, 1.0)
    denom = float(np.sum(w))
    if denom <= 1e-6:
        return float(np.mean(e))
    return float(np.sum(e * w) / denom)


def wavelet_refine_tir(
    rgb_patch_u8: np.ndarray,
    tir_patch_u8: np.ndarray,
    mask_u8: Optional[np.ndarray],
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """
    Lightweight analytic wavelet refiner for tiny/small ROI.
    Returns (tir_refined_u8, stats).
    """
    mode = str(cfg.get("wavelet_refiner_mode", "analytic")).strip().lower()
    steps = int(cfg.get("wavelet_refiner_steps", 2))
    strength = float(cfg.get("wavelet_refiner_strength", 0.8))
    wavelet_name = str(cfg.get("wavelet_name", "haar")).strip().lower()

    if mode != "analytic" or wavelet_name != "haar" or steps <= 0:
        return tir_patch_u8, {
            "wavelet_refiner_enabled": 0,
            "wavelet_mode": mode,
            "wavelet_steps": int(max(0, steps)),
            "wavelet_energy_before": 0.0,
            "wavelet_energy_after": 0.0,
            "wavelet_gain_ratio": 1.0,
            "tir_edge_energy_gain": 1.0,
        }

    rgb = rgb_patch_u8
    tir = _to_float01(tir_patch_u8)
    h0, w0 = tir.shape

    # Ensure even shape for Haar blocks.
    pad_h = h0 % 2
    pad_w = w0 % 2
    if pad_h or pad_w:
        tir = np.pad(tir, ((0, pad_h), (0, pad_w)), mode="edge")
        rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE)
        if mask_u8 is not None:
            mask_u8 = cv2.copyMakeBorder(mask_u8, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / float(np.percentile(edge, 95) + 1e-6)
    edge = np.clip(edge, 0.0, 1.0)

    mask_f = None
    if mask_u8 is not None:
        mask_f = np.clip(mask_u8.astype(np.float32) / 255.0, 0.0, 1.0)

    # Track gradient energy gain on TIR itself (readable metric).
    gxb = cv2.Sobel(tir, cv2.CV_32F, 1, 0, ksize=3)
    gyb = cv2.Sobel(tir, cv2.CV_32F, 0, 1, ksize=3)
    tir_edge_before = float(np.mean(np.sqrt(gxb * gxb + gyb * gyb)))

    energy_before = None
    cur = tir.copy()
    for _ in range(int(steps)):
        ll, lh, hl, hh = _haar_dwt2(cur)
        h2, w2 = ll.shape
        edge_small = cv2.resize(edge, (w2, h2), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if mask_f is not None:
            mask_small = cv2.resize(mask_f, (w2, h2), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        else:
            mask_small = None

        eb = _hf_energy(lh, hl, hh, mask_small)
        if energy_before is None:
            energy_before = eb

        # Analytic high-frequency boost guided by RGB edge prior.
        # Strong edge -> keep/boost HF, flat area -> mild shrink to suppress paste artifacts.
        boost = 0.85 + float(strength) * edge_small
        if mask_small is not None:
            boost = 1.0 + (boost - 1.0) * mask_small

        lh2 = lh * boost
        hl2 = hl * boost
        hh2 = hh * boost
        cur = np.clip(_haar_idwt2(ll, lh2, hl2, hh2), 0.0, 1.0)

    ll, lh, hl, hh = _haar_dwt2(cur)
    if mask_f is not None:
        mask_small = cv2.resize(mask_f, (ll.shape[1], ll.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        mask_small = None
    energy_after = _hf_energy(lh, hl, hh, mask_small)
    energy_before = float(energy_before if energy_before is not None else energy_after)
    gain = float(energy_after / max(1e-8, energy_before))

    gxa = cv2.Sobel(cur, cv2.CV_32F, 1, 0, ksize=3)
    gya = cv2.Sobel(cur, cv2.CV_32F, 0, 1, ksize=3)
    tir_edge_after = float(np.mean(np.sqrt(gxa * gxa + gya * gya)))
    tir_edge_gain = float(tir_edge_after / max(1e-8, tir_edge_before))

    out = (np.clip(cur, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out = out[:h0, :w0]
    return out, {
        "wavelet_refiner_enabled": 1,
        "wavelet_mode": mode,
        "wavelet_steps": int(steps),
        "wavelet_energy_before": float(energy_before),
        "wavelet_energy_after": float(energy_after),
        "wavelet_gain_ratio": float(gain),
        "tir_edge_energy_gain": float(tir_edge_gain),
    }

