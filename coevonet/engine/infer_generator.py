# coevonet/engine/infer_generator.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from coevonet.models.generator import SiameseWKANAE


@dataclass
class CoEvoNetInferConfig:
    base_ch: int = 32
    latent_ch: int = 256
    device: str = "cuda"
    amp: bool = True


class CoEvoNetTranslator:
    """
    Standard inference wrapper for Step2 translator (SiameseWKANAE).
    """

    def __init__(self, ckpt_path: Path, cfg: Optional[CoEvoNetInferConfig] = None):
        self.ckpt_path = Path(ckpt_path)
        if cfg is None:
            cfg = CoEvoNetInferConfig()

        self.cfg = cfg
        dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.device = dev

        self.model = SiameseWKANAE(base_ch=cfg.base_ch, latent_ch=cfg.latent_ch).to(self.device)
        state = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # AMP only valid on CUDA
        self.use_amp = bool(cfg.amp) and (self.device.type == "cuda")

    @torch.no_grad()
    def __call__(self, rgb: torch.Tensor, tir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rgb: [B,3,H,W] float in [0,1]
        tir: [B,1,H,W] float in [0,1]
        return:
          rgb_hat: [B,3,H,W]
          tir_hat: [B,1,H,W]
        """
        rgb = rgb.to(self.device, non_blocking=True)
        tir = tir.to(self.device, non_blocking=True)

        if self.use_amp:
            with torch.autocast(device_type="cuda", enabled=True):
                rgb_hat, tir_hat = self.model(rgb, tir)
        else:
            rgb_hat, tir_hat = self.model(rgb, tir)

        return rgb_hat, tir_hat
