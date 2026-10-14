from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class DualStreamResNet(BaseModule):
    """Two-branch ResNet backbone with simple late fusion.

    Input is a 6-channel tensor formed by concatenating RGB and TIR channels.
    Each branch uses an independent ResNet. Stage features are fused with
    ``concat + 1x1 conv`` before passing to the shared FPN.
    """

    def __init__(
        self,
        backbone_cfg: dict,
        out_channels: Sequence[int] = (256, 512, 1024, 2048),
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU"),
        pretrained_path: str | None = None,
        rgb_init_cfg: dict | None = None,
        tir_init_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.out_channels = tuple(out_channels)
        rgb_backbone_cfg = copy.deepcopy(backbone_cfg)
        tir_backbone_cfg = copy.deepcopy(backbone_cfg)

        if pretrained_path is not None:
            checkpoint = str(Path(pretrained_path))
            shared_init_cfg = dict(type="Pretrained", checkpoint=checkpoint)
            rgb_backbone_cfg["init_cfg"] = copy.deepcopy(shared_init_cfg)
            tir_backbone_cfg["init_cfg"] = copy.deepcopy(shared_init_cfg)

        if rgb_init_cfg is not None:
            rgb_backbone_cfg["init_cfg"] = copy.deepcopy(rgb_init_cfg)
        if tir_init_cfg is not None:
            tir_backbone_cfg["init_cfg"] = copy.deepcopy(tir_init_cfg)

        self.rgb_backbone = MODELS.build(rgb_backbone_cfg)
        self.tir_backbone = MODELS.build(tir_backbone_cfg)
        self.fusion_convs = nn.ModuleList(
            [
                ConvModule(
                    in_channels=2 * channels,
                    out_channels=channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for channels in self.out_channels
            ]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        if x.ndim != 4 or x.size(1) != 6:
            raise ValueError(
                f"DualStreamResNet expects input shaped (N, 6, H, W), got {tuple(x.shape)}"
            )

        rgb = x[:, :3, :, :]
        tir = x[:, 3:, :, :]

        rgb_feats = self.rgb_backbone(rgb)
        tir_feats = self.tir_backbone(tir)
        if len(rgb_feats) != len(self.fusion_convs) or len(tir_feats) != len(self.fusion_convs):
            raise ValueError(
                "Unexpected number of stages returned by the inner backbones: "
                f"rgb={len(rgb_feats)}, tir={len(tir_feats)}, fusion={len(self.fusion_convs)}"
            )

        fused_feats = []
        for fusion_conv, feat_rgb, feat_tir in zip(self.fusion_convs, rgb_feats, tir_feats):
            fused_feats.append(fusion_conv(torch.cat([feat_rgb, feat_tir], dim=1)))
        return tuple(fused_feats)
