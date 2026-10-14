"""CoEvoNet UAV-RGBT dataset adapter for the MMDetection 3.x migration."""

from __future__ import annotations

import os.path as osp
from typing import Iterable, Optional, Sequence, Tuple

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class CoEvoNetUAVRGBTDataset(CocoDataset):
    """Multi-class COCO-style UAV-RGBT dataset for CoEvoNet experiments."""

    METAINFO = {
        "classes": (
            "ship",
            "car",
            "cyclist",
            "pedestrian",
            "bus",
            "drone",
            "plane",
        ),
        "palette": [
            (54, 94, 239),
            (245, 158, 11),
            (16, 185, 129),
            (236, 72, 153),
            (249, 115, 22),
            (139, 92, 246),
            (14, 165, 233),
        ],
    }

    def __init__(
        self,
        *args,
        tir_data_prefix: Optional[str] = None,
        filename_replacements: Optional[Sequence[Tuple[str, str]]] = None,
        **kwargs,
    ) -> None:
        self.tir_data_prefix = tir_data_prefix
        self.filename_replacements = tuple(filename_replacements or ())
        super().__init__(*args, **kwargs)

    def _derive_tir_path(self, rgb_path: str, file_name: str) -> str:
        if self.tir_data_prefix:
            return osp.join(self.tir_data_prefix, file_name)

        tir_path = rgb_path
        for src, dst in self.filename_replacements:
            tir_path = tir_path.replace(src, dst)
        return tir_path

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)
        raw_img_info = raw_data_info.get("raw_img_info", {})
        file_name = raw_img_info.get("file_name", "")
        data_info["tir_path"] = self._derive_tir_path(data_info["img_path"], file_name)
        return data_info
