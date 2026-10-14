from __future__ import annotations

from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import get as fileio_get

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadRGBTConcatenatedImageFromFile(BaseTransform):
    """Load an aligned RGB/TIR pair and concatenate it into a 6-channel image.

    The dataset prepares ``img_path`` and ``tir_path``. This transform keeps the
    rest of the MMDetection pipeline standard by exposing a single HWC tensor.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        imdecode_backend: str = "cv2",
        to_rgb: bool = True,
        file_client_args: Optional[dict] = None,
    ) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.to_rgb = to_rgb
        self.file_client_args = file_client_args or {"backend": "disk"}

    def _load_single(self, filename: str) -> np.ndarray:
        img = mmcv.imfrombytes(
            fileio_get(filename),
            flag=self.color_type,
            backend=self.imdecode_backend,
        )
        if self.to_rgb and img.ndim == 3 and img.shape[2] == 3:
            img = mmcv.bgr2rgb(img)
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def transform(self, results: dict) -> dict:
        filename_rgb = results.get("img_path")
        filename_tir = results.get("tir_path")
        if not filename_rgb or not filename_tir:
            raise KeyError(
                "LoadRGBTConcatenatedImageFromFile expects both 'img_path' and "
                "'tir_path' to be prepared by the dataset."
            )

        img_rgb = self._load_single(filename_rgb)
        img_tir = self._load_single(filename_tir)

        if img_rgb.shape[:2] != img_tir.shape[:2]:
            raise ValueError(
                f"RGB/TIR spatial mismatch: {filename_rgb} -> {img_rgb.shape}, "
                f"{filename_tir} -> {img_tir.shape}"
            )

        img = np.concatenate([img_rgb, img_tir], axis=2)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        results["tir_path"] = filename_tir
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(to_float32={self.to_float32}, "
            f"color_type={self.color_type!r}, imdecode_backend={self.imdecode_backend!r}, "
            f"to_rgb={self.to_rgb})"
        )


@TRANSFORMS.register_module()
class NormalizeConcatenatedRGBT(BaseTransform):
    """Normalize a concatenated 6-channel RGB/TIR image."""

    def __init__(self, mean, std) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        if self.mean.shape[0] != 6 or self.std.shape[0] != 6:
            raise ValueError("NormalizeConcatenatedRGBT expects 6-value mean/std.")

    def transform(self, results: dict) -> dict:
        img = results["img"].astype(np.float32, copy=False)
        results["img"] = (img - self.mean.reshape(1, 1, 6)) / self.std.reshape(1, 1, 6)
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=False)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean.tolist()}, std={self.std.tolist()})"
