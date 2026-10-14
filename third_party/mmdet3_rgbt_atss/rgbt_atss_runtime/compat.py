from __future__ import annotations

import importlib
import sys
import types


def ensure_mmcv_ops_fallback() -> bool:
    try:
        from mmcv.ops import batched_nms  # noqa: F401
        from mmcv.ops import sigmoid_focal_loss  # noqa: F401
        return False
    except Exception:
        import torch
        from torchvision.ops import batched_nms as tv_batched_nms
        from torchvision.ops import nms as tv_nms
        from torchvision.ops import roi_align as tv_roi_align
        from torchvision.ops import sigmoid_focal_loss as tv_sigmoid_focal_loss

        def _batched_nms(boxes, scores, idxs, nms_cfg=None, class_agnostic=False):
            iou_threshold = 0.5
            if isinstance(nms_cfg, dict):
                iou_threshold = float(
                    nms_cfg.get("iou_threshold", nms_cfg.get("iou_thr", 0.5))
                )
            if class_agnostic:
                idxs = torch.zeros_like(idxs)
            keep = tv_batched_nms(boxes, scores, idxs, iou_threshold)
            dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
            return dets, keep

        def _nms(boxes, scores, iou_threshold, offset=0, score_threshold=0.0, max_num=-1):
            keep = tv_nms(boxes, scores, float(iou_threshold))
            if score_threshold and score_threshold > 0:
                keep = keep[scores[keep] > score_threshold]
            if max_num is not None and max_num > 0:
                keep = keep[: int(max_num)]
            dets = torch.cat([boxes[keep], scores[keep, None]], dim=1)
            return dets, keep

        def _sigmoid_focal_loss(inputs, targets, gamma=2.0, alpha=0.25, weight=None, reduction="none"):
            loss = tv_sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma, reduction="none")
            if weight is not None:
                loss = loss * weight
            if reduction == "sum":
                return loss.sum()
            if reduction == "mean":
                return loss.mean()
            return loss

        def _roi_align(input, rois, output_size, spatial_scale=1.0, sampling_ratio=0, pool_mode="avg", aligned=True):
            return tv_roi_align(
                input,
                rois,
                output_size=output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio,
                aligned=aligned,
            )

        mmcv_ops = types.ModuleType("mmcv.ops")
        mmcv_ops.batched_nms = _batched_nms
        mmcv_ops.nms = _nms
        mmcv_ops.sigmoid_focal_loss = _sigmoid_focal_loss
        mmcv_ops.roi_align = _roi_align

        mmcv_ops_nms = types.ModuleType("mmcv.ops.nms")
        mmcv_ops_nms.batched_nms = _batched_nms
        mmcv_ops_nms.nms = _nms

        mmcv_ops_roi_align = types.ModuleType("mmcv.ops.roi_align")
        mmcv_ops_roi_align.roi_align = _roi_align

        sys.modules["mmcv.ops"] = mmcv_ops
        sys.modules["mmcv.ops.nms"] = mmcv_ops_nms
        sys.modules["mmcv.ops.roi_align"] = mmcv_ops_roi_align
        return True


def bootstrap_rgbt_atss_runtime() -> None:
    ensure_mmcv_ops_fallback()

    required_modules = [
        "mmdet.datasets",
        "mmdet.datasets.transforms.loading",
        "mmdet.datasets.transforms.transforms",
        "mmdet.datasets.transforms.formatting",
        "mmdet.evaluation.metrics.coco_metric",
        "mmdet.models.data_preprocessors.data_preprocessor",
        "mmdet.models.backbones.resnet",
        "mmdet.models.necks.fpn",
        "mmdet.models.losses.cross_entropy_loss",
        "mmdet.models.losses.focal_loss",
        "mmdet.models.losses.iou_loss",
        "mmdet.models.task_modules.prior_generators.anchor_generator",
        "mmdet.models.task_modules.assigners.atss_assigner",
        "mmdet.models.task_modules.coders.delta_xywh_bbox_coder",
        "mmdet.models.dense_heads.anchor_head",
        "mmdet.models.dense_heads.atss_head",
        "mmdet.models.detectors.base",
        "mmdet.models.detectors.single_stage",
        "mmdet.models.detectors.atss",
        "rgbt_atss_runtime.datasets",
        "rgbt_atss_runtime.datasets.transforms",
        "rgbt_atss_runtime.models",
        "rgbt_atss_runtime.samplers",
    ]

    for module_name in required_modules:
        importlib.import_module(module_name)

    from mmengine.registry import DefaultScope

    current = DefaultScope.get_current_instance()
    if current is None or current.scope_name != "mmdet":
        DefaultScope.get_instance("rgbt-atss-runtime", scope_name="mmdet")
