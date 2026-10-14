from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig_mmdet3")))

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="1-batch smoke for dual-stream MMDet3 ATSS")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "dualstream_atss_r50_fpn_1x_coevonet_real_only.py"),
    )
    return parser.parse_args()


def reduce_loss_dict(losses: dict) -> tuple[dict, torch.Tensor]:
    reduced = {}
    total = None
    for key, value in losses.items():
        if isinstance(value, (list, tuple)):
            scalar = sum(v.mean() for v in value)
        else:
            scalar = value.mean()
        reduced[key] = float(scalar.detach().cpu())
        total = scalar if total is None else total + scalar
    return reduced, total


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    sample = dataset[0]
    img = sample["inputs"]
    data_sample = sample["data_samples"]
    print("dataset_len=", len(dataset))
    print("sample_input_shape=", tuple(img.shape))
    print("sample_bbox_count=", int(len(data_sample.gt_instances.labels)))

    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    batch = next(iter(dataloader))
    model = MODELS.build(cfg.model)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    processed = model.data_preprocessor(batch, training=True)
    batch_inputs = processed["inputs"]
    batch_data_samples = processed["data_samples"]
    losses = model(batch_inputs, batch_data_samples, mode="loss")
    reduced, total_loss = reduce_loss_dict(losses)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("batch_size=", int(batch_inputs.shape[0]))
    print("model_cls=", model.__class__.__name__)
    print("loss_keys=", sorted(losses.keys()))
    print("loss_total=", float(total_loss.detach().cpu()))
    print("grad_step_ok=", True)
    print("loss_summary=", reduced)


if __name__ == "__main__":
    main()
