# scripts/11_infer_coevonet.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from coevonet.data.patch_bank import PatchBankDataset
from coevonet.engine.infer_generator import CoEvoNetTranslator, CoEvoNetInferConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_bank_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_batches", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=2)

    # NEW: fast subset controls
    ap.add_argument("--max_pairs", type=int, default=256, help="limit paired samples for fast infer")
    ap.add_argument("--class_filter", type=str, default="", help="comma-separated class names, e.g., ship,car")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    class_filter = None
    if isinstance(args.class_filter, str) and args.class_filter.strip():
        class_filter = [x.strip() for x in args.class_filter.split(",") if x.strip()]

    ds = PatchBankDataset(
        Path(args.patch_bank_root),
        img_size=int(args.img_size),
        max_pairs=int(args.max_pairs) if args.max_pairs > 0 else None,
        class_filter=class_filter,
        seed=int(args.seed),
        verbose=True,
    )

    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    translator = CoEvoNetTranslator(
        ckpt_path=Path(args.ckpt),
        cfg=CoEvoNetInferConfig(device=args.device, amp=bool(args.amp)),
    )

    n = 0
    for rgb, tir in dl:
        rgb_hat, tir_hat = translator(rgb, tir)
        print(
            f"[batch {n}] rgb={tuple(rgb.shape)} tir={tuple(tir.shape)} "
            f"-> rgb_hat={tuple(rgb_hat.shape)} tir_hat={tuple(tir_hat.shape)}"
        )
        n += 1
        if n >= int(args.num_batches):
            break

    print("✅ inference ok")


if __name__ == "__main__":
    main()
