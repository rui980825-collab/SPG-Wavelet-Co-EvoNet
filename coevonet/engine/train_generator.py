# coevonet/engine/train_generator.py
# -*- coding: utf-8 -*-

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# NOTE: use the new AMP API to avoid FutureWarning
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt

from coevonet.data.patch_bank import PatchBankDataset
from coevonet.models.generator import SiameseWKANAE
from coevonet.losses.hsg_alignment import HSGLoss
from coevonet.losses.fib_frequency import FIBWaveletLoss, wavelet_energy_stats


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_id():
    return time.strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _save_recon_grid(path: Path, rgb, tir, rgb_hat, tir_hat, n=16):
    """
    4-row grid, cols=b (<=n):
      Row0: RGB_in
      Row1: TIR_in
      Row2: RGB_hat
      Row3: TIR_hat
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    b = min(int(n), int(rgb.shape[0]))
    cols = b
    rows = 4

    # tensors -> numpy
    rgb = rgb[:b].detach().cpu().float().numpy()
    tir = tir[:b].detach().cpu().float().numpy()
    rgb_hat = rgb_hat[:b].detach().cpu().float().numpy()
    tir_hat = tir_hat[:b].detach().cpu().float().numpy()

    plt.figure(figsize=(cols * 1.2, rows * 1.2))

    for i in range(b):
        # Row 0
        ax = plt.subplot(rows, cols, 0 * cols + i + 1)
        # rgb: [3,H,W]
        ax.imshow(np.transpose(rgb[i], (1, 2, 0)))
        ax.axis("off")
        if i == 0:
            ax.set_title("RGB_in", fontsize=10)

        # Row 1
        ax = plt.subplot(rows, cols, 1 * cols + i + 1)
        ax.imshow(tir[i, 0], cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("TIR_in", fontsize=10)

        # Row 2
        ax = plt.subplot(rows, cols, 2 * cols + i + 1)
        ax.imshow(np.transpose(rgb_hat[i], (1, 2, 0)))
        ax.axis("off")
        if i == 0:
            ax.set_title("RGB_hat", fontsize=10)

        # Row 3 (FIX: was 4*cols -> out of range)
        ax = plt.subplot(rows, cols, 3 * cols + i + 1)
        ax.imshow(tir_hat[i, 0], cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("TIR_hat", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def train_generator(cfg: dict, cfg_path: Path):
    _set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    runs_root = Path(cfg.get("runs_root", "runs"))
    dataset_tag = str(cfg.get("dataset_tag", "dataset"))
    run_id = _run_id()

    run_dir = runs_root / "gen" / dataset_tag / run_id
    ckpt_dir = run_dir / "checkpoints"
    fig_dir = run_dir / "figures"
    tab_dir = run_dir / "tables"
    meta_dir = run_dir / "meta"
    cfg_dir = run_dir / "configs"
    for d in [ckpt_dir, fig_dir, tab_dir, meta_dir, cfg_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # snapshot config
    (cfg_dir / cfg_path.name).write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    _save_json(
        meta_dir / "run_meta.json",
        {
            "run_id": run_id,
            "dataset_tag": dataset_tag,
            "patch_bank_root": cfg.get("patch_bank_root"),
            "img_size": cfg.get("img_size"),
            "device": str(device),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # dataset
    ds = PatchBankDataset(Path(cfg["patch_bank_root"]), img_size=int(cfg["img_size"]))
    n_total = len(ds)
    n_train = int(n_total * float(cfg.get("train_ratio", 0.9)))
    n_val = n_total - n_train
    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )

    # model & losses
    model = SiameseWKANAE(
        base_ch=int(cfg.get("base_ch", 32)),
        latent_ch=int(cfg.get("latent_ch", 256)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    l1 = nn.L1Loss()
    hsg = HSGLoss().to(device)
    fib = FIBWaveletLoss().to(device)

    lam_rec = float(cfg.get("lambda_rec", 1.0))
    lam_hsg = float(cfg.get("lambda_hsg", 0.5))
    lam_fib = float(cfg.get("lambda_fib", 0.2))

    epochs = int(cfg.get("epochs", 20))
    export_every = int(cfg.get("export_every", 1))
    export_n_vis = int(cfg.get("export_n_vis", 16))

    use_amp = bool(cfg.get("amp", True))

    # new AMP scaler signature
    scaler = GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else GradScaler(enabled=False)

    loss_rows = []
    wavelet_rows = []
    align_rows = []

    best_val = float("inf")

    print(f"🚀 Step2 Training Start | run_dir={run_dir}")
    print(f"   device={device} amp={use_amp} pairs={n_total} (train={n_train}, val={n_val})")

    for ep in range(1, epochs + 1):
        # ---------------- train ----------------
        model.train()
        tr_sum = 0.0
        tr_n = 0

        for rgb, tir in train_loader:
            rgb = rgb.to(device, non_blocking=True)
            tir = tir.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # 1) forward under autocast
            with autocast(device_type=device.type, enabled=use_amp):
                rgb_hat, tir_hat = model(rgb, tir)
                l_rec = l1(rgb_hat, rgb) + l1(tir_hat, tir)

            # 2) compute conv-heavy losses in float32 to avoid Half/Float kernel mismatch
            rgb_hat_f = rgb_hat.float()
            tir_hat_f = tir_hat.float()
            rgb_f = rgb.float()
            tir_f = tir.float()

            l_hsg = hsg(rgb_hat_f, tir_hat_f)
            l_fib = fib(rgb_hat_f, rgb_f) + fib(tir_hat_f, tir_f)

            loss = lam_rec * l_rec.float() + lam_hsg * l_hsg + lam_fib * l_fib

            # numeric fuse
            if not torch.isfinite(loss):
                print("⚠️ non-finite loss detected. Skipping batch. (AMP will be disabled next)")
                use_amp = False
                scaler = GradScaler("cuda", enabled=False) if device.type == "cuda" else GradScaler(enabled=False)
                continue

            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            tr_sum += float(loss.item()) * rgb.shape[0]
            tr_n += rgb.shape[0]

        tr_loss = tr_sum / max(1, tr_n)

        # ---------------- val ----------------
        model.eval()
        va_sum = va_rec = va_hsg = va_fib = 0.0
        va_n = 0

        e_rgb_ll = e_rgb_hf = e_tir_ll = e_tir_hf = 0.0
        align_score = 0.0

        vis_pack = None

        with torch.no_grad():
            for rgb, tir in val_loader:
                rgb = rgb.to(device, non_blocking=True)
                tir = tir.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=use_amp):
                    rgb_hat, tir_hat = model(rgb, tir)
                    l_rec_v = l1(rgb_hat, rgb) + l1(tir_hat, tir)

                # float32 metrics/losses
                rgb_hat_f = rgb_hat.float()
                tir_hat_f = tir_hat.float()
                rgb_f = rgb.float()
                tir_f = tir.float()

                l_hsg_v = hsg(rgb_hat_f, tir_hat_f)
                l_fib_v = fib(rgb_hat_f, rgb_f) + fib(tir_hat_f, tir_f)

                loss_v = lam_rec * l_rec_v.float() + lam_hsg * l_hsg_v + lam_fib * l_fib_v

                bs = rgb.shape[0]
                va_sum += float(loss_v.item()) * bs
                va_rec += float(l_rec_v.item()) * bs
                va_hsg += float(l_hsg_v.item()) * bs
                va_fib += float(l_fib_v.item()) * bs
                va_n += bs

                # wavelet stats in float32
                st_rgb = wavelet_energy_stats(rgb_hat_f, rgb_f)
                st_tir = wavelet_energy_stats(tir_hat_f, tir_f)
                e_rgb_ll += st_rgb["E_ll"] * bs
                e_rgb_hf += st_rgb["E_hf"] * bs
                e_tir_ll += st_tir["E_ll"] * bs
                e_tir_hf += st_tir["E_hf"] * bs

                # grad alignment score in float32 (avoid AMP dtype mismatch)
                align_score += float(hsg.grad_alignment_score(rgb_hat_f, tir_hat_f)) * bs

                if vis_pack is None:
                    vis_pack = (
                        rgb.detach().cpu(),
                        tir.detach().cpu(),
                        rgb_hat.detach().cpu(),
                        tir_hat.detach().cpu(),
                    )

        va_loss = va_sum / max(1, va_n)
        va_rec_m = va_rec / max(1, va_n)
        va_hsg_m = va_hsg / max(1, va_n)
        va_fib_m = va_fib / max(1, va_n)

        e_rgb_ll /= max(1, va_n)
        e_rgb_hf /= max(1, va_n)
        e_tir_ll /= max(1, va_n)
        e_tir_hf /= max(1, va_n)
        align_score /= max(1, va_n)

        loss_rows.append([ep, tr_loss, va_loss, va_rec_m, va_hsg_m, va_fib_m])
        wavelet_rows.append([ep, e_rgb_ll, e_rgb_hf, e_tir_ll, e_tir_hf])
        align_rows.append([ep, align_score])

        # checkpoints
        torch.save(model.state_dict(), ckpt_dir / "last.pth")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pth")

        # export tables (overwrite each epoch; deterministic for paper)
        _write_csv(
            tab_dir / "gen_loss_curves.csv",
            ["epoch", "train_loss", "val_loss", "val_rec", "val_hsg", "val_fib"],
            loss_rows,
        )
        _write_csv(
            tab_dir / "wavelet_energy.csv",
            ["epoch", "E_rgb_LL", "E_rgb_HF", "E_tir_LL", "E_tir_HF"],
            wavelet_rows,
        )
        _write_csv(
            tab_dir / "alignment_stats.csv",
            ["epoch", "grad_align_score"],
            align_rows,
        )

        # export figures
        if export_every > 0 and (ep % export_every == 0) and vis_pack is not None:
            rgb0, tir0, rgbh0, tirh0 = vis_pack
            _save_recon_grid(
                fig_dir / f"fig_gen_recon_grid_ep{ep:03d}.png",
                rgb0,
                tir0,
                rgbh0,
                tirh0,
                n=export_n_vis,
            )

        print(
            f"[Ep {ep:03d}/{epochs}] "
            f"train={tr_loss:.4f} val={va_loss:.4f} "
            f"(rec={va_rec_m:.4f}, hsg={va_hsg_m:.4f}, fib={va_fib_m:.4f}) "
            f"| align={align_score:.4f} "
            f"| wavelet(HF rgb={e_rgb_hf:.4f}, tir={e_tir_hf:.4f})"
        )

    print(f"✅ Step2 done. best_val={best_val:.4f} | run_dir={run_dir}")
