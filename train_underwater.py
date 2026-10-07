import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from underwater_loader import UnderwaterDataset

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/underwater.yml")
parser.add_argument("--repo", default="BVI-Mamba")
parser.add_argument("--out", default="checkpoints")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--max_hours", type=float, default=0)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

sys.path.insert(0, str(Path(args.repo).resolve()))
from arch.BVIMamba import STASUNet


def build_model(cfg):
    m, d = cfg["model"], cfg["dataset"]
    return STASUNet(
        num_in_ch=m["num_in_ch"],
        num_out_ch=m["num_out_ch"],
        num_feat=m["num_feat"],
        num_frame=d["num_frames"],
        deformable_groups=m["deformable_groups"],
        num_extract_block=m["num_extract_block"],
        num_reconstruct_block=m["num_reconstruct_block"],
        center_frame_idx=None,
        hr_in=m["hr_in"],
        img_size=d["image_size"],
        patch_size=m["patch_size"],
        embed_dim=m["embed_dim"],
        depths=m["depths"],
        num_heads=m["num_heads"],
        window_size=m["window_size"],
        patch_norm=m["patch_norm"],
        final_upsample="Dual up-sample",
    )


def to_uint8(t):
    x = t.detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def psnr(out, gt):
    o = out.clamp(-1, 1) * 0.5 + 0.5
    g = gt * 0.5 + 0.5
    mse = torch.mean((o - g) ** 2).item()
    return 100.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


@torch.no_grad()
def validate(model, loader, device, criterion, save_dir, epoch, n_save=4):
    model.eval()
    save_ids = set(np.linspace(0, len(loader) - 1, n_save).astype(int).tolist())
    losses, psnrs = [], []
    for i, s in enumerate(loader):
        x = s["image"].to(device)
        y = s["groundtruth"].to(device)
        out = model(x)
        losses.append(criterion(out, y).item())
        psnrs.append(psnr(out[0], y[0]))
        if i in save_ids:
            center = x[0, x.shape[1] // 2]
            panel = np.concatenate(
                [to_uint8(center), to_uint8(out[0]), to_uint8(y[0])], axis=1
            )
            cv2.imwrite(str(save_dir / f"ep{epoch:03d}_{i:03d}.jpg"), panel)
    return float(np.mean(losses)), float(np.mean(psnrs))


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    d, t = cfg["dataset"], cfg["training"]

    per_epoch = d["samples_per_epoch"]
    val_samples = d["val_samples"]
    maxepoch = t["maxepoch"]
    out_dir = Path(args.out)
    if args.smoke:
        per_epoch, val_samples, maxepoch = 80, 4, 2
        out_dir = out_dir / "smoke"
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = UnderwaterDataset(
        d["root"], d["split_file"], "train", d["num_frames"], d["image_size"], train=True
    )
    val_set = UnderwaterDataset(
        d["root"], d["split_file"], "val", d["num_frames"], d["image_size"], train=False
    )
    idx = np.linspace(0, len(val_set) - 1, min(val_samples, len(val_set))).astype(int)
    val_loader = DataLoader(
        Subset(val_set, idx.tolist()),
        batch_size=1,
        shuffle=False,
        num_workers=d["num_workers"],
    )
    sampler = WeightedRandomSampler(
        train_set.category_weights(), num_samples=per_epoch, replacement=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=t["batch_size"],
        sampler=sampler,
        num_workers=d["num_workers"],
        pin_memory=True,
    )
    print(f"train: {len(train_set)} | val pool: {len(val_set)} | per epoch: {per_epoch}")

    model = build_model(cfg).to(device)
    print("parameters (M): {:.2f}".format(sum(p.numel() for p in model.parameters()) / 1e6))
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=t["lr"])

    start, best_psnr = 0, -1.0
    last_path = out_dir / "last.pth"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start = ck["epoch"] + 1
        best_psnr = ck["best_psnr"]
        print("resumed from epoch", ck["epoch"])

    log_path = out_dir / "log.csv"
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if new_log:
        writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "sec_per_iter", "minutes"])

    limit = args.max_hours * 3600
    t0 = time.time()
    last_dur = 0.0
    for epoch in range(start, maxepoch):
        if limit and (time.time() - t0) + last_dur > limit:
            print("time budget reached, stopping")
            break

        model.train()
        e0 = time.time()
        run, n = 0.0, 0
        for s in tqdm(train_loader, desc=f"epoch {epoch}"):
            x = s["image"].to(device)
            y = s["groundtruth"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = run / n
        train_time = time.time() - e0

        val_loss, val_psnr = validate(model, val_loader, device, criterion, sample_dir, epoch)
        last_dur = time.time() - e0

        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_psnr": max(best_psnr, val_psnr),
            },
            last_path,
        )
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            atomic_save(model.state_dict(), out_dir / "best.pth")

        sec_iter = train_time / max(1, len(train_loader))
        writer.writerow(
            [epoch, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_psnr:.3f}",
             f"{sec_iter:.3f}", f"{last_dur / 60:.1f}"]
        )
        log_file.flush()
        msg = (
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_psnr={val_psnr:.2f} sec/iter={sec_iter:.3f} epoch_min={last_dur / 60:.1f}"
        )
        if device.type == "cuda":
            msg += f" max_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.1f}"
        print(msg)

    log_file.close()
    print("total hours: {:.2f}".format((time.time() - t0) / 3600))


if __name__ == "__main__":
    main()