import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="test_on_video/footage_1.mp4")
parser.add_argument("--ckpt", default="checkpoints/best.pth")
parser.add_argument("--config", default="configs/underwater.yml")
parser.add_argument("--repo", default="BVI-Mamba")
parser.add_argument("--out", default="results_infer")
parser.add_argument("--start", type=float, default=0.0)
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--short", type=int, default=512)
parser.add_argument("--stride", type=int, default=192)
parser.add_argument("--tile_batch", type=int, default=6)
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


def read_frames(path, start, seconds, short):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    limit = int(seconds * fps)
    frames = []
    while len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = short / min(h, w)
        if s < 1:
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames, fps


def positions(length, tile, stride):
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile, stride))
    pos.append(length - tile)
    return pos


@torch.inference_mode()
def enhance(model, norm, t, ys, xs, tile, half, device, tile_batch, wmap):
    n = len(norm)
    idx = [min(max(i, 0), n - 1) for i in range(t - half, t + half + 1)]
    h, w = norm[0].shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 1), np.float32)
    coords = [(y, x) for y in ys for x in xs]
    for b in range(0, len(coords), tile_batch):
        chunk = coords[b:b + tile_batch]
        batch = np.stack(
            [np.stack([norm[i][y:y + tile, x:x + tile] for i in idx]) for y, x in chunk]
        )
        inp = torch.from_numpy(batch).permute(0, 1, 4, 2, 3).contiguous().to(device)
        out = model(inp).clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()
        for (y, x), o in zip(chunk, out):
            acc[y:y + tile, x:x + tile] += o * wmap
            wsum[y:y + tile, x:x + tile] += wmap
    res = acc / wsum
    return ((res * 0.5 + 0.5) * 255.0).round().clip(0, 255).astype(np.uint8)


def main():
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    tile = cfg["dataset"]["image_size"]
    half = cfg["dataset"]["num_frames"] // 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    frames, fps = read_frames(args.video, args.start, args.seconds, args.short)
    if not frames:
        raise RuntimeError("no frames read")
    h, w = frames[0].shape[:2]
    assert min(h, w) >= tile, "frame is smaller than the tile size"
    print(f"frames: {len(frames)} | size: {w}x{h} | fps: {fps:.2f}")

    ys = positions(h, tile, args.stride)
    xs = positions(w, tile, args.stride)
    print(f"tiles per frame: {len(ys) * len(xs)}")

    norm = [(f.astype("float32") / 255.0 - 0.5) / 0.5 for f in frames]
    w1 = np.hanning(tile + 2)[1:-1].astype("float32")
    wmap = np.outer(w1, w1)[..., None]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_enh = cv2.VideoWriter(str(out_dir / "enhanced.mp4"), fourcc, fps, (w, h))
    vw_side = cv2.VideoWriter(str(out_dir / "side_by_side.mp4"), fourcc, fps, (w * 2, h))
    still_ids = set(np.linspace(0, len(frames) - 1, 4).astype(int).tolist())

    t0 = time.time()
    for t in tqdm(range(len(frames))):
        enh = enhance(model, norm, t, ys, xs, tile, half, device, args.tile_batch, wmap)
        vw_enh.write(enh)
        side = np.concatenate([frames[t], enh], axis=1)
        vw_side.write(side)
        if t in still_ids:
            cv2.imwrite(str(out_dir / f"still_{t:04d}.jpg"), side)
    vw_enh.release()
    vw_side.release()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()