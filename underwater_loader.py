import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UnderwaterDataset(Dataset):
    def __init__(self, root, split_file, split, numframes=5, crop=256, train=True):
        self.root = root
        self.half = numframes // 2
        self.crop = crop
        self.train = train

        with open(split_file, "r") as f:
            split_data = json.load(f)

        self.samples = []
        for cat, parts in split_data.items():
            files = os.listdir(os.path.join(root, "input", cat))
            existing = {
                int(os.path.splitext(n)[0])
                for n in files
                if n.lower().endswith((".jpg", ".jpeg"))
            }
            for seg in parts[split]:
                for name in seg:
                    stem, ext = os.path.splitext(name)
                    c = int(stem)
                    window = range(c - self.half, c + self.half + 1)
                    if all(k in existing for k in window):
                        self.samples.append((cat, c, len(stem), ext))

    def __len__(self):
        return len(self.samples)

    def category_weights(self):
        counts = {}
        for cat, *_ in self.samples:
            counts[cat] = counts.get(cat, 0) + 1
        return [1.0 / counts[cat] for cat, *_ in self.samples]

    def _crop_and_flip(self, image, gt):
        if self.crop:
            h, w = image.shape[:2]
            ch, cw = min(self.crop, h), min(self.crop, w)
            if self.train:
                top = np.random.randint(0, h - ch + 1)
                left = np.random.randint(0, w - cw + 1)
            else:
                top, left = 0, 0
            image = image[top:top + ch, left:left + cw]
            gt = gt[top:top + ch, left:left + cw]
        if self.train:
            op = np.random.randint(0, 3)
            if op < 2:
                image = np.flip(image, op)
                gt = np.flip(gt, op)
        return image, gt

    def __getitem__(self, idx):
        cat, c, width, ext = self.samples[idx]
        in_dir = os.path.join(self.root, "input", cat)
        gt_path = os.path.join(self.root, "gt", cat, f"{c:0{width}d}{ext}")

        frames = []
        for i in range(c - self.half, c + self.half + 1):
            path = os.path.join(in_dir, f"{i:0{width}d}{ext}")
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype("float32") / 255.0
            frames.append(img[..., np.newaxis])
        image = np.concatenate(frames, axis=3)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype("float32") / 255.0

        image, gt = self._crop_and_flip(image, gt)

        image = torch.from_numpy(image.transpose((3, 2, 0, 1)).copy())
        gt = torch.from_numpy(gt.transpose((2, 0, 1)).copy())
        image = (image - 0.5) / 0.5
        gt = (gt - 0.5) / 0.5

        return {"image": image, "groundtruth": gt, "img_id": f"{cat}-{c:0{width}d}"}