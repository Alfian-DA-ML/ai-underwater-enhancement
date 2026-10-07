import json
from pathlib import Path
import cv2

ROOT = Path("data_processed")
VAL_FRAC = 0.10
GAP = 8
MIN_VAL = 16
MIN_TRAIN = 32


def frame_key(name):
    return int(Path(name).stem)


def segments(names):
    segs, cur = [], [names[0]]
    for prev, name in zip(names, names[1:]):
        if frame_key(name) - frame_key(prev) != 1:
            segs.append(cur)
            cur = []
        cur.append(name)
    segs.append(cur)
    return segs


def split_segment(seg):
    n = len(seg)
    n_val = max(MIN_VAL, round(n * VAL_FRAC))
    train_end = n - n_val - GAP
    if train_end < MIN_TRAIN:
        return seg, []
    return seg[:train_end], seg[n - n_val:]


def main():
    in_root = ROOT / "input"
    gt_root = ROOT / "gt"
    split = {}
    totals = {"total": 0, "train": 0, "val": 0, "gap": 0}

    for d in sorted(p for p in in_root.iterdir() if p.is_dir()):
        names = sorted((f.name for f in d.glob("*.jpg")), key=frame_key)
        gt_names = sorted(
            (f.name for f in (gt_root / d.name).glob("*.jpg")), key=frame_key
        )
        assert names == gt_names, f"{d.name}: input and gt filenames differ"

        h1, w1 = cv2.imread(str(d / names[0])).shape[:2]
        h2, w2 = cv2.imread(str(gt_root / d.name / names[0])).shape[:2]
        assert (h1, w1) == (h2, w2), f"{d.name}: input and gt sizes differ"

        segs = segments(names)
        train, val = [], []
        for seg in segs:
            t, v = split_segment(seg)
            if t:
                train.append(t)
            if v:
                val.append(v)

        train_set = {n for s in train for n in s}
        val_set = {n for s in val for n in s}
        assert not (train_set & val_set), f"{d.name}: train/val overlap"

        n_total = len(names)
        n_train = len(train_set)
        n_val = len(val_set)
        n_gap = n_total - n_train - n_val
        print(
            f"{d.name}: size={w1}x{h1} segments={len(segs)} total={n_total} "
            f"train={n_train} val={n_val} gap={n_gap}"
        )
        if n_val == 0:
            print(f"  WARNING: {d.name} has no validation frames")

        split[d.name] = {"train": train, "val": val}
        totals["total"] += n_total
        totals["train"] += n_train
        totals["val"] += n_val
        totals["gap"] += n_gap

    print("\nTOTAL:", totals)
    with open(ROOT / "split.json", "w") as f:
        json.dump(split, f)
    print("saved:", ROOT / "split.json")


if __name__ == "__main__":
    main()