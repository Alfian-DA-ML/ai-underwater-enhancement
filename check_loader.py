from collections import Counter
from underwater_loader import UnderwaterDataset

R = "data_processed"
S = "data_processed/split.json"

for split in ["train", "val"]:
    ds = UnderwaterDataset(R, S, split, 5, crop=256, train=(split == "train"))
    print(split, len(ds), dict(Counter(s[0] for s in ds.samples)))

ds = UnderwaterDataset(R, S, "train", 5, crop=256, train=True)
seen = set()
for i, s in enumerate(ds.samples):
    if s[0] not in seen:
        seen.add(s[0])
        x = ds[i]
        print(x["img_id"], tuple(x["image"].shape), tuple(x["groundtruth"].shape))