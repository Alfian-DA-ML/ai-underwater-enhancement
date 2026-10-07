import random
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

SRC = Path("data")
DST = Path("data_processed")
SHORT = 512
MULT = 8
JPG_QUALITY = 95
MAP = {"raw": "input", "reference": "gt"}


def target_size(h, w):
    s = min(SHORT, min(h, w))
    if h <= w:
        nh, nw = s, w * s / h
    else:
        nw, nh = s, h * s / w
    nh = max(MULT, round(nh / MULT) * MULT)
    nw = max(MULT, round(nw / MULT) * MULT)
    return nh, nw


def process(job):
    src, dst = job
    if dst.exists():
        return 0
    img = cv2.imread(str(src))
    if img is None:
        return 1
    h, w = img.shape[:2]
    nh, nw = target_size(h, w)
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])
    return 0


def resize_all():
    jobs = []
    for old, new in MAP.items():
        for f in (SRC / old).rglob("*"):
            if f.suffix.lower() in {".jpg", ".jpeg"}:
                jobs.append((f, DST / new / f.relative_to(SRC / old)))

    print("total files:", len(jobs))
    with ThreadPoolExecutor(max_workers=8) as ex:
        errors = sum(tqdm(ex.map(process, jobs), total=len(jobs)))
    print("failed to read:", errors)


def verify():
    counts = {}
    for sub in ["input", "gt"]:
        root = DST / sub
        print(f"\n=== {sub} ===")
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            n = len(list(d.rglob("*.jpg")))
            counts[(sub, d.name)] = n
            print(f"{d.name}: {n} frames")

    print("\n=== input vs gt frame count ===")
    for (sub, name), n in counts.items():
        if sub == "input":
            m = counts.get(("gt", name))
            print(f"{name}: {'OK' if n == m else 'MISMATCH'} ({n} vs {m})")

    f = random.choice(list((DST / "input").rglob("*.jpg")))
    g = DST / "gt" / f.relative_to(DST / "input")
    print("\nsample:", f.relative_to(DST / "input"))
    print("input:", cv2.imread(str(f)).shape, "| gt:", cv2.imread(str(g)).shape)


if __name__ == "__main__":
    resize_all()
    verify()