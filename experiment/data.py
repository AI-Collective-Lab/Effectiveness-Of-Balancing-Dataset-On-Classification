"""Fixed split and image cache for the corrected re-run.

    python data.py                 # writes splits/split.csv (once) and the 64x64 image cache

The split is stratified per class, 80/10/10, seed 42, over the original PlantVillage colour
images, where each leaf appears exactly once. It is committed to the repo so every method
uses identical train / val / test images. The cache (uint8, 64x64) lives next to the dataset,
not in the repo.
"""
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_ROOT = Path(os.environ.get("PLANTVILLAGE_ROOT", r"D:/Datasets/PlantVillage/color"))
CACHE = DATA_ROOT.parent / "cache64.npz"
SPLIT_CSV = Path(__file__).resolve().parent / "splits" / "split.csv"
IMG_SIZE = 64
SPLIT_SEED = 42


def make_split():
    if SPLIT_CSV.exists():
        return
    classes = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir())
    files, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted((DATA_ROOT / c).iterdir()):
            if f.is_file():
                files.append(f"{c}/{f.name}")
                labels.append(i)
    tr, rest, ytr, yrest = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=SPLIT_SEED)
    va, te, _, _ = train_test_split(rest, yrest, test_size=0.5, stratify=yrest, random_state=SPLIT_SEED)
    which = {**{f: "train" for f in tr}, **{f: "val" for f in va}, **{f: "test" for f in te}}
    SPLIT_CSV.parent.mkdir(exist_ok=True)
    with open(SPLIT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "class", "split"])
        for f in files:
            w.writerow([f, f.split("/")[0], which[f]])
    print(f"split written: {len(tr)} train / {len(va)} val / {len(te)} test")


def read_split():
    with open(SPLIT_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    classes = sorted({r["class"] for r in rows})
    return rows, classes


def build_cache():
    """All images resized once to 64x64 (bilinear, as the original transforms.Resize)."""
    if CACHE.exists():
        return
    rows, classes = read_split()
    idx = {c: i for i, c in enumerate(classes)}
    x = np.empty((len(rows), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    for k, r in enumerate(rows):
        with Image.open(DATA_ROOT / r["file"]) as im:
            x[k] = np.asarray(im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
        if k % 5000 == 0:
            print(f"  cached {k}/{len(rows)}")
    y = np.array([idx[r["class"]] for r in rows], np.int64)
    split = np.array([r["split"] for r in rows])
    np.savez(CACHE, x=x, y=y, split=split, classes=np.array(classes), files=np.array([r["file"] for r in rows]))
    print(f"cache written: {CACHE} ({x.nbytes / 2**20:.0f} MB)")


def load():
    """{split: (uint8 images NHWC, int labels)}, class names."""
    z = np.load(CACHE)
    out = {s: (z["x"][z["split"] == s], z["y"][z["split"] == s]) for s in ("train", "val", "test")}
    return out, [str(c) for c in z["classes"]]


if __name__ == "__main__":
    make_split()
    build_cache()
