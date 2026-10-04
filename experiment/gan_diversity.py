"""Diversity of the generated images, per scenario and level (diagnostic for mode collapse).

    python gan_diversity.py      # writes gan_results/diversity.json

Score = mean pairwise pixel distance between generated images of a class, divided by the same
for its real training images. About 1: as varied as the real images. Near 0: mode collapse
(the GAN produces almost the same image whatever its random input).
Added 2026-10-04 after S1's progress sheets showed collapse; it is a diagnostic, not a decision rule.
"""
import json
from pathlib import Path

import numpy as np

import data
import metrics
from train import apply_level


def mean_pairwise(a, n=200, seed=0):
    a = a.reshape(len(a), -1).astype(np.float32) / 255
    a = a[np.random.default_rng(seed).choice(len(a), min(n, len(a)), replace=False)]
    sq = (a ** 2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * a @ a.T, 0))
    return float(d[np.triu_indices(len(a), 1)].mean())


def main():
    splits, classes = data.load()
    X, Y = splits["train"]
    out = {}
    for f in sorted((data.DATA_ROOT.parent / "gan").glob("*.npz")):
        name, level = f.stem, f.stem.rsplit("_", 1)[1]
        g = np.load(f)
        x, y = apply_level(X, Y, classes, level)
        per = {c: mean_pairwise(g["x"][g["y"] == classes.index(c)]) / mean_pairwise(x[y == classes.index(c)])
               for c in metrics.MINORITY_CLASSES}
        out[name] = {"mean": float(np.mean(list(per.values()))), "per_class": per}
        print(f"{name:24s} diversity {out[name]['mean']:.2f}")
    (Path(__file__).resolve().parent / "gan_results" / "diversity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
