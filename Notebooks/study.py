"""Helpers shared by the notebooks: read the saved result files of the corrected study.

Nothing here trains a model. The training code is in ../experiment; these notebooks only
read what it wrote (experiment/results, experiment/gan_results, experiment/splits)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiment"
sys.path.insert(0, str(EXP))
import metrics  # noqa: E402  (the study's own metric definitions and decision rule)

RESULTS, GAN_RESULTS = EXP / "results", EXP / "gan_results"
LEVELS = list(metrics.LEVELS)                 # full / 100 / 30 real minority images per class
METHODS = ["baseline", "class_weighted", "oversampling"] + list(metrics.GAN_SCENARIOS)
LABEL = {"baseline": "Baseline", "class_weighted": "Class-weighted", "oversampling": "Oversampling",
         "gan_s1_cond_minority": "GAN S1", "gan_s2_cond_all": "GAN S2", "gan_s3_per_class": "GAN S3"}
COLOR = {"baseline": "#2a78d6", "class_weighted": "#eb6834", "oversampling": "#1baf7a",
         "gan_s1_cond_minority": "#eda100", "gan_s2_cond_all": "#e87ba4", "gan_s3_per_class": "#008300"}


def load_runs(method, level, width="1", seeds=None, prefix=""):
    """All per-seed reports for one cell of the design."""
    out = []
    for p in sorted(RESULTS.glob(f"{prefix}{method}_{level}_w{width}_seed*.json")):
        if p.name.endswith("_history.csv"):
            continue
        r = json.loads(p.read_text())
        if seeds is None or r["seed"] in seeds:
            out.append(r)
    return out


def table(key="minority_macro_f1", width="1", methods=METHODS, seeds=None, prefix=""):
    """DataFrame of 'mean ± std' strings (rows = methods, columns = levels), plus the numeric means."""
    text, mean = {}, {}
    for m in methods:
        text[LABEL.get(m, m)], mean[m] = {}, {}
        for lv in LEVELS:
            v = [r[key] for r in load_runs(m, lv, width, seeds, prefix)]
            text[LABEL.get(m, m)][lv] = f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
            mean[m][lv] = float(np.mean(v))
    return pd.DataFrame(text).T[LEVELS], pd.DataFrame(mean).T[LEVELS]


def summaries(level, width="1"):
    """{method: {metric: (mean, std)}} for the decision rule."""
    return {m: metrics.summarize_seeds(load_runs(m, level, width)) for m in METHODS}


def split_counts():
    df = pd.read_csv(EXP / "splits" / "split.csv")
    return df.pivot_table(index="class", columns="split", values="file", aggfunc="count").fillna(0).astype(int)
