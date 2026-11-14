"""Summarise the two remaining checks of PROTOCOL.md: train-on-synthetic-test-on-real (tstr_*) and the
pretrained ResNet-18 (r18_*). Mean +- std over seeds 0-2 of minority macro-F1 and macro-F1, test split."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent / "results"
LEVELS = ("full", "100", "30")


def cell(prefix, method, level, seeds=(0, 1, 2), width="1"):
    v = [json.loads((R / f"{prefix}{method}_{level}_w{width}_seed{s}.json").read_text()) for s in seeds]
    m, f = [x["minority_macro_f1"] for x in v], [x["macro_f1"] for x in v]
    return np.mean(m), np.std(m), np.mean(f), np.std(f)


def row(label, prefix, method):
    out = [f"{label:34s}"]
    for lv in LEVELS:
        m, s, f, fs = cell(prefix, method, lv)
        out.append(f"{m:.3f}±{s:.3f} (all {f:.3f})")
    return "  ".join(out)


print("== Train on synthetic, test on real (main CNN, seeds 0-2): minority macro-F1 full / 100 / 30")
print(f"{'':34s}" + "  ".join(f"{l:>22s}" for l in LEVELS))
print(row("baseline (real data only)", "", "baseline"))
print(row("oversampling (real, repeated)", "", "oversampling"))
for m in ("s1_cond_minority", "s2_cond_all", "s3_per_class"):
    print(row(f"synthetic only: {m}", "tstr_", m))
print("\n== Pretrained ResNet-18 (seeds 0-2): minority macro-F1 full / 100 / 30")
print(row("baseline", "r18_", "baseline"))
print(row("oversampling", "r18_", "oversampling"))
