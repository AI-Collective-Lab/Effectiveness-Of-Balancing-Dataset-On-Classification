"""Evaluation for the corrected re-run, fixed before any model is trained.

Primary measure:  minority macro-F1, the mean F1 over the MINORITY_CLASSES
                  (the 7 classes with fewer than 500 images in the original
                  PlantVillage colour set). Every method is judged on this.
Co-primary:       macro-F1 over all 38 classes.
Secondary:        balanced accuracy, MCC, macro PR-AUC over the minority
                  classes, per-class precision / recall / F1 for the minority
                  classes and for any class a method balanced.
Context only:     overall accuracy (dominated by the 3 largest classes).
Not used:         ROC-AUC (near-perfect on imbalanced one-vs-rest problems).

All numbers come from the untouched test split, which is never used to pick
an epoch or a setting. Each method runs with 5 seeds; results are reported as
mean ± std across seeds, plus a 95% bootstrap CI on the test set per seed.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)

MINORITY_THRESHOLD = 500          # images per class in the full dataset
MINORITY_CLASSES = [              # folder names; fixed from the class counts in the README
    "Potato___healthy",                       # 152
    "Apple___Cedar_apple_rust",               # 275
    "Peach___healthy",                        # 360
    "Raspberry___healthy",                    # 371
    "Tomato___Tomato_mosaic_virus",           # 373
    "Grape___healthy",                        # 423
    "Strawberry___healthy",                   # 456
]
SEEDS = (0, 1, 2, 3, 4)
LEVELS = ("full", "100", "30")   # real training images kept per minority class
WIDTHS = (1.0, 0.25)             # CropDiseaseCNN and a quarter-width version
TARGET_PER_CLASS = 874            # training-split median; every minority class is topped up to this
GAN_SCENARIOS = ("gan_s1_cond_minority", "gan_s2_cond_all", "gan_s3_per_class")


def minority_from_counts(counts, threshold=MINORITY_THRESHOLD):
    """Recompute the minority list from {class: n_images}; used to check MINORITY_CLASSES."""
    return sorted((c for c, n in counts.items() if n < threshold), key=counts.get)


def evaluate(y_true, y_pred, y_score, class_names, balanced_classes=()):
    """All measures for one model on one test set.

    y_true, y_pred: int labels, shape (n,). y_score: softmax probabilities, shape (n, n_classes).
    balanced_classes: class names the method added data for (reported per class as well).
    """
    y_true, y_pred, y_score = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score)
    labels = np.arange(len(class_names))
    idx = {c: i for i, c in enumerate(class_names)}
    minority = [idx[c] for c in MINORITY_CLASSES if c in idx]
    assert len(minority) == len(MINORITY_CLASSES), "class names do not match MINORITY_CLASSES"

    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    ap = []
    for i in minority:  # average precision = area under the PR curve, one-vs-rest
        pos = (y_true == i)
        ap.append(average_precision_score(pos, y_score[:, i]) if pos.any() else np.nan)

    report = {
        "minority_macro_f1": float(np.mean(f[minority])),                      # PRIMARY
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "minority_macro_pr_auc": float(np.nanmean(ap)),
        "minority_macro_recall": float(np.mean(r[minority])),
        "minority_macro_precision": float(np.mean(p[minority])),
        "accuracy": float(accuracy_score(y_true, y_pred)),                     # context only
    }
    per_class = {}
    for c in dict.fromkeys(list(MINORITY_CLASSES) + list(balanced_classes)):
        i = idx[c]
        per_class[c] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
    report["per_class"] = per_class
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return report


def bootstrap_ci(y_true, y_pred, class_names, metric="minority_macro_f1", n_boot=1000, seed=0, alpha=0.05):
    """95% CI for a label-based metric by resampling test images with replacement."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.arange(len(class_names))
    minority = [class_names.index(c) for c in MINORITY_CLASSES]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = rng.integers(0, len(y_true), len(y_true))
        t, q = y_true[s], y_pred[s]
        if metric == "minority_macro_f1":
            f = precision_recall_fscore_support(t, q, labels=labels, zero_division=0)[2]
            vals.append(np.mean(f[minority]))
        elif metric == "macro_f1":
            vals.append(f1_score(t, q, labels=labels, average="macro", zero_division=0))
        else:
            raise ValueError(metric)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


SUMMARY_KEYS = ("minority_macro_f1", "macro_f1", "balanced_accuracy", "mcc",
                "minority_macro_pr_auc", "minority_macro_recall", "minority_macro_precision", "accuracy")


def summarize_seeds(reports):
    """{metric: (mean, std)} across the per-seed reports of one method."""
    return {k: (float(np.mean([r[k] for r in reports])), float(np.std([r[k] for r in reports], ddof=1)))
            for k in SUMMARY_KEYS}


def gan_works(summaries, gan="gan", rivals=("baseline", "class_weighted", "oversampling"), key="minority_macro_f1"):
    """Decision rule, fixed in advance: GAN balancing 'works' only if its mean minority
    macro-F1 beats every rival by more than the larger seed std of the two methods."""
    g_mean, g_std = summaries[gan][key]
    verdicts = {}
    for r in rivals:
        r_mean, r_std = summaries[r][key]
        margin = g_mean - r_mean
        verdicts[r] = {"margin": margin, "noise": max(g_std, r_std), "beats": margin > max(g_std, r_std)}
    return all(v["beats"] for v in verdicts.values()), verdicts


# ---------------------------------------------------------------- GAN image quality
def kid(feat_real, feat_fake, n_subsets=50, subset_size=100, seed=0):
    """Kernel Inception Distance (unbiased MMD², cubic polynomial kernel), mean and std over
    random subsets. Unlike FID it is unbiased for small samples, which matters here
    (e.g. ~122 Potato-healthy training images). Features: Inception-v3 pool (2048-d)."""
    feat_real, feat_fake = np.asarray(feat_real, np.float64), np.asarray(feat_fake, np.float64)
    d = feat_real.shape[1]
    m = min(subset_size, len(feat_real), len(feat_fake))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_subsets):
        x = feat_real[rng.choice(len(feat_real), m, replace=False)]
        y = feat_fake[rng.choice(len(feat_fake), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def inception_features(images, device="cuda", batch_size=64):
    """2048-d Inception-v3 pool features for a float tensor of images in [0, 1], shape (n, 3, H, W)."""
    import torch
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights, inception_v3
    net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True).to(device).eval()
    net.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = images[i:i + batch_size].to(device)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            out.append(net((x - mean) / std).cpu())
    return torch.cat(out).numpy()
