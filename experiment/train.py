"""Train and evaluate one method with one seed.

    python train.py --method baseline --seed 0 --level full --width 1.0

Same CNN, input size, augmentation, optimiser and epochs for every method (see PROTOCOL.md).
The epoch is chosen on validation macro-F1; the test split is scored once, at the end.
Writes results/<method>_<level>_w<width>_seed<seed>.json and a matching _history.csv.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import data
import metrics

OUT = Path(__file__).resolve().parent / "results"
EPOCHS = 20
BATCH = 128
LR = 1e-3


class CropDiseaseCNN(nn.Module):
    """The project's original CNN (Custom_CNN.py), with the nn.Covn2d typo fixed.
    width scales every layer: 1.0 is the original, 0.25 the small model of the capacity factor."""
    def __init__(self, num_classes, width=1.0):
        super().__init__()
        chans = [3] + [max(4, round(c * width)) for c in (32, 64, 128, 256)]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(chans, chans[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm2d(b) for b in chans[1:])
        self.pool, self.gap = nn.MaxPool2d(2, 2), nn.AdaptiveAvgPool2d(1)
        hidden = max(16, round(128 * width))
        self.fc1, self.dropout, self.fc2 = nn.Linear(chans[-1], hidden), nn.Dropout(0.5), nn.Linear(hidden, num_classes)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.pool(F.relu(bn(conv(x))))
        x = self.gap(x).flatten(1)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


def augment(x):
    """Per-image augmentation on the GPU, matching the original: horizontal flip, rotation ±15°,
    brightness / contrast ±0.1. x: float (N, 3, H, W) in [0, 1]."""
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    a = (torch.rand(n, device=x.device) * 30 - 15) * math.pi / 180
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = a.cos(), -a.sin(), a.sin(), a.cos()
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 0.2 - 0.1)
    c = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 0.2 - 0.1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x * b - mean) * c + mean).clamp(0, 1)


def to_tensor(imgs, device):
    return torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).float().div_(255)


@torch.no_grad()
def predict(model, x, norm, device):
    model.eval()
    probs = []
    for i in range(0, len(x), 512):
        probs.append(F.softmax(model(norm(to_tensor(x[i:i + 512], device))), 1).cpu())
    return torch.cat(probs).numpy()


CLASSES = []        # filled in main(); used by train_set
LEVEL_SEED = 1234   # fixed: every method and seed sees the same reduced minority sets


def apply_level(x, y, classes, level):
    """Keep `level` real training images per minority class (nested: the 30 are among the 100)."""
    if level == "full":
        return x, y
    keep_n = int(level)
    rng = np.random.default_rng(LEVEL_SEED)
    keep = np.ones(len(y), bool)
    for c in metrics.MINORITY_CLASSES:
        idx = np.flatnonzero(y == classes.index(c))
        idx = idx[rng.permutation(len(idx))]
        keep[idx[keep_n:]] = False
    return x[keep], y[keep]


def train_set(method, x, y, num_classes, rng):
    """Training images and the loss weights for one method. Only the training split is touched."""
    weights = None
    if method == "baseline":
        pass
    elif method == "class_weighted":
        counts = np.bincount(y, minlength=num_classes)
        weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)
    elif method == "oversampling":
        # repeat real minority images (drawn at random, with replacement) up to the target;
        # the usual augmentation is applied on the fly, so repeats are not pixel-identical
        extra = []
        for c in metrics.MINORITY_CLASSES:
            idx = np.flatnonzero(y == CLASSES.index(c))
            need = max(0, metrics.TARGET_PER_CLASS - len(idx))
            extra.append(rng.choice(idx, need, replace=True))
        extra = np.concatenate(extra)
        x, y = np.concatenate([x, x[extra]]), np.concatenate([y, y[extra]])
    else:
        raise NotImplementedError(f"method {method!r} is added in a later step")
    return x, y, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--level", default="full", choices=metrics.LEVELS)
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, classes = data.load()
    CLASSES[:] = classes
    (xtr, ytr), (xva, yva), (xte, yte) = splits["train"], splits["val"], splits["test"]
    xtr, ytr = apply_level(xtr, ytr, classes, args.level)
    rng = np.random.default_rng(args.seed)
    xtr, ytr, weights = train_set(args.method, xtr, ytr, len(classes), rng)

    # normalisation statistics from the training split only
    mean = torch.tensor(xtr.reshape(-1, 3).mean(0) / 255, device=device).view(1, 3, 1, 1).float()
    std = torch.tensor(xtr.reshape(-1, 3).std(0) / 255, device=device).view(1, 3, 1, 1).float()
    norm = lambda t: (t - mean) / std

    model = CropDiseaseCNN(len(classes), args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=3)
    crit = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device), label_smoothing=0.1)

    OUT.mkdir(exist_ok=True)
    tag = f"{args.method}_{args.level}_w{args.width:g}_seed{args.seed}"
    best_f1, best_state, history = -1, None, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(xtr))
        loss_sum = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            xb = norm(augment(to_tensor(xtr[b], device)))
            yb = torch.from_numpy(ytr[b]).to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += loss.item() * len(b)
        pva = predict(model, xva, norm, device).argmax(1)
        val_f1 = f1_score(yva, pva, average="macro", zero_division=0)
        sched.step(val_f1)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(xtr), "val_macro_f1": val_f1,
                        "val_acc": float((pva == yva).mean()), "lr": opt.param_groups[0]["lr"],
                        "seconds": round(time.time() - t0, 1)})
        print(f"[{tag}] epoch {epoch:2d}  loss {loss_sum / len(xtr):.4f}  val macro-F1 {val_f1:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - t0

    # the test split is scored exactly once, with the epoch chosen on validation
    model.load_state_dict(best_state)
    score = predict(model, xte, norm, device)
    pred = score.argmax(1)
    report = metrics.evaluate(yte, pred, score, classes)
    report.update(method=args.method, seed=args.seed, level=args.level, width=args.width,
                  n_params=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch, best_val_macro_f1=best_f1,
                  train_seconds=round(train_seconds, 1), n_train=int(len(xtr)), epochs=args.epochs,
                  ci95_minority_macro_f1=metrics.bootstrap_ci(yte, pred, classes, "minority_macro_f1"),
                  ci95_macro_f1=metrics.bootstrap_ci(yte, pred, classes, "macro_f1"))
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=1))
    with open(OUT / f"{tag}_history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    print(f"[{tag}] TEST  minority macro-F1 {report['minority_macro_f1']:.4f}  macro-F1 {report['macro_f1']:.4f}  "
          f"bal.acc {report['balanced_accuracy']:.4f}  acc {report['accuracy']:.4f}  (best epoch {best_epoch})",
          flush=True)


if __name__ == "__main__":
    main()
