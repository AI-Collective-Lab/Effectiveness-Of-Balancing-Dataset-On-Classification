"""DCGANs for the three balancing scenarios (see PROTOCOL.md).

    python gan.py --scenario s3_per_class   --level full
    python gan.py --scenario s1_cond_minority --level 30
    python gan.py --scenario s2_cond_all    --level 100

S1  one conditional DCGAN on the 7 minority classes
S2  one conditional DCGAN on all 38 classes
S3  seven plain DCGANs, one per minority class

Every GAN is trained on the training split of the given imbalance level only, with the project's
original DCGAN architecture (64x64), DiffAugment (colour, translation, cutout) on real and fake
images, and a fixed iteration budget. Every CHECK_EVERY iterations the KID of each minority class
against its real training images is computed and the best checkpoint is kept. The best generator
then tops each minority class up to metrics.TARGET_PER_CLASS images.

Outputs
  D:/Datasets/PlantVillage/gan/<scenario>_<level>.npz   generated images (uint8 64x64) + labels
  gan_results/<scenario>_<level>.json                   KID, memorisation check, iterations, time
  gan_results/<scenario>_<level>.png                    sample grid, real vs generated per class
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import data
import metrics
from train import apply_level

HERE = Path(__file__).resolve().parent
GEN_DIR = data.DATA_ROOT.parent / "gan"
RES_DIR = HERE / "gan_results"
Z_DIM, EMB = 100, 50
ITERS = 20000
CHECK_EVERY = 2000
BATCH = 64
LR = 2e-4
GAN_SEED = 0


# ---------------------------------------------------------------- DiffAugment (Zhao et al., 2020)
def diff_augment(x):
    """Same random colour / translation / cutout for the images D sees, real and fake alike.
    x in [-1, 1], shape (N, 3, 64, 64)."""
    n = x.shape[0]
    dev = x.device
    x = x + (torch.rand(n, 1, 1, 1, device=dev) - 0.5)                                   # brightness
    m = x.mean(1, keepdim=True)
    x = (x - m) * (torch.rand(n, 1, 1, 1, device=dev) * 2) + m                             # saturation
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * (torch.rand(n, 1, 1, 1, device=dev) + 0.5) + m                           # contrast
    # translation by up to 1/8 of the image, zero padded
    s = x.shape[-1] // 8
    tx = torch.randint(-s, s + 1, (n, 1, 1), device=dev)
    ty = torch.randint(-s, s + 1, (n, 1, 1), device=dev)
    gy, gx = torch.meshgrid(torch.arange(x.shape[2], device=dev), torch.arange(x.shape[3], device=dev), indexing="ij")
    gx = (gx.unsqueeze(0) + tx).clamp(-1, x.shape[3])
    gy = (gy.unsqueeze(0) + ty).clamp(-1, x.shape[2])
    xp = F.pad(x, (1, 1, 1, 1))
    x = xp.permute(0, 2, 3, 1)[torch.arange(n, device=dev).view(-1, 1, 1), gy + 1, gx + 1].permute(0, 3, 1, 2)
    # cutout: zero a square of half the image size at a random centre
    half = x.shape[-1] // 4
    cx = torch.randint(0, x.shape[3], (n, 1, 1), device=dev)
    cy = torch.randint(0, x.shape[2], (n, 1, 1), device=dev)
    cols = torch.arange(x.shape[3], device=dev).view(1, 1, -1)
    rows = torch.arange(x.shape[2], device=dev).view(1, -1, 1)
    inside = ((cols - cx).abs() < half) & ((rows - cy).abs() < half)
    return x * (~inside).unsqueeze(1)


# ---------------------------------------------------------------- the project's DCGAN, optionally conditional
class Generator(nn.Module):
    def __init__(self, n_classes=0):
        super().__init__()
        self.emb = nn.Embedding(n_classes, EMB) if n_classes else None
        zin = Z_DIM + (EMB if n_classes else 0)
        def up(a, b, k=4, s=2, p=1):
            return [nn.ConvTranspose2d(a, b, k, s, p, bias=False), nn.BatchNorm2d(b), nn.ReLU(True)]
        self.net = nn.Sequential(*up(zin, 512, 4, 1, 0), *up(512, 256), *up(256, 128), *up(128, 64),
                                 nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z, y=None):
        if self.emb is not None:
            z = torch.cat([z, self.emb(y)], 1)
        return self.net(z.view(z.shape[0], -1, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, n_classes=0):
        super().__init__()
        self.emb = nn.Embedding(n_classes, 64 * 64) if n_classes else None
        cin = 3 + (1 if n_classes else 0)
        def down(a, b, bn=True):
            return [nn.Conv2d(a, b, 4, 2, 1, bias=False)] + ([nn.BatchNorm2d(b)] if bn else []) + [nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*down(cin, 64, False), *down(64, 128), *down(128, 256), *down(256, 512),
                                 nn.Conv2d(512, 1, 4, 1, 0, bias=False))   # logits; BCEWithLogits = sigmoid + BCE

    def forward(self, x, y=None):
        if self.emb is not None:
            x = torch.cat([x, self.emb(y).view(-1, 1, 64, 64)], 1)
        return self.net(x).view(-1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)


# ---------------------------------------------------------------- helpers
def to_gan(x_uint8, device):
    return torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div(127.5).sub(1)


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def generate(G, n, label, device, n_classes):
    G.eval()
    out = []
    for i in range(0, n, 256):
        k = min(256, n - i)
        z = torch.randn(k, Z_DIM, device=device)
        y = torch.full((k,), label, dtype=torch.long, device=device) if n_classes else None
        out.append(to_uint8(G(z, y)))
    G.train()
    return np.concatenate(out) if out else np.empty((0, 64, 64, 3), np.uint8)


def features(x_uint8, device):
    t = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float().div(255)
    return metrics.inception_features(t, device=device)


def memorisation(real, fake):
    """Nearest-neighbour distances in pixel space. Each generated image's distance to its closest
    real training image, compared with how close real images are to each other. A GAN that copies
    gives many generated images closer to a real one than real images ever are to each other."""
    r = real.reshape(len(real), -1).astype(np.float32) / 255
    f = fake.reshape(len(fake), -1).astype(np.float32) / 255
    def nn_dist(a, b, same=False):
        d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T
        if same:
            np.fill_diagonal(d, np.inf)
        return np.sqrt(np.maximum(d.min(1), 0))
    real_real = nn_dist(r, r, same=True)
    fake_real = nn_dist(f, r)
    floor = np.quantile(real_real, 0.05)
    return {"median_nn_real_to_real": float(np.median(real_real)),
            "median_nn_generated_to_real": float(np.median(fake_real)),
            "share_generated_closer_than_real_5pct": float((fake_real < floor).mean())}


def save_grid(path, rows):
    """rows: list of (label, real uint8 images, generated uint8 images); 8 real then 8 generated per row."""
    tile = 64
    grid = np.full((len(rows) * (tile + 4), 17 * (tile + 2), 3), 255, np.uint8)
    for i, (_, real, fake) in enumerate(rows):
        for j, img in enumerate(list(real[:8]) + [None] + list(fake[:8])):
            if img is not None:
                y0, x0 = i * (tile + 4), j * (tile + 2)
                grid[y0:y0 + tile, x0:x0 + tile] = img
    Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(path)


# ---------------------------------------------------------------- one GAN
def train_gan(x, y, n_classes, minority_labels, real_feats, device, log, name):
    """x: uint8 training images, y: labels in [0, n_classes) (unused when n_classes == 0).
    minority_labels: GAN labels whose KID decides the checkpoint."""
    torch.manual_seed(GAN_SEED)
    G, D = Generator(n_classes).to(device), Discriminator(n_classes).to(device)
    G.apply(init_weights); D.apply(init_weights)
    oG = torch.optim.Adam(G.parameters(), LR, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), LR, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(GAN_SEED)
    xt = to_gan(x, device)
    yt = torch.from_numpy(y).to(device) if n_classes else None
    best = (np.inf, None, 0)
    t0 = time.time()
    for it in range(1, ITERS + 1):
        b = torch.from_numpy(rng.integers(0, len(x), BATCH)).to(device)
        real = xt[b]
        yb = yt[b] if n_classes else None
        z = torch.randn(BATCH, Z_DIM, device=device)
        fake = G(z, yb)
        # D: real -> 0.9 (one-sided label smoothing, as in the original), fake -> 0
        lD = bce(D(diff_augment(real), yb), torch.full((BATCH,), 0.9, device=device)) + \
             bce(D(diff_augment(fake.detach()), yb), torch.zeros(BATCH, device=device))
        oD.zero_grad(set_to_none=True); lD.backward(); oD.step()
        lG = bce(D(diff_augment(fake), yb), torch.ones(BATCH, device=device))
        oG.zero_grad(set_to_none=True); lG.backward(); oG.step()
        if it % CHECK_EVERY == 0:
            kids = []
            for lab in minority_labels:
                fake_imgs = generate(G, 500, lab, device, n_classes)
                kids.append(metrics.kid(real_feats[lab], features(fake_imgs, device), subset_size=50)[0])
            mean_kid = float(np.mean(kids))
            log(f"  [{name}] iter {it:5d}  D {lD.item():.3f}  G {lG.item():.3f}  mean KID {mean_kid:.4f}  ({time.time() - t0:.0f}s)")
            if mean_kid < best[0]:
                best = (mean_kid, {k: v.detach().clone() for k, v in G.state_dict().items()}, it)
    G.load_state_dict(best[1])
    return G, {"best_iter": best[2], "best_mean_kid": best[0], "train_seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- one scenario at one level
def main():
    global ITERS, CHECK_EVERY
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["s1_cond_minority", "s2_cond_all", "s3_per_class"])
    ap.add_argument("--level", default="full", choices=metrics.LEVELS)
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args()
    ITERS = args.iters
    CHECK_EVERY = min(CHECK_EVERY, ITERS)
    device = "cuda"
    GEN_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)
    name = f"{args.scenario}_{args.level}"
    log_path = RES_DIR / f"{name}.log"
    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    splits, classes = data.load()
    x, y = apply_level(*splits["train"], classes, args.level)
    minority = [classes.index(c) for c in metrics.MINORITY_CLASSES]
    log(f"=== {name}: {len(y)} training images, minority sizes {[int((y == m).sum()) for m in minority]} ===")
    real_feats = {}
    t0 = time.time()
    out_x, out_y, report = [], [], {"scenario": args.scenario, "level": args.level, "iters": ITERS,
                                    "diffaugment": True, "per_class": {}}

    if args.scenario == "s3_per_class":
        for m in minority:
            xc = x[y == m]
            real_feats = {0: features(xc, device)}
            G, info = train_gan(xc, np.zeros(len(xc), np.int64), 0, [0], real_feats, device, log,
                                f"{name}/{classes[m]}")
            n_add = max(0, metrics.TARGET_PER_CLASS - len(xc))
            fake = generate(G, n_add, 0, device, 0)
            out_x.append(fake); out_y.append(np.full(n_add, m))
            report["per_class"][classes[m]] = {**info, "n_real": int(len(xc)), "n_generated": n_add}
    else:
        if args.scenario == "s1_cond_minority":
            keep = np.isin(y, minority)
            xs, ys = x[keep], y[keep]
            to_gan_label = {m: i for i, m in enumerate(minority)}
        else:  # s2_cond_all
            xs, ys = x, y
            to_gan_label = {c: c for c in range(len(classes))}
        yg = np.array([to_gan_label[v] for v in ys], np.int64)
        gan_minority = [to_gan_label[m] for m in minority]
        real_feats = {to_gan_label[m]: features(x[y == m], device) for m in minority}
        G, info = train_gan(xs, yg, len(to_gan_label), gan_minority, real_feats, device, log, name)
        report.update(info)
        for m in minority:
            n_real = int((y == m).sum())
            n_add = max(0, metrics.TARGET_PER_CLASS - n_real)
            fake = generate(G, n_add, to_gan_label[m], device, len(to_gan_label))
            out_x.append(fake); out_y.append(np.full(n_add, m))
            report["per_class"][classes[m]] = {"n_real": n_real, "n_generated": n_add}

    # final quality checks on the images that will actually be used
    rows = []
    for m, fx in zip(minority, out_x):
        c = classes[m]
        real = x[y == m]
        rf = features(real, device)
        report["per_class"][c]["kid"] = metrics.kid(rf, features(fx, device), subset_size=min(100, len(real), len(fx)))
        report["per_class"][c]["memorisation"] = memorisation(real, fx)
        rows.append((c, real, fx))
    report["mean_kid"] = float(np.mean([v["kid"][0] for v in report["per_class"].values()]))
    report["total_seconds"] = round(time.time() - t0, 1)
    np.savez(GEN_DIR / f"{name}.npz", x=np.concatenate(out_x), y=np.concatenate(out_y).astype(np.int64))
    save_grid(RES_DIR / f"{name}.png", rows)
    (RES_DIR / f"{name}.json").write_text(json.dumps(report, indent=1))
    log(f"=== {name} done: mean KID {report['mean_kid']:.4f}, {len(np.concatenate(out_y))} images, "
        f"{report['total_seconds'] / 60:.0f} min ===")


if __name__ == "__main__":
    main()
