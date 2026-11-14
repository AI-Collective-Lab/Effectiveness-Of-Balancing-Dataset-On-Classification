"""Run every classifier job of the design, one after another, skipping finished ones.

    python run_queue.py --methods baseline class_weighted oversampling
    python run_queue.py --methods gan_s1_cond_minority gan_s2_cond_all gan_s3_per_class --widths 1.0

Safe to stop and restart: a job whose results/<tag>.json exists is skipped. Progress goes to
results/queue.log (unbuffered), which the progress window follows.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import metrics

HERE = Path(__file__).resolve().parent
LOG = HERE / "results" / "queue.log"


def jobs(methods, widths, levels, seeds):
    # main model first, level by level, so the most important results arrive earliest
    for width in widths:
        for level in levels:
            for method in methods:
                for seed in seeds:
                    yield method, level, width, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--widths", nargs="+", type=float, default=list(metrics.WIDTHS))
    ap.add_argument("--arch", default="cnn", choices=("cnn", "resnet18"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(metrics.SEEDS))
    ap.add_argument("--levels", nargs="+", default=list(metrics.LEVELS))
    args = ap.parse_args()
    pre = "r18_" if args.arch == "resnet18" else ""
    todo = [j for j in jobs(args.methods, args.widths, args.levels, args.seeds)
            if not (HERE / "results" / f"{pre}{j[0]}_{j[1]}_w{j[2]:g}_seed{j[3]}.json").exists()]
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n=== queue started {time.strftime('%Y-%m-%d %H:%M')}: {len(todo)} runs to do ===\n")
        log.flush()
        t0 = time.time()
        for k, (method, level, width, seed) in enumerate(todo, 1):
            done = time.time() - t0
            eta = f", about {done / (k - 1) * (len(todo) - k + 1) / 3600:.1f} h left" if k > 1 else ""
            log.write(f"--- run {k}/{len(todo)}: {method} | level {level} | width {width:g} | seed {seed}{eta}\n")
            log.flush()
            subprocess.run([sys.executable, "-u", "train.py", "--method", method, "--level", level,
                            "--width", str(width), "--seed", str(seed), "--arch", args.arch],
                           cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"=== QUEUE DONE in {(time.time() - t0) / 3600:.1f} h ===\n")


if __name__ == "__main__":
    main()
