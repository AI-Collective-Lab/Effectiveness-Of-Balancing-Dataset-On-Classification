"""Train all 27 GANs one after another (S1 and S2 first, then the 21 per-class GANs of S3),
skipping any whose gan_results/<scenario>_<level>.json exists. Log: gan_results/gan_queue.log.

    python run_gans.py
"""
import subprocess
import sys
import time
from pathlib import Path

import metrics

HERE = Path(__file__).resolve().parent
RES = HERE / "gan_results"
LOG = RES / "gan_queue.log"


def main():
    RES.mkdir(exist_ok=True)
    jobs = [(s, lv) for s in ("s1_cond_minority", "s2_cond_all", "s3_per_class") for lv in metrics.LEVELS]
    todo = [j for j in jobs if not (RES / f"{j[0]}_{j[1]}.json").exists()]
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n=== GAN queue started {time.strftime('%Y-%m-%d %H:%M')}: {len(todo)} jobs "
                  f"(S3 jobs contain 7 GANs each) ===\n"); log.flush()
        t0 = time.time()
        for k, (scenario, level) in enumerate(todo, 1):
            log.write(f"--- job {k}/{len(todo)}: {scenario} | level {level} ({(time.time() - t0) / 60:.0f} min so far)\n")
            log.flush()
            subprocess.run([sys.executable, "-u", "gan.py", "--scenario", scenario, "--level", level],
                           cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"=== GAN QUEUE DONE in {(time.time() - t0) / 3600:.1f} h ===\n")


if __name__ == "__main__":
    main()
