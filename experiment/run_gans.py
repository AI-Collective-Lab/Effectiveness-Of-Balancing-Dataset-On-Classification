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
STALL_MIN = 15   # a GAN logs a line every 3-4 min; this long without one means it hung


def run_with_watchdog(scenario, level, log):
    """Run one job; kill it if its own log stops growing for STALL_MIN minutes. True if it finished."""
    job_log = RES / f"{scenario}_{level}.log"
    p = subprocess.Popen([sys.executable, "-u", "gan.py", "--scenario", scenario, "--level", level],
                         cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    last_size, last_change = -1, time.time()
    while p.poll() is None:
        time.sleep(20)
        size = job_log.stat().st_size if job_log.exists() else 0
        if size != last_size:
            last_size, last_change = size, time.time()
        elif time.time() - last_change > STALL_MIN * 60:
            p.kill(); p.wait()
            return False
    return True


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
            for attempt in range(1, 4):
                if run_with_watchdog(scenario, level, log):
                    break
                log.write(f"!!! {scenario} {level}: no progress for {STALL_MIN} min - killed, retry {attempt}/2\n")
                log.flush()
        log.write(f"=== GAN QUEUE DONE in {(time.time() - t0) / 3600:.1f} h ===\n")


if __name__ == "__main__":
    main()
