"""Rescatter every drawn country, a few at a time (2026-10-08: leftover marks in scatter.py and
the cross-border regroup both change every country's dots).

    python tools/rescatter_all.py              # 4 at a time
    python tools/rescatter_all.py -j 3 --only us,uk,ca
    python tools/rescatter_all.py --skip zw,ga,ba

Each country's output goes to data/processed/scatter_logs/<cc>.log; a line per country as it
finishes, and the failures at the end. Then run the build tail (COMMANDS.txt).
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
LOGS = HERE / "data" / "processed" / "scatter_logs"

# biggest first, so the long ones are not left running alone at the end
FIRST = ["us", "in", "cn", "uk", "ca", "br", "au", "id", "ie", "ru", "mx", "za", "de", "fr"]


def run(cc):
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    with open(LOGS / f"{cc}.log", "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, str(HERE / "scatter.py"), "--country", cc],
                           cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
    return cc, r.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--only")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    from countries import COUNTRIES
    ccs = a.only.split(",") if a.only else sorted(COUNTRIES)
    skip = set(filter(None, a.skip.split(",")))
    ccs = [c for c in ccs if c not in skip]
    ccs = [c for c in FIRST if c in ccs] + [c for c in ccs if c not in FIRST]
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"rescattering {len(ccs)} countries, {a.j} at a time; logs in {LOGS}", flush=True)
    t0, failed = time.time(), []
    with ThreadPoolExecutor(a.j) as ex:
        futs = [ex.submit(run, c) for c in ccs]
        for i, f in enumerate(as_completed(futs), 1):
            cc, rc, secs = f.result()
            if rc:
                failed.append(cc)
            print(f"  [{i}/{len(ccs)}] {cc} {'FAILED' if rc else 'ok'} {secs:.0f}s "
                  f"({(time.time() - t0) / 60:.0f} min in)", flush=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min; "
          + (f"FAILED: {' '.join(failed)} (see their logs)" if failed else "no failures"))


if __name__ == "__main__":
    main()
