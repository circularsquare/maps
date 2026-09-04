"""Run every country's scatter at once, one process each.

    python scatter_all.py                        # all 14 countries, both editions
    python scatter_all.py --countries us,in      # a subset
    python scatter_all.py --dot-values 1000      # the fine edition only
    python scatter_all.py --jobs 14

A scatter is a whole process reading its own inputs and writing its own two files, so
there is nothing to coordinate and no shared state to get wrong: this just runs
scatter.py the same way COMMANDS.txt does, several at a time.  It exists because the set
is 28 runs (fourteen countries at two dot values, §4.1b) and they were serial.

LONGEST FIRST.  The runs are wildly uneven — India's fine edition is minutes and Estonia's
coarse one is a second — so they are started in descending order of their last known cost.
Scheduling the long ones last leaves the machine finishing one country on one core with
fifteen idle, which is most of the wall clock back again.

EVERY CHILD'S OUTPUT IS PRINTED IN FULL, as one block when it finishes rather than
interleaved.  Those lines are the reconciliation — the carry, the people under one dot,
the ring count, and for the US which rows got demographic weights — and they are the
reason to watch a scatter run at all.  A failure prints the child's stderr and stops the
run, because a missing dots_<cc>.geojson makes the next tile build silently short a
country (tiles.py's --countries note).
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from countries import COUNTRIES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent

# Rough seconds per run, only ever used to decide what to start first, so it does not need
# to be right — it needs to put the United Kingdom before Estonia. Measured 2026-09-04
# after the shapely.prepare fix; anything not listed sorts as average.
#
# THE COARSE EDITION IS NOT A TENTH OF THE COST and must not be scheduled as though it
# were. It has a tenth of the dots and the same everything else — the same placement layer
# read, the same counts, the same joins — so it runs at 0.2x the fine edition for India and
# 0.95x for New Zealand, whose entire cost is parsing one 140 MB GeoJSON. Both editions of
# a country therefore sort together, near the front if the country is slow.
COST = {"uk": 68, "us": 48, "nz": 39, "in": 36, "ca": 32, "mx": 24, "au": 13, "br": 11,
        "ie": 8, "de": 6, "cz": 4, "pl": 3, "ro": 2, "ee": 1, "hr": 1}
COARSE_FACTOR = 0.8


def run(job):
    cc, dv = job
    cmd = [sys.executable, "scatter.py", "--country", cc]
    if dv is not None:
        cmd += ["--dot-value", str(dv)]
    env = dict(os.environ)
    # One process per core already; letting each one's BLAS open sixteen threads on top
    # oversubscribes the machine and makes every run slower than it was serially.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return job, p.returncode, p.stdout, p.stderr, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default=",".join(sorted(COUNTRIES)),
                    help="comma-separated; defaults to every country in countries.py")
    ap.add_argument("--dot-values", default="1000,10000",
                    help="comma-separated; 1000 is the default edition and 10000 is the "
                         "one tiles.py --coarse needs (§4.1b)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="how many at once. Defaults to half the cores, which leaves the "
                         "machine usable; raise it if nothing else is running.")
    args = ap.parse_args()

    ccs = [c.strip() for c in args.countries.split(",") if c.strip()]
    unknown = [c for c in ccs if c not in COUNTRIES]
    if unknown:
        raise SystemExit(f"not in countries.py: {unknown}")
    dvs = [int(v) for v in args.dot_values.split(",")]

    jobs = [(cc, None if dv == 1000 else dv) for cc in ccs for dv in dvs]
    jobs.sort(key=lambda j: -COST.get(j[0], 20) * (1 if j[1] is None else COARSE_FACTOR))

    print(f"{len(jobs)} runs, {args.jobs} at a time")
    t0 = time.perf_counter()
    failed, done = [], 0
    # Threads, not processes: every one of these is just waiting on a subprocess.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for fut in as_completed(futures):
            (cc, dv), rc, out, err, dt = fut.result()
            done += 1
            label = f"{cc}" + (f" 1:{dv:,}" if dv else " 1:1,000")
            print(f"\n{'=' * 78}\n[{done}/{len(jobs)}] {label}   {dt:.1f}s"
                  + ("" if rc == 0 else "   FAILED") + f"\n{'=' * 78}")
            print(out.rstrip())
            if rc != 0:
                print(err.rstrip(), file=sys.stderr)
                failed.append(label)

    print(f"\n{len(jobs)} runs in {time.perf_counter() - t0:.1f}s")
    if failed:
        raise SystemExit(f"FAILED: {', '.join(failed)} — do not tile until these are fixed")


if __name__ == "__main__":
    main()
