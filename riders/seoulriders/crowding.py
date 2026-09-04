# -*- coding: utf-8 -*-
"""Make crowded trains less attractive, by iterating the build against itself.

**The problem.** `build.py` routes every rider onto the fastest journey, wait
included. On a line with both 급행 and 일반 that means everyone who *can* take
the express does, because it always is faster. Real passengers do not: the
express is already full, and plenty of people would rather sit on a local for
twenty minutes than stand on an express for fourteen. Measured against
서울시메트로9호선's own published figures, the build put **61.7%** of 9호선's
riders on the 급행 against a real **40.2%** -- the express came out right to
within 2% and the local carried less than half what it should. See "RAPTOR
over-fills the 급행" in README.md.

**Why it cannot just be a capacity check.** Whether you can board the 08:05
급행 at 노량진 depends on everyone who got on upstream at 김포공항 and 여의도.
That is a shared, global constraint, and the routing is deliberately the
opposite of that -- 624 origins routed independently across the cores, no
worker knowing what any other worker loaded onto a train. Adding a live "is
this train full" test would serialise the whole build.

**So iterate instead.** This is the standard transit-assignment answer, method
of successive averages:

    round 0   build with no penalty          -> loads
    round i   loads -> penalty per segment
              build again, riders now avoid the crush
              average the new loads into the old ones with weight 1/(i+1)

Averaging is what makes it converge rather than oscillate: without it round 1
empties the express, round 2 refills it, and it rings forever. Each round is
still a fully parallel build, because the penalty is a read-only table computed
*between* rounds and dropped in `data/crowding.npz`.

**The penalty.** Crowding `c` is riders over 정원. From the published figures
for lines 1-8, 정원 is routinely exceeded -- 1.57% of all station-halfhours sit
above 100% -- but *nothing on the network exceeds 145%*, which is the real
ceiling. So the penalty is zero up to `C_FREE`, and rises as a square to
`PENALTY_MAX` times the segment's own run time at `C_CRUSH`:

    factor(c) = 1 + ALPHA * clamp((c - C_FREE) / (C_CRUSH - C_FREE), 0, 1) ** 2

A minute on a train at 145% is then worth `1 + ALPHA` minutes of perceived
time. ALPHA is the one knob, and it is calibrated against the only measurement
that can settle it: 9호선's published express share.

**What can and cannot be checked.** 9호선 is the *only* line with a published
express/local split, so ALPHA is fitted on one line and applied to all of them.
1호선's 급행 carries 28.8% of that line's riders in our build with nothing to
check it against. That is a real limitation and it is why this file prints the
fitted value rather than burying it.

    python crowding.py                  # 4 rounds at the default ALPHA
    python crowding.py --rounds 6
    python crowding.py --calibrate      # sweep ALPHA, report the express share
    python crowding.py --reset          # delete the penalties, back to round 0
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

import lines as LR

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

CROWDING = os.path.join(D, "crowding.npz")
LOADS = os.path.join(D, "loads.npz")
AVG = os.path.join(D, "crowding_loads_avg.npz")

# 정원 is a design figure, not a wall -- see the module docstring. Both numbers
# come from the published 혼잡도 for lines 1-8: the p99 of every
# station-direction-halfhour is 108%, and the single busiest cell on the whole
# network is 144.6% (2호선 사당 외선, 08:30). Nothing goes above 150%.
C_FREE = 1.00
C_CRUSH = 1.45

# How much worse a minute at C_CRUSH feels than a minute in an empty train.
# Fitted, not guessed -- see --calibrate.
ALPHA = 3.0

ROUNDS = 4

# 9호선's published express share of riders, from congestion_line9.xlsx via
# validate.py. The single number this whole file is fitted against.
TARGET_EXPRESS_SHARE = 0.402


def run_build(loads_out, final=False, jobs=None, sample=1):
    cmd = [sys.executable, os.path.join(HERE, "build.py")]
    if not final:
        cmd += ["--no-output"]
    if loads_out:
        cmd += ["--loads-out", loads_out]
    if jobs:
        cmd += ["--jobs", str(jobs)]
    if sample > 1:
        cmd += ["--sample", str(sample)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit("build.py failed (%d)" % r.returncode)
    return time.time() - t0


def load_arrays(path):
    with np.load(path, allow_pickle=True) as z:
        return [z["p%d" % i].astype(np.float64) for i in range(int(z["npat"]))]


def pattern_meta():
    """Per-pattern capacity and run times, straight off the timetable.

    Imports build.py rather than re-deriving them, so the two cannot drift.
    It is the same load_patterns() the build uses.
    """
    import build
    with io.open(os.path.join(D, "stations.json"), encoding="utf-8") as f:
        net = json.load(f)
    pats = build.load_patterns(net, quiet=True)
    caps, runs, lines, express = [], [], [], []
    for p in pats:
        cap = LR.CAPACITY.get(p["line"]) or 1000
        caps.append(float(cap))
        lines.append(p["line"])
        express.append(bool(p["express"]))
        # Seconds from departing stop si to arriving at si+1, per trip.
        # load_patterns() hands back the raw (trips x stops) arrays; the list
        # forms RAPTOR walks are built later, in prepare_scan().
        dep = np.asarray(p["dep"], dtype=np.float64)
        arr = np.asarray(p["arr"], dtype=np.float64)
        rt = np.zeros_like(dep)
        if dep.shape[1] > 1:
            rt[:, :-1] = np.clip(arr[:, 1:] - dep[:, :-1], 0.0, 1800.0)
        runs.append(rt)
    return caps, runs, lines, express


def penalties(loads, caps, runs, alpha):
    """Cumulative perceived penalty in seconds, per pattern/trip/stop.

    Returns (arrays, stats). Read the stats rather than just the headline: the
    *worst* whole-trip penalty is a tail, and a tail of two hours would wreck
    the search, because the labels RAPTOR compares against real departure times
    are perceived ones and riders would start missing connections they would
    really make. What matters is the typical penalty on a ride that is crowded
    at all, which is the median and p90.
    """
    out = []
    worst = 0.0
    crowded = []
    for m, cap, rt in zip(loads, caps, runs):
        c = m / cap
        x = np.clip((c - C_FREE) / (C_CRUSH - C_FREE), 0.0, 1.0)
        # per-segment penalty: the run time, times how much worse it feels
        seg = rt * (alpha * x * x)
        cum = np.cumsum(seg, axis=1).astype(np.float32)
        out.append(cum)
        if cum.size:
            worst = max(worst, float(cum[:, -1].max()))
            v = seg[seg > 0]
            if v.size:
                crowded.append(v)
    if crowded:
        v = np.concatenate(crowded)
        stats = (worst, float(np.median(v)), float(np.percentile(v, 90)),
                 100.0 * v.size / max(1, sum(a.size for a in out)))
    else:
        stats = (0.0, 0.0, 0.0, 0.0)
    return out, stats


def write_penalties(pen, iteration):
    np.savez_compressed(
        CROWDING, npat=np.array(len(pen)), iteration=np.array(iteration),
        **dict(("p%d" % i, a) for i, a in enumerate(pen)))


def express_share(loads, lines, express, only="9"):
    """Share of a line's carried riders that are on an express, from loads."""
    tot = x = 0.0
    for m, line, xp in zip(loads, lines, express):
        if line != only:
            continue
        s = float(m.sum())
        tot += s
        if xp:
            x += s
    return (x / tot) if tot else float("nan")


def report(loads, lines, express):
    s9 = express_share(loads, lines, express, "9")
    s1 = express_share(loads, lines, express, "1")
    print("   9호선 express share %.1f%%  (published %.1f%%, off by %+.1f pts)"
          % (100 * s9, 100 * TARGET_EXPRESS_SHARE,
             100 * (s9 - TARGET_EXPRESS_SHARE)))
    print("   1호선 express share %.1f%%  (nothing published to check it)"
          % (100 * s1))
    return s9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--jobs", type=int, default=0,
                    help="passed through to build.py (default: its own half"
                         "-the-cores)")
    ap.add_argument("--calibrate", action="store_true",
                    help="after the rounds, sweep alpha over the final loads "
                         "and print the express share each one would give")
    ap.add_argument("--sample", type=int, default=1,
                    help="pass --sample to build.py and scale the loads back "
                         "up by the same factor, so a cheap run still "
                         "produces realistic crowding. For checking the loop "
                         "works, not for a real answer")
    ap.add_argument("--reset", action="store_true",
                    help="delete the penalties and stop")
    args = ap.parse_args()

    if args.reset:
        for p in (CROWDING, AVG, LOADS):
            if os.path.exists(p):
                os.remove(p)
                print("removed %s" % os.path.basename(p))
        print("\nNext build will be the uncrowded one. Re-run build.py.")
        return

    caps, runs, lines, express = pattern_meta()
    print("%d patterns, %d with express service\n" % (len(caps), sum(express)))

    # round 0 -- no penalty, which is whatever crowding.npz currently says.
    if os.path.exists(CROWDING):
        os.remove(CROWDING)
        print("cleared the old penalties; starting from the uncrowded build\n")

    avg = None
    for it in range(args.rounds + 1):
        final = it == args.rounds
        print("=" * 70)
        print("round %d of %d%s" % (it, args.rounds,
                                    "  (final -- writes trains.json)" if final
                                    else ""))
        print("=" * 70)
        secs = run_build(LOADS, final=final, jobs=args.jobs,
                         sample=args.sample)
        loads = load_arrays(LOADS)
        if args.sample > 1:
            # A sampled build carries 1/N of the riders, so nothing would ever
            # look crowded and every penalty would be zero. Scaling back up
            # makes the loop exercise the same code paths it will in anger.
            loads = [m * args.sample for m in loads]
        print("   %.0fs" % secs)
        share = report(loads, lines, express)

        # Method of successive averages. Without it round 1 empties the
        # express, round 2 refills it, and the loop rings forever.
        if avg is None:
            avg = loads
        else:
            w = 1.0 / (it + 1)
            avg = [a * (1 - w) + b * w for a, b in zip(avg, loads)]

        if final:
            break
        pen, (worst, p50, p90, share) = penalties(avg, caps, runs, args.alpha)
        write_penalties(pen, it + 1)
        print("   penalties for round %d: %.2f%% of segments penalised at all,"
              " median +%.0fs, p90 +%.0fs, worst whole trip +%.1f min"
              % (it + 1, share, p50, p90, worst / 60.0))

    np.savez_compressed(AVG, npat=np.array(len(avg)),
                        **dict(("p%d" % i, a.astype(np.float32))
                               for i, a in enumerate(avg)))

    if args.calibrate:
        print("\n" + "=" * 70)
        print("alpha sweep over the final loads -- what each would penalise by")
        print("=" * 70)
        for a in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
            _, (worst, p50, p90, share) = penalties(avg, caps, runs, a)
            print("   alpha %5.1f   median +%3.0fs  p90 +%4.0fs  worst trip "
                  "+%5.1f min" % (a, p50, p90, worst / 60.0))
        print("\n   A sweep cannot tell you the resulting express share -- that")
        print("   needs a build per value. Run --rounds 1 --alpha X to test one.")

    if args.sample > 1:
        print("\ndone -- but --sample %d, so the outputs are the *.sample.json"
              % args.sample)
        print("pair, and every load was scaled back up by %d to fake the"
              % args.sample)
        print("crowding. Plumbing only. data/crowding.npz now holds those")
        print("faked penalties: run --reset before any real build.")
        return
    print("\ndone. trains.json and stats.json are the crowded build.")
    print("Check it with: python validate.py --congestion")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
