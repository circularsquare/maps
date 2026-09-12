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

**The penalty.** Crowding `c` is riders over 정원. The penalty is zero while
there is a seat to be had, and then rises without limit:

    x = max(c - C_SEATS, 0) / (C_CRUSH - C_SEATS)
    factor(c) = 1 + ALPHA * x * (1 + x) / 2

A minute on a train at C_CRUSH is worth `1 + ALPHA` minutes of perceived time.
ALPHA is the one knob, calibrated against the only measurement that can settle
it: 9호선's published express share.

Two things about that shape are load-bearing, and the first version of this
file got both wrong:

**It starts at the seats, not at 정원.** 정원 is a crush figure -- 160 to a
car against about 54 seats -- so a train at a third of it is already full of
people standing. The published 9호선 figures show the 급행 running 1.4x-2.25x
fuller than the 일반 at *every hour*, midday included, where it sits at 39% of
정원 against the 일반's 21%. Riders are trading crowding against time all day
long, at loads that a threshold at 정원 cannot see. With the threshold at
1.00 the penalty touched 1.9% of the network and the express was free
everywhere it was not already at crush.

**It has no ceiling.** The old form clipped at C_CRUSH, so a train at 265% of
정원 was penalised exactly as much as one at 145% -- no marginal deterrent at
all, and no equilibrium for the loop to find. Line 1's 급행 duly ran away with
4,000 riders on a single train at 255% of 정원 while the 일반 behind it sat at
a fifth of 정원.

**The crush cap does not interfere with any of this, and that is deliberate.**
`build.py` caps single trains at 1.5x 정원 on the way out to `trains.json`, but
it does so *after* `dump_loads()`, so every load this file reads is the true
uncapped demand. Feeding the loop capped loads would hide the crowding from the
one mechanism meant to deter it. Only the final round writes a capped
`trains.json` at all; the intermediate rounds run `--no-output` and never reach
the capping pass. See "The crush cap" in README.md.

**What can and cannot be checked.** 9호선 is the *only* line with a published
express/local split, so ALPHA is fitted on one line and applied to all of them.
1호선's 급행 carries 28.8% of that line's riders in our build with nothing to
check it against. That is a real limitation and it is why this file prints the
fitted value rather than burying it.

    python crowding.py                  # 4 rounds at the default ALPHA
    python crowding.py --rounds 6
    python crowding.py --calibrate      # sweep ALPHA, report the express share
    python crowding.py --reset          # delete the penalties, back to round 0

**Fitting ALPHA cheaply.** A full pass is five builds and the better part of
an hour, which is a miserable inner loop for one number. Use `--sample`: it
routes one origin in N and scales the loads back up by N, so every load factor
and every penalty is realistic even though a sixth of the riders are being
moved. The express share is a *ratio* over the same network, so it survives
sampling far better than any of the levels do.

    python crowding.py --sample 6 --rounds 3 --alpha 3
    python crowding.py --sample 6 --rounds 3 --alpha 6      # etc

Read the 9호선 lines each round: the share should be walking towards 40.2%
and the 급행/일반 load ratio towards about 1.75x. Then confirm the winner on a
full run, and `--reset` before any ordinary build -- a sampled pass leaves
faked penalties in data/crowding.npz.
"""

import argparse
import io
import json
import os
import sys
import time

import numpy as np

import lines as LR

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

CROWDING = os.path.join(D, "crowding.npz")
LOADS = os.path.join(D, "loads.npz")
AVG = os.path.join(D, "crowding_loads_avg.npz")

# Where a rider starts to mind. NOT 정원: 정원 counts every standing place at
# crush, about 160 to a car against roughly 54 seats, so the seats are gone at
# a third of it and everything above that is somebody standing.
#
# It used to be 1.00, and that was the single thing most wrong with this file.
# The published 9호선 split says the 급행 runs 1.4x to 2.25x fuller than the
# 일반 at *every hour of the day* -- at midday it sits at 39% of 정원 against
# the 일반's 21% -- so riders are trading crowding off against time all day, at
# loads a threshold-at-정원 model cannot see at all. With C_SEATS = 1.00 the
# penalty was identically zero for all but 1.9% of the network, the express
# was free everywhere it was not already at crush, and the build put 55% of
# 9호선's riders on the 급행 against a published 40.2% -- with the gap widest
# (60% against 42%) in the quiet middle of the day, where the old penalty did
# nothing whatsoever. See "What the published split actually says" in README.
C_SEATS = 0.34

# The busiest cell on the whole published network: 2호선 사당 외선 at 08:30,
# 144.6%. It is not a ceiling any more, only the point ALPHA is quoted at.
C_CRUSH = 1.45

# How much worse a minute at C_CRUSH feels than a minute in a seat.
# Fitted, not guessed -- see --calibrate.
#
# Was 3.0. Lowered to 2.0 on 2026-09-06 after an A/B at --sample 6 --rounds 3,
# written up as "ALPHA 3 against ALPHA 2, measured" in README.md. The reason is
# *not* the express share -- that got nominally worse (39.0% -> 43.4% against a
# published 40.2%), and the sampled loop cannot resolve a difference that size
# anyway, since sampled and full runs at the same ALPHA differ by 4.2 points.
# The reasons are the two things it *can* resolve:
#
#   - the 급행/일반 load ratio moved *towards* published, 1.24x -> 1.42x
#     against 1.52x, and that is the number that says whether riders split for
#     the right reason rather than merely in the right proportion;
#   - the platform crowds fell by a quarter (사당 4,571 -> 3,295, p90 wait
#     9.9 -> 7.8 min), which settled the open question of whether ALPHA drives
#     them at all or whether they were the structural peak-spreading limit.
#
# ALPHA is buying the split partly with a smear the model cannot represent --
# riders avoiding a crush can only ever take a slower path, never leave
# earlier, because the OD fixes how many depart in each hour. Less ALPHA is
# less of that. See "Two things it cost" in README.md.
ALPHA = 2.0

ROUNDS = 4

# 9호선's published express share of riders, from congestion_line9.xlsx via
# validate.py. The single number this whole file is fitted against.
TARGET_EXPRESS_SHARE = 0.402


def run_build(loads_out, final=False, jobs=None, sample=1):
    """One round, in this process.

    Deliberately not a subprocess. Shelling out to `python build.py` put the
    multiprocessing pool three levels deep -- shell, crowding.py, build.py,
    workers -- and 17 minutes into a round the pool's attempt to replace a
    worker died with `PermissionError: [WinError 5]` from `DuplicateHandle` in
    spawn_main, taking the whole run with it. Nothing to do with memory; the
    machine had 40 GB free. Calling build.main() directly puts the pool at the
    same depth as a plain `python build.py`, which is the shape that works.
    """
    import build

    argv = []
    if not final:
        argv += ["--no-output"]
    if loads_out:
        argv += ["--loads-out", loads_out]
    if jobs:
        argv += ["--jobs", str(jobs)]
    if sample > 1:
        argv += ["--sample", str(sample)]
    t0 = time.time()
    build.main(argv)
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
        # Standing load, as a fraction of the way from the last seat to crush.
        # Clipped at the bottom only: there is no ceiling any more. The old
        # clip at 1.0 meant a train at 265% of 정원 was no less attractive
        # than one at 145%, so nothing stopped a 급행 from running away with
        # 4,000 riders aboard while the 일반 behind it ran at a fifth of 정원.
        x = np.clip((c - C_SEATS) / (C_CRUSH - C_SEATS), 0.0, None)
        # Linear where people are merely standing, quadratic once it is a
        # crush, and unbounded above -- which is what makes the loop settle
        # instead of overshooting. Normalised so ALPHA still means "how much
        # worse a minute at C_CRUSH is", as it did before.
        seg = rt * (alpha * 0.5 * x * (1.0 + x))
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
    """Share of a line's carried riders that are on an express, from loads.

    Person-pattern-stops, which is the same quantity write_stats() sums per
    segment -- so this is directly comparable with the 61.7% / 40.2% pair in
    the README, and with `validate.py --congestion`.
    """
    tot = x = 0.0
    for m, line, xp in zip(loads, lines, express):
        if line != only:
            continue
        s = float(m.sum())
        tot += s
        if xp:
            x += s
    return (x / tot) if tot else float("nan")


def load_ratio(loads, caps, lines, express, only="9"):
    """How much fuller the express runs than the local, rider-weighted.

    The share says how the riders split; this says whether they split for the
    right *reason*. A crowding model that works settles where the express is
    persistently fuller than the local by just enough to offset the time it
    saves. Measured this same way on the published 9호선 figures, that ratio
    is **1.52x** over the day -- 1.44x at the peaks, 1.96x off them, so it
    compresses as the whole line fills, which is what a convex crowding cost
    does. A model with no crowding cost below 정원 cannot produce any of that:
    it fills the express until something else stops it, and came out at 2.21x.
    """
    tot = {True: 0.0, False: 0.0}
    wt = {True: 0.0, False: 0.0}
    for m, cap, line, xp in zip(loads, caps, lines, express):
        if line != only or m.size == 0:
            continue
        v = m[m > 0]
        if not v.size:
            continue
        # rider-weighted mean load factor: an empty terminal segment is one
        # number in the file but it is not one question
        tot[bool(xp)] += float((v * v).sum()) / cap
        wt[bool(xp)] += float(v.sum())
    lo = tot[False] / wt[False] if wt[False] else 0.0
    hi = tot[True] / wt[True] if wt[True] else 0.0
    return hi, lo, (hi / lo if lo else float("nan"))


# Published 급행/일반 load ratio, rider-weighted over the cells validate.py
# matches -- the same measure load_ratio() applies to ours. 1.44x at the peaks
# and 1.96x off them.
TARGET_LOAD_RATIO = 1.52


def report(loads, caps, lines, express):
    s9 = express_share(loads, lines, express, "9")
    s1 = express_share(loads, lines, express, "1")
    print("   9호선 express share %.1f%%  (published %.1f%%, off by %+.1f pts)"
          % (100 * s9, 100 * TARGET_EXPRESS_SHARE,
             100 * (s9 - TARGET_EXPRESS_SHARE)))
    hi, lo, ratio = load_ratio(loads, caps, lines, express, "9")
    print("   9호선 loads: 급행 %.0f%% of 정원, 일반 %.0f%%, ratio %.2fx  "
          "(published about %.2fx)" % (100 * hi, 100 * lo, ratio,
                                       TARGET_LOAD_RATIO))
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
        share = report(loads, caps, lines, express)

        # Method of successive averages. Without it round 1 empties the
        # express, round 2 refills it, and the loop rings forever.
        if avg is None:
            avg = loads
        else:
            w = 1.0 / (it + 1)
            avg = [a * (1 - w) + b * w for a, b in zip(avg, loads)]

        # Written every round, not just at the end: a round is a quarter of an
        # hour and losing four of them to a crash on the fifth is a bad trade
        # for one np.savez_compressed.
        np.savez_compressed(AVG, npat=np.array(len(avg)),
                            **dict(("p%d" % i, a.astype(np.float32))
                                   for i, a in enumerate(avg)))

        if final:
            break
        pen, (worst, p50, p90, share) = penalties(avg, caps, runs, args.alpha)
        write_penalties(pen, it + 1)
        print("   penalties for round %d: %.2f%% of segments penalised at all,"
              " median +%.0fs, p90 +%.0fs, worst whole trip +%.1f min"
              % (it + 1, share, p50, p90, worst / 60.0))

    if args.calibrate:
        print("\n" + "=" * 70)
        print("alpha sweep over the final loads -- what each would penalise by")
        print("=" * 70)
        for a in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
            _, (worst, p50, p90, share) = penalties(avg, caps, runs, a)
            print("   alpha %5.1f   median +%3.0fs  p90 +%4.0fs  worst trip "
                  "+%5.1f min" % (a, p50, p90, worst / 60.0))
        print("\n   A sweep cannot tell you the resulting express share -- that")
        print("   needs a build per value. The cheap way to get one is")
        print("   --sample 6 --rounds 3 --alpha X; see the module docstring.")

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
