# -*- coding: utf-8 -*-
"""Route riders over the real timetable and write data/trains.json.

Reads the hourly OD from build_od.py, finds each pair a journey with RAPTOR
over the timetable for that day type, boards the riders onto specific trains,
and writes one timeline per train run.

**There is no --day flag here.** The day is whatever `build_od.py` last built,
read back out of `data/od_hourly.npz` -- see `_day()` and `daytype.py`. Two
flags that had to be kept in step would eventually drift, and a weekday routed
over a Sunday timetable produces no error, just a thinner weekday.

Riders spawn every 2.5 minutes inside their hour rather than all at once, and
RAPTOR runs afresh for each of those bins so that riders board the train that
is actually next, not the one that was next at the top of the hour. That is
what stops a whole hour of 잠실 piling onto a single midnight train. Each
origin's bins are offset by dep_phase() so that the whole network does not
release its crowds on the same tick, and most of those searches never actually
run -- see spawn_key(). See DEP_BIN.

Finishes by calling build_shapes.main(), which bends the straight
station-to-station hops onto the real track. Forgetting that step is what makes
the trains visibly cut corners, so it is no longer a separate thing to remember.

The routing is one independent search per (origin complex, spawn time), so it
runs across the cores, leaving JOB_HEADROOM of them alone so the machine stays
usable. The origins are split into a fixed 64 chunks whichever way it runs, so
--jobs changes how long the build takes and not what comes out of it. About
three minutes on fourteen workers, at roughly 300 MB each.

    python build.py             # full build
    python build.py --sample 20 # every 20th origin, for a quick check
    python build.py --jobs 4    # leave some of the machine alone
    python build.py --no-shapes # skip the track-shaping step
"""

import argparse
import collections
import csv
import io
import json
import math
import os
import random
import re
import sys
import time
from bisect import bisect_left
from datetime import datetime

import numpy as np

import daytype
import lines as LR

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = os.path.join(D, "trains.json")

# A sampled run is a real build of a fake question -- every count in it is low
# by the sampling factor. It gets its own filenames so that checking the
# pipeline can never quietly replace the build the page is showing, which is
# exactly how an afternoon went missing once.
SAMPLE_OUT = os.path.join(D, "trains.sample.json")
STATS_OUT = os.path.join(D, "stats.json")
STATS_SAMPLE_OUT = os.path.join(D, "stats.sample.json")


def out_paths(sample):
    """(trains, stats) to write for this run."""
    if sample > 1:
        return SAMPLE_OUT, STATS_SAMPLE_OUT
    return OUT, STATS_OUT


def write_json(path, obj):
    """Write a whole file or none of it.

    Half a gigabyte of JSON takes long enough to serialise that a Ctrl-C in the
    middle is a real possibility, and a truncated trains.json is a blank map
    with no clue as to why. Write beside the target and rename over it: the
    rename is atomic, so a reader sees either the old file or the new one.
    """
    tmp = path + ".part"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)

OD_HOURLY = os.path.join(D, "od_hourly.npz")


def _day():
    """Which day build_od.py built, read back out of its own output.

    Resolved at import, not in main(), and that is not an accident. The pool
    workers re-import this module and call build_world() for themselves, so a
    value set inside main() would reach the parent and not the children --
    weekday riders would be routed over a Sunday timetable in every worker.
    build.py deliberately takes no --day flag of its own for the same reason:
    one decision, made in build_od.py, recorded in the file.
    """
    if not os.path.exists(OD_HOURLY):
        raise SystemExit("missing %s.\nRun: python build_od.py" % OD_HOURLY)
    with np.load(OD_HOURLY, allow_pickle=True) as z:
        name = str(z["day"]) if "day" in z.files else "nye"
    return name, daytype.get(name)


DAY_NAME, DAY = _day()
OD_DATE = DAY.get("date", "")
SERVICE = DAY["service"]           # 주중주말 code to select in the timetables
MAX_ROUNDS = 4             # journeys of up to 3 transfers
TRANSFER_SEC = 180
DEP_BIN = 150              # spawn riders every 2.5 minutes inside their hour
# 150 s is the knee, not a guess. Simulated against the real departure times
# and hourly volumes, the train-to-train load step falls 1.93 -> 1.36 -> 1.02
# at 600 / 300 / 150 s against a continuous-arrival floor of 0.98. So 150 s
# closes 95% of the gap and 60 s closes 99% -- there is nothing below here
# worth paying for. Cost is held down by spawn_key(), not by coarse bins.
MIN_RIDERS = 0.05          # ignore cells below this many people

# Riders in an hour cell have no arrival time of their own, so they are spread
# over DEP_BIN-wide spawns. Do that on a shared grid and every station in the
# network releases its crowd on the same tick -- at 600 s that was :05, :15,
# :25 and so on -- and whichever train happens to be pulling out just after
# each tick scoops the lot, so trains a few minutes apart on the same line
# carry wildly different loads for no reason in the data. Offsetting each
# origin by a stable fraction of a bin desynchronises them: the network no
# longer breathes in step, and a line's trains fill at the rate the timetable
# and the OD say they should.
# Deterministic, not hash() -- that is salted per process, and the routing runs
# in a pool, so hash() would give a different map every build.
def dep_phase(origin):
    return (int(origin) * 2654435761) % DEP_BIN

# No trip in the regular Sunday timetable starts at or after 00:00 -- only 105
# trains are still finishing their runs, and the last one ends at 00:42. Yet
# the gate counts record roughly 230,000 journeys in the post-midnight bin.
# That gap is the evidence: Seoul extends subway service on New Year's Eve for
# the Bosingak bell, and the 2026 timetable we have does not carry those trains.
#
# So repeat the last two hours of departures, shifted forward, to stand in for
# the extra service. It is a reconstruction, not a record, and it is the one
# invented thing in the pipeline -- set this False to see the night as the
# regular timetable would have it, with the midnight exodus mostly stranded.
#
# **It applies to --day nye and to nothing else.** On any ordinary day the
# timetable and the gate counts agree about when service stops, so there is no
# gap to reconstruct and repeating an hour of departures would be inventing
# trains that did not run.
EXTEND_LAST_HOUR = DAY["extend_late"]

# The "24시이후" column is open-ended: everything after midnight, not
# 00:00-01:00. Compressing it into a single hour put ~90,000 riders onto the
# handful of trains still running and made the dots balloon. Spread it across
# the window the extended service actually covers instead -- two hours on New
# Year's Eve, one on a day when the last train is the last train.
LATE_BIN_HOURS = 2 if DAY["extend_late"] else 1

LINE_COLORS = LR.COLORS

# --------------------------------------------------------------------------
# animation times
#
# Routing uses the timetable exactly as published. Everything below applies
# only to the times written into trains.json for the animation to draw, held
# separately as pattern["vdep"]. Keeping the two apart matters: a rider should
# be put on the train the timetable says they caught, not on one we nudged.
# --------------------------------------------------------------------------

# The published stop times land on twelve distinct second values, 60% of them
# on :00 or :30. Drawn as-is, trains scheduled to the same minute sit exactly
# on top of each other and the map pulses instead of flowing. A small
# deterministic per-stop offset breaks that up; the speed smoothing below then
# absorbs any segment the offset made too fast.
JITTER_S = 10

# Speed ceilings, km/h, by what the line is rather than what it is called.
# Line 1 shares Korail track out to 인천 / 소요산 and really does run at 100+
# between the far-out stops; so do the commuter lines kric.py adds. City metro
# is built for about 80, and the light rail lines are slower still. The
# published timetables are already almost clean against these -- roughly one
# segment in a thousand is over -- so this pass is mostly here to clean up
# after JITTER_S, and to catch the 30-second hops that are artefacts of coarse
# rounding.
SPEED_THRESHOLDS = {
    # 서울 도시철도
    "2": 90, "3": 90, "4": 90, "5": 90,
    "6": 90, "7": 90, "8": 90, "9": 90,
    # Korail 광역전철: long outer stretches on main-line track
    "1": 110, "SB": 110, "GJ": 110, "GC": 110, "SH": 110, "GG": 110,
    # 공항철도 직통 is the fastest thing on the map at 110 design speed
    "AR": 130,
    "SN": 110,          # 신분당선 runs 90-110 between its widely spaced stops
    "I1": 90, "I2": 90,
    # rubber-tyred and light rail
    "UI": 80, "SL": 80, "GP": 90, "UJ": 80,
}
DEFAULT_SPEED_THRESHOLD = 90
SMOOTH_DWELL_S = 20                  # time at a platform, not moving
SMOOTH_ACCEL_DECEL_S = 20            # time lost pulling away and braking
SMOOTH_OVERHEAD_S = SMOOTH_DWELL_S + SMOOTH_ACCEL_DECEL_S
SMOOTH_PASSES = 3                    # chains of fast segments need repeats
SOFT_RESCUE_PASSES = 2               # then let donors drift slightly over
SOFT_DONOR_FACTOR = 1.18
SOFT_TAKE_FRACTION = 0.5

# Waiting bubbles. Arrivals at a platform are bucketed so near-simultaneous
# spawns merge into one step; boarding times are left exact so the bubble
# drops on the same frame the train's own count picks up.
WAIT_BUCKET_S = 20

# Bounds on a single train run, as a backstop against a mis-parsed timetable.
# The longest real run is line 1 end to end at about 2h50m.
MAX_RUN_S = 6 * 3600
MAX_END_S = 30 * 3600


def norm(s):
    return re.sub(r"\s+", "", (s or "").strip())


def base(s):
    return re.sub(r"\(.*?\)$", "", norm(s))


def read_cp949(path):
    with io.open(path, encoding="cp949", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def hhmmss(s):
    s = (s or "").strip()
    if not s or s.count(":") != 2:
        return None
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


# --------------------------------------------------------------------------
# timetable -> patterns
# --------------------------------------------------------------------------

def load_patterns(net, quiet=False):
    """Group the day's trips by their stop sequence."""
    out = (lambda s: None) if quiet else (lambda s: print(s))
    keep = set()
    for c in net["complexes"]:
        for p in c["platforms"]:
            keep.add(p["code"])

    rows = read_cp949(os.path.join(D, "timetable_raw.csv"))
    extra = os.path.join(D, "timetable_extra.csv")
    if os.path.exists(extra):
        more = read_cp949(extra)
        out("   +%s rows from timetable_extra.csv" % format(len(more), ","))
        rows += more
    else:
        out("   timetable_extra.csv not found -- lines 1-9 only. "
            "Run kric.py to add the rest.")
    trips = collections.defaultdict(list)
    for r in rows:
        if norm(r["주중주말"]) != SERVICE:
            continue
        line = norm(r["호선"])
        if line not in LINE_COLORS:
            continue
        code = norm(r["역사코드"])
        if code not in keep:
            continue
        arr, dep = hhmmss(r["열차도착시간"]), hhmmss(r["열차출발시간"])
        # 00:00:00 is this file's null marker for a terminus that only arrives
        # or only departs -- not a train at midnight. Genuine after-midnight
        # times are written 24:xx and up. Read as a real time it puts a zero in
        # the arrival matrix, which then looks like the earliest possible way
        # to reach that stop and drags riders onto the first train of the day.
        if arr == 0:
            arr = None
        if dep == 0:
            dep = None
        if arr is None and dep is None:
            continue
        if arr is None:
            arr = dep
        if dep is None:
            dep = arr
        key = (line, norm(r["방향"]), norm(r["열차코드"]))
        trips[key].append((dep, arr, code, norm(r["급행여부"]) == "1"))

    # Most after-midnight times are written 24:12, but not all -- some trips
    # wrap to 00:12 instead. Sorting those by raw time turns a 90-minute run
    # into a 23-hour one that sits on the map all day, so unwrap first.
    wrapped = 0
    for key, stops in trips.items():
        ts = [s[0] for s in stops] + [s[1] for s in stops]
        if max(ts) - min(ts) > 12 * 3600:
            trips[key] = [(d + 86400 if d < 4 * 3600 else d,
                           a + 86400 if a < 4 * 3600 else a, c, x)
                          for (d, a, c, x) in stops]
            wrapped += 1
    if wrapped:
        out("   unwrapped %d trips written past midnight as 00:xx" % wrapped)

    # Runs that cover ground faster than the line can physically move.
    #
    # 신림선's file carries 30 two-row trips a day -- 관악산 06:37 -> 샛강 06:40,
    # the whole 7.8 km line in three minutes with none of the nine stations
    # between. They are summary rows, not trains. Left in they are drawn as a
    # dot rocketing the length of the line at 400 km/h, and worse, RAPTOR
    # boards riders onto them *because* they are the fastest thing going --
    # 434 riders were teleported on the 2026-09-04 full build.
    #
    # The speed smoothing cannot save these: it borrows seconds from slower
    # neighbouring segments, and a two-stop run has no neighbours. So drop
    # them here, on end-to-end straight-line speed against the line's own
    # ceiling with a wide margin. Real service is nowhere near it -- straight
    # line understates the track, and dwells are included -- so the fastest
    # genuine run on the map, 공항철도 직통, comes out at less than half its
    # threshold.
    coord_of = {}
    for c in net["complexes"]:
        for p in c["platforms"]:
            coord_of[p["code"]] = (c["lat"], c["lon"])

    impossible = collections.Counter()
    for key in list(trips):
        stops = sorted(trips[key])
        a, b = coord_of.get(stops[0][2]), coord_of.get(stops[-1][2])
        dt = stops[-1][1] - stops[0][0]
        if not a or not b or dt <= 0:
            continue
        dy = (a[0] - b[0]) * 111320.0
        dx = (a[1] - b[1]) * 111320.0 * math.cos(math.radians(a[0]))
        kmh = math.hypot(dx, dy) / dt * 3.6
        ceiling = SPEED_THRESHOLDS.get(key[0], DEFAULT_SPEED_THRESHOLD)
        if kmh > ceiling * 1.5:
            impossible[(key[0], len(stops))] += 1
            del trips[key]
    if impossible:
        out("   dropped %d trip(s) faster than their line can run:"
            % sum(impossible.values()))
        for (line, nstops), n in sorted(impossible.items()):
            out("      line %-4s %d x %d-stop run" % (line, n, nstops))

    if EXTEND_LAST_HOUR:
        added = 0
        for key in list(trips):
            stops = trips[key]
            first = min(s[0] for s in stops)
            # 23:00 hour repeated at +1h, and the whole 22:00-24:00 block at
            # +2h, so service thins out past midnight the way a real late
            # night does rather than stopping dead at 01:00.
            for lo, hi, shift, tag in ((23, 24, 3600, "+X1"),
                                       (22, 24, 7200, "+X2")):
                if lo * 3600 <= first < hi * 3600:
                    trips[(key[0], key[1], key[2] + tag)] = [
                        (d + shift, a + shift, c, x) for (d, a, c, x) in stops]
                    added += 1
        out("   +%d reconstructed late trips (EXTEND_LAST_HOUR)" % added)

    patterns = {}
    for key, stops in trips.items():
        stops.sort()
        codes = tuple(s[2] for s in stops)
        if len(codes) < 2:
            continue
        pkey = (key[0], key[1], codes)
        p = patterns.setdefault(pkey, {
            "line": key[0], "dir": key[1], "stops": list(codes),
            "express": stops[0][3], "trips": [],
        })
        p["trips"].append({
            "id": "%s-%s-%s" % key,
            "arr": [s[1] for s in stops],
            "dep": [s[0] for s in stops],
        })

    out = []
    for pkey, p in patterns.items():
        p["trips"].sort(key=lambda t: t["dep"][0])
        p["dep"] = np.array([t["dep"] for t in p["trips"]], dtype=np.int32)
        p["arr"] = np.array([t["arr"] for t in p["trips"]], dtype=np.int32)
        out.append(p)
    return out


def build_index(patterns):
    stop_pats = collections.defaultdict(list)
    for pi, p in enumerate(patterns):
        for si, code in enumerate(p["stops"]):
            stop_pats[code].append((pi, si))
    return stop_pats


# --------------------------------------------------------------------------
# animation times: jitter, then smooth away impossible speeds
#
# Ported from londonriders, where minute-rounded PDF timetables make this
# essential. Seoul's timetable is cleaner -- it carries seconds, and only about
# one segment in a thousand was over its line's ceiling before we touched it --
# so the smoothing here is mostly cleaning up after our own jitter.
# --------------------------------------------------------------------------

def metres(a, b):
    """Flat-earth distance between two (lat, lon) pairs. Fine at city scale."""
    dy = (a[0] - b[0]) * 111320.0
    dx = (a[1] - b[1]) * 111320.0 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def cruise_kmh(dist_m, dt_s):
    """Speed while actually moving, i.e. after the dwell and the accel taper."""
    cruise = dt_s - SMOOTH_OVERHEAD_S
    if cruise <= 0:
        return 9999.0
    return dist_m / cruise * 3.6


def _time_for_speed(dist_m, kmh):
    return dist_m / (kmh / 3.6) + SMOOTH_OVERHEAD_S


def _smooth_once(times, dists, hard, target_factor=1.0, take_fraction=1.0,
                 donor_factor=1.0):
    """One smoothing pass over a single trip's stop times, in place.

    Any segment above target_factor * hard borrows seconds from whichever
    neighbours still have slack, which slows it without changing when the trip
    starts or ends. A donor may only give up time it can spare while staying
    under donor_factor * hard itself, so fixing one segment never breaks the
    next one.

    Returns (n_over, n_fixed): how many segments were too fast, and how many
    got all the time they asked for.
    """
    n = len(times)
    seg_target = hard * target_factor
    donor_cap = hard * donor_factor
    n_over = n_fixed = 0
    for j in range(n - 1):
        d = dists[j]
        if d is None:
            continue
        dt = times[j + 1] - times[j]
        if dt <= 0:
            continue
        if cruise_kmh(d, dt) <= seg_target:
            continue
        n_over += 1
        want = max(1, int(math.ceil(
            (math.ceil(_time_for_speed(d, seg_target)) - dt) * take_fraction)))

        before = 0
        if j > 0 and dists[j - 1] is not None:
            spare = (times[j] - times[j - 1]) - int(math.ceil(
                _time_for_speed(dists[j - 1], donor_cap)))
            before = max(0, spare)
        after = 0
        if j + 2 < n and dists[j + 1] is not None:
            spare = (times[j + 2] - times[j + 1]) - int(math.ceil(
                _time_for_speed(dists[j + 1], donor_cap)))
            after = max(0, spare)

        avail = before + after
        if avail == 0:
            continue
        take = min(want, avail)
        take_before = int(round(take * before / float(avail)))
        take_after = take - take_before
        if take_before:
            times[j] -= take_before
        if take_after:
            times[j + 1] += take_after
        if take >= want:
            n_fixed += 1
    return n_over, n_fixed


def build_visual_times(patterns, coord, quiet=False):
    """Fill pattern["vdep"]: the departure times the animation draws.

    Same shape as pattern["dep"], which is left untouched for RAPTOR. Each
    trip is jittered by a few seconds per stop, then smoothed until no segment
    claims a speed its line cannot do.
    """
    if not quiet:
        print("preparing animation times (jitter +/-%ds, then speed "
              "smoothing) ..." % JITTER_S)
    rng = random.Random(7)

    over_before = over_after = total_segs = 0
    for p in patterns:
        stops = p["stops"]
        hard = SPEED_THRESHOLDS.get(p["line"], DEFAULT_SPEED_THRESHOLD)
        # Distances depend only on the stop sequence, so one list serves every
        # trip on the pattern.
        dists = []
        for i in range(len(stops) - 1):
            a, b = coord.get(stops[i]), coord.get(stops[i + 1])
            dists.append(metres(a, b) if a and b else None)

        vdep = p["dep"].astype(np.int64).copy()
        for ti in range(vdep.shape[0]):
            times = [int(x) for x in vdep[ti]]

            for j in range(len(times) - 1):
                if dists[j] is not None and times[j + 1] > times[j]:
                    total_segs += 1
                    if cruise_kmh(dists[j], times[j + 1] - times[j]) > hard:
                        over_before += 1

            prev = -10 ** 9
            for j in range(len(times)):
                times[j] += rng.randint(-JITTER_S, JITTER_S)
                if times[j] <= prev:
                    times[j] = prev + 1
                prev = times[j]

            for _ in range(SMOOTH_PASSES):
                n_over, _ = _smooth_once(times, dists, hard)
                if n_over == 0:
                    break
            for _ in range(SOFT_RESCUE_PASSES):
                n_over, _ = _smooth_once(times, dists, hard,
                                         donor_factor=SOFT_DONOR_FACTOR,
                                         take_fraction=SOFT_TAKE_FRACTION)
                if n_over == 0:
                    break

            for j in range(len(times) - 1):
                if dists[j] is not None and times[j + 1] > times[j]:
                    if cruise_kmh(dists[j], times[j + 1] - times[j]) > hard:
                        over_after += 1
            vdep[ti] = times
        p["vdep"] = vdep

    if not quiet:
        print("   segments over their line ceiling: %s -> %s of %s"
              % (format(over_before, ","), format(over_after, ","),
                 format(total_segs, ",")))


# --------------------------------------------------------------------------
# waiting bubbles
# --------------------------------------------------------------------------

def build_waiting_timelines(deltas, ncomplex):
    """Turn per-complex {time: +/- riders} into [[t, count], ...] step curves.

    An entry only goes in when the crowd changes by enough to move the bubble
    on screen. The page draws radius = sqrt(count) * k, so at 2,000 waiting it
    takes about nine people to shift the edge by a twentieth of a pixel -- and
    a busy station changes by one person hundreds of times an hour. Writing
    every one of those costs megabytes and draws nothing.
    """
    out = [[] for _ in range(ncomplex)]
    entries = 0
    for ci, bucket in deltas.items():
        cum = 0.0
        last = 0.0
        tl = []
        for t in sorted(bucket):
            cum += bucket[t]
            if cum < 0.0:
                cum = 0.0
            step = max(1.0, 0.2 * math.sqrt(max(cum, last)))
            if abs(cum - last) >= step or (cum < 0.5) != (last < 0.5):
                tl.append([int(t), int(round(cum))])
                last = cum
        out[ci] = tl
        entries += len(tl)
    nonempty = sum(1 for tl in out if tl)
    print("   waiting timelines: %d stations, %s entries (avg %d each)"
          % (nonempty, format(entries, ","), entries // max(1, nonempty)))
    return out


# --------------------------------------------------------------------------
# RAPTOR
# --------------------------------------------------------------------------

INF = 1 << 30


def prepare_scan(patterns, stop_pats, transfers, codes_idx):
    """Re-shape the routing tables into plain Python lists.

    RAPTOR is a tight interpreted loop over scalars, and numpy is the wrong
    container for that: every deps[ti, sj] builds a boxed scalar, and every
    np.searchsorted on a column view costs about 2.7us of dispatch before it
    does any work. The same data as lists of ints is roughly ten times faster
    to walk, and the arrays stay on the pattern for everything else.

    Departure times are kept column-major (one list per stop, across trips)
    because that is what the "which trip can I catch here" search bisects;
    arrivals are kept row-major (one list per trip) because that is read along
    a trip once boarded.
    """
    n = len(codes_idx)
    for p in patterns:
        dep, arr = p["dep"], p["arr"]
        p["sidx"] = [codes_idx[c] for c in p["stops"]]
        p["depc"] = [[int(v) for v in dep[:, s]] for s in range(dep.shape[1])]
        p["arrr"] = [[int(v) for v in row] for row in arr]
        p["ntr"] = int(dep.shape[0])
        p["ns"] = len(p["stops"])
        # Drawn departure times, read a trip at a time when riders are
        # booked onto one -- the same reason arrivals are row-major.
        p["vdepr"] = [[int(v) for v in row] for row in p["vdep"]]
        # Trips are sorted by their first departure, which normally leaves
        # every later column sorted too -- but three patterns have a train
        # overtaking another partway along, and the bisect shortcut below only
        # holds where the column really does rise.
        p["srt"] = bool(np.all(dep[:-1, :] <= dep[1:, :])) if dep.shape[0] > 1 \
            else True

    stop_pats_i = [()] * n
    for code, v in stop_pats.items():
        if code in codes_idx:
            stop_pats_i[codes_idx[code]] = tuple(v)
    transfers_i = [()] * n
    for i, v in transfers.items():
        transfers_i[i] = tuple(v)

    # Every departure at each stop, sorted. Only used to tell whether two
    # spawn times can share one RAPTOR result -- see spawn_key().
    dep_at = [[] for _ in range(n)]
    for p in patterns:
        for s, si in enumerate(p["sidx"]):
            dep_at[si].extend(p["depc"][s])
    dep_at = [sorted(v) for v in dep_at]
    return stop_pats_i, transfers_i, dep_at


def spawn_seeds(origin_idx, transfers):
    """The stops RAPTOR seeds, and the walk cost to reach each.

    The origin's own platforms at zero, plus whatever a transfer reaches, at
    its cost. A stop reachable both ways keeps the cheaper offset, which is
    what the seeding loop in raptor() settles on too.
    """
    seeds = {}
    for i in origin_idx:
        seeds[i] = 0
    for i in origin_idx:
        for j, cost in transfers[i]:
            if cost < seeds.get(j, 1 << 30):
                seeds[j] = cost
    return tuple(sorted(seeds.items()))


def spawn_key(seeds, dep_at, t):
    """What a RAPTOR search from `t` actually depends on.

    A search leaving at `t` is decided entirely by which departure is next at
    each seeded stop. Nothing else about `t` reaches the answer: every label
    from round 1 on is an absolute timetable time, so two spawn times with the
    same key produce bit-identical tau, parent and best -- apart from the round
    0 labels at the seeds themselves, which the caller never reads. So the
    result can be reused, and at 150 s bins most of them can: a quiet outer
    station sees four departures an hour against twenty-four spawns.

    Cheap enough to be worth it -- a handful of bisects against a RAPTOR run.
    """
    return tuple(bisect_left(dep_at[s], t + c) for s, c in seeds)


def raptor(origin_idx, dep_time, patterns, stop_pats, transfers, n, npat,
           _first=None):
    """RAPTOR with one arrival label per round.

    Keeping a single overwritten parent pointer does not work: a later round
    can improve a stop that an earlier leg was chained through, and the
    backtrace then follows a state that no longer exists. Labels are per round
    (tau[k] = earliest arrival using at most k trips) so the trace can walk
    k downwards and always land on a state that really happened.

    Stops are addressed by index throughout; stop_pats and transfers are the
    lists prepare_scan() built, not the dicts.
    """
    tau = [[INF] * n for _ in range(MAX_ROUNDS + 1)]
    parent = [{} for _ in range(MAX_ROUNDS + 1)]
    best = [INF] * n
    # Earliest stop each pattern was reached at. Entries are cleared as the
    # round consumes them, so the caller can hand the same scratch list to
    # every search instead of allocating one per round.
    first_si = _first if _first is not None else [-1] * npat

    t0 = tau[0]
    marked = set()
    for i in origin_idx:
        t0[i] = dep_time
        best[i] = dep_time
        marked.add(i)
    for i in list(marked):
        for j, cost in transfers[i]:
            t = dep_time + cost
            if t < t0[j]:
                t0[j] = t
                best[j] = t
                parent[0][j] = ("walk", i)
                marked.add(j)

    for k in range(1, MAX_ROUNDS + 1):
        tk = tau[k]
        tprev = tau[k - 1]
        tk[:] = tprev
        pk = parent[k]

        order = []
        for i in marked:
            for pi, si in stop_pats[i]:
                f = first_si[pi]
                if f < 0:
                    first_si[pi] = si
                    order.append(pi)
                elif si < f:
                    first_si[pi] = si
        if not order:
            break

        newly = set()
        for pi in order:
            si0 = first_si[pi]
            first_si[pi] = -1
            p = patterns[pi]
            depc, arrr, sidx, srt = p["depc"], p["arrr"], p["sidx"], p["srt"]
            ntr = p["ntr"]
            ti = -1
            board_si = -1
            arow = None
            for sj in range(si0, len(sidx)):
                j = sidx[sj]
                if arow is not None:
                    t = arow[sj]
                    if t < best[j]:
                        tk[j] = t
                        best[j] = t
                        pk[j] = ("ride", pi, ti, board_si, sj, sidx[board_si])
                        newly.add(j)
                # Could we have caught an earlier trip by boarding here? Only
                # worth a search when we are not already on a trip that leaves
                # this stop no later than we could reach it -- that guard is
                # what keeps the scan near O(stops) instead of O(stops log n).
                prev = tprev[j]
                if prev < INF:
                    col = depc[sj]
                    if ti < 0:
                        cand = bisect_left(col, prev)
                        if cand < ntr:
                            ti, board_si, arow = cand, sj, arrr[cand]
                    elif srt:
                        # The column rises with trip index, so an earlier
                        # boardable trip exists only if the one immediately
                        # before ours is still catchable. One lookup answers
                        # that, and it skips most of the bisects.
                        if ti and col[ti - 1] >= prev:
                            ti = bisect_left(col, prev, 0, ti)
                            board_si, arow = sj, arrr[ti]
                    elif prev <= col[ti]:
                        cand = bisect_left(col, prev)
                        if cand < ti:
                            ti, board_si, arow = cand, sj, arrr[cand]

        for j in list(newly):
            for m, cost in transfers[j]:
                t = tk[j] + cost
                if t < tk[m] and t < best[m]:
                    tk[m] = t
                    best[m] = t
                    pk[m] = ("walk", j)
                    newly.add(m)
        marked = newly
        if not marked:
            break
    return tau, parent, best


def trace(dest_i, tau, parent, origin_set):
    """Ride legs from origin to destination, earliest first."""
    k = min(range(len(tau)), key=lambda r: tau[r][dest_i])
    if tau[k][dest_i] >= INF:
        return None
    legs = []
    cur = dest_i
    for _ in range(64):
        if cur in origin_set:
            legs.reverse()
            return legs
        P = parent[k].get(cur)
        if P is None:
            if k == 0:
                return None
            k -= 1                      # label was carried forward unchanged
            continue
        if P[0] == "walk":
            cur = P[1]
            continue
        _, pi, ti, si, sj, board_i = P
        legs.append((pi, ti, si, sj))
        cur = board_i
        k -= 1
        if k < 0:
            return None
    return None


# --------------------------------------------------------------------------

class CodeIndex(object):
    def __init__(self, codes):
        self._f = dict((c, i) for i, c in enumerate(codes))
        self.inv = list(codes)

    def __getitem__(self, c):
        return self._f[c]

    def __contains__(self, c):
        return c in self._f

    def __len__(self):
        return len(self.inv)


# --------------------------------------------------------------------------
# the routing world
#
# Everything the origin loop reads and nothing it writes. Worker processes
# rebuild it from disk rather than being handed a copy: the scan tables are a
# few million small Python ints, which parse faster than they unpickle, and
# every step that makes them is deterministic, so a worker's pattern indices
# come out matching the parent's. world_fingerprint() is the proof of that,
# checked once per worker rather than assumed.
# --------------------------------------------------------------------------

# Riders waiting on a platform are keyed by (complex, arrival bucket) packed
# into one int, so a chunk's contribution ships as two flat arrays. The
# multiplier only has to clear the largest bucket index, and the timetable ends
# before 30:00.
WKEY = 1 << 20


class World(object):
    pass


def build_world(quiet=False):
    out = (lambda s: None) if quiet else (lambda s: print(s))

    out("loading network ...")
    with io.open(os.path.join(D, "stations.json"), encoding="utf-8") as f:
        net = json.load(f)
    complexes = net["complexes"]

    out("loading timetable patterns (%s service) ..." % SERVICE)
    patterns = load_patterns(net, quiet=quiet)
    ntrips = sum(len(p["trips"]) for p in patterns)
    out("   %d patterns, %d trips" % (len(patterns), ntrips))

    codes = sorted(set(c for p in patterns for c in p["stops"]))
    codes_idx = CodeIndex(codes)
    stop_pats = build_index(patterns)
    out("   %d platforms served" % len(codes))

    # platform -> complex, and transfers within a complex
    code_cx = {}
    cx_codes = collections.defaultdict(list)
    for i, c in enumerate(complexes):
        for p in c["platforms"]:
            if p["code"] in codes_idx:
                code_cx[p["code"]] = i
                cx_codes[i].append(p["code"])

    coord = {}
    for c in complexes:
        for p in c["platforms"]:
            coord[p["code"]] = (c["lat"], c["lon"])

    build_visual_times(patterns, coord, quiet=quiet)

    transfers = collections.defaultdict(list)
    for i, cl in cx_codes.items():
        for a in cl:
            for b in cl:
                if a != b:
                    transfers[codes_idx[a]].append((codes_idx[b], TRANSFER_SEC))

    stop_pats_i, transfers_i, dep_at = prepare_scan(
        patterns, stop_pats, transfers, codes_idx)

    out("loading hourly OD ...")
    z = np.load(os.path.join(D, "od_hourly.npz"), allow_pickle=True)
    X, pairs, hours = z["x"], z["pairs"], z["hours"]
    out("   %d pairs x %d hours, %s riders"
        % (len(pairs), len(hours), format(int(X.sum()), ",")))

    by_origin = collections.defaultdict(list)
    for k, (o, d) in enumerate(pairs):
        by_origin[int(o)].append(k)

    w = World()
    w.net, w.complexes, w.coord = net, complexes, coord
    w.patterns = patterns
    w.codes_idx, w.n, w.npat = codes_idx, len(codes), len(patterns)
    w.stop_pats, w.transfers = stop_pats_i, transfers_i
    w.dep_at = dep_at
    w.cx_codes, w.code_cx = cx_codes, code_cx
    w.X, w.pairs, w.hours, w.by_origin = X, pairs, hours, by_origin

    # (pattern, trip, stop) -> one slot in a flat array, so a chunk of origins
    # can hand its boardings back as numbers rather than nested dicts.
    off = 0
    w.pat_off = []
    for p in patterns:
        w.pat_off.append(off)
        off += p["ntr"] * p["ns"]
    w.nslot = off
    return w


def world_fingerprint(w):
    """Enough of the world to catch a worker that built a different one."""
    return (w.npat, w.n, w.nslot,
            int(sum(int(p["vdep"].sum()) for p in w.patterns)))


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

def _pack(d):
    if not d:
        return (np.zeros(0, np.int64), np.zeros(0, np.float64))
    return (np.fromiter(d.keys(), np.int64, len(d)),
            np.fromiter(d.values(), np.float64, len(d)))


def route_origins(w, origins):
    """Board every rider leaving these origin complexes.

    Returns the chunk's contribution as flat (index, value) arrays rather than
    the nested dicts the output pass wants: a worker has to ship this back down
    a pipe, and a hundred thousand nested float entries pickle for longer than
    the routing itself took.
    """
    patterns, pat_off = w.patterns, w.pat_off
    stop_pats, transfers = w.stop_pats, w.transfers
    n, npat = w.n, w.npat
    cx_codes, code_cx, codes_idx = w.cx_codes, w.code_cx, w.codes_idx
    X, pairs, hours, by_origin = w.X, w.pairs, w.hours, w.by_origin
    dep_at = w.dep_at
    reused = searched = 0

    board = collections.defaultdict(float)
    alight = collections.defaultdict(float)
    # complex -> {arrival bucket: riders}, positive side only. The matching
    # negative -- the moment the train takes them away -- is exactly the
    # boarding, so merge_chunks() reads it back off the board array rather than
    # sending the same numbers twice.
    wait = collections.defaultdict(float)
    xtab = collections.Counter()
    unrouted_by_hour = collections.Counter()
    routed = unrouted = late = 0.0
    first_si = [-1] * npat

    for o in origins:
        ocodes = [c for c in cx_codes.get(o, []) if c in codes_idx]
        if not ocodes:
            continue
        oidx = [codes_idx[c] for c in ocodes]
        origin_set = set(oidx)
        ks = by_origin[o]
        seeds = spawn_seeds(oidx, transfers)
        # One-deep, and that is enough: spawns are walked in time order, so
        # the runs that share a key are consecutive.
        last_key, last_res = None, None

        for hi, hour in enumerate(hours):
            vol = X[ks, hi]
            live = np.where(vol > MIN_RIDERS)[0]
            if len(live) == 0:
                continue

            span = LATE_BIN_HOURS if int(hour) == 24 else 1
            nbins = max(1, span * 3600 // DEP_BIN)
            phase = dep_phase(o)
            fb = None
            for b in range(nbins):
                t = int(hour) * 3600 + b * DEP_BIN + phase
                key = spawn_key(seeds, dep_at, t)
                if key == last_key:
                    tau, parent, best = last_res
                    reused += 1
                else:
                    tau, parent, best = raptor(oidx, t, patterns, stop_pats,
                                               transfers, n, npat, first_si)
                    last_key, last_res = key, (tau, parent, best)
                    searched += 1
                if b == 0:
                    fb = (tau, parent, best)
                for li in live:
                    k = ks[li]
                    d = int(pairs[k][1])
                    riders = float(vol[li]) / nbins
                    dcodes = [c for c in cx_codes.get(d, []) if c in codes_idx]
                    if not dcodes:
                        unrouted += riders; unrouted_by_hour[int(hour)] += riders
                        continue
                    dis = [codes_idx[c] for c in dcodes]

                    legs = None
                    di = min(dis, key=lambda i: best[i])
                    if best[di] < INF:
                        legs = trace(di, tau, parent, origin_set)
                    if legs is None and fb is not None and b > 0:
                        # No train left at this spawn time. Put the rider on
                        # the last one that did run rather than delete them.
                        ftau, fparent, fbest = fb
                        di = min(dis, key=lambda i: fbest[i])
                        if fbest[di] < INF:
                            legs = trace(di, ftau, fparent, origin_set)
                            if legs:
                                late += riders
                    if not legs:
                        unrouted += riders; unrouted_by_hour[int(hour)] += riders
                        continue
                    routed += riders
                    on_platform = t
                    for (pi, ti, si, sj) in legs:
                        P = patterns[pi]
                        off = pat_off[pi] + ti * P["ns"]
                        board[off + si] += riders
                        alight[off + sj] += riders
                        xtab[(int(hour), P["depc"][si][ti] // 3600)] += riders
                        # Wait on the platform from arriving there until the
                        # train pulls out. A rider who fell back to an earlier
                        # train boards before they spawned, so clamp rather
                        # than let the crowd go negative.
                        vrow = P["vdepr"][ti]
                        board_t = vrow[si]
                        arrive_t = min(on_platform, board_t)
                        wait[code_cx[P["stops"][si]] * WKEY
                             + arrive_t // WAIT_BUCKET_S] += riders
                        on_platform = vrow[sj] + TRANSFER_SEC

    return {"board": _pack(board), "alight": _pack(alight),
            "wait": _pack(wait), "xtab": dict(xtab),
            "unrouted_by_hour": dict(unrouted_by_hour),
            "routed": routed, "unrouted": unrouted, "late": late,
            "reused": reused, "searched": searched,
            "fp": world_fingerprint(w)}


def merge_chunks(w, chunks):
    """Fold the chunks back into the nested form the output pass wants."""
    board = np.zeros(w.nslot)
    alight = np.zeros(w.nslot)
    wait = collections.defaultdict(float)
    xtab = collections.Counter()
    unrouted_by_hour = collections.Counter()
    routed = unrouted = late = 0.0
    reused = searched = 0
    for c in chunks:
        idx, val = c["board"]; board[idx] += val
        idx, val = c["alight"]; alight[idx] += val
        idx, val = c["wait"]
        for key, v in zip(idx.tolist(), val.tolist()):
            wait[key] += v
        xtab.update(c["xtab"])
        unrouted_by_hour.update(c["unrouted_by_hour"])
        routed += c["routed"]; unrouted += c["unrouted"]; late += c["late"]
        reused += c.get("reused", 0); searched += c.get("searched", 0)

    boardings, alightings = {}, {}
    for pi, p in enumerate(w.patterns):
        off, ns, ntr = w.pat_off[pi], p["ns"], p["ntr"]
        bb = board[off:off + ntr * ns].reshape(ntr, ns)
        aa = alight[off:off + ntr * ns].reshape(ntr, ns)
        for ti in np.nonzero(bb.any(1) | aa.any(1))[0].tolist():
            rb, ra = bb[ti], aa[ti]
            boardings[(pi, ti)] = dict(
                (int(si), float(rb[si])) for si in np.nonzero(rb)[0])
            alightings[(pi, ti)] = dict(
                (int(si), float(ra[si])) for si in np.nonzero(ra)[0])

    wait_deltas = collections.defaultdict(dict)
    for key, v in wait.items():
        ci, bkt = divmod(key, WKEY)
        dd = wait_deltas[ci]
        t = bkt * WAIT_BUCKET_S
        dd[t] = dd.get(t, 0.0) + v
    for (pi, ti), bd in boardings.items():
        p = w.patterns[pi]
        vrow, stops = p["vdepr"][ti], p["stops"]
        for si, r in bd.items():
            dd = wait_deltas[w.code_cx[stops[si]]]
            t = vrow[si]
            dd[t] = dd.get(t, 0.0) - r

    return {"boardings": boardings, "alightings": alightings,
            "wait_deltas": wait_deltas, "xtab": xtab,
            "unrouted_by_hour": unrouted_by_hour,
            "routed": routed, "unrouted": unrouted, "late": late,
            "reused": reused, "searched": searched}


# --------------------------------------------------------------------------
# worker side
#
# Origins are cut into the same fixed number of chunks whichever way the build
# runs, and merged back in chunk order, so --jobs changes how long it takes and
# not what comes out.
# --------------------------------------------------------------------------

NCHUNK = 64

# Cores left alone by default. A full build is minutes of every core otherwise,
# which is enough to make the rest of the machine unpleasant to use while it
# runs -- and it is rarely the only thing running.
JOB_HEADROOM = 2

_W = None


def _worker_init():
    global _W
    sys.stdout = io.open(os.devnull, "w")
    _W = build_world(quiet=True)


def _worker_chunk(job):
    ci, origins = job
    return ci, route_origins(_W, origins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1,
                    help="use every Nth origin complex (1 = all)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes for the routing (0 = one per core "
                         "bar %d, 1 = stay in this one)" % JOB_HEADROOM)
    ap.add_argument("--no-shapes", action="store_true",
                    help="skip the build_shapes.py pass at the end")
    args = ap.parse_args()

    w = build_world()

    origins = sorted(w.by_origin)
    if args.sample > 1:
        origins = origins[::args.sample]
        print("   sampling %d origins" % len(origins))

    jobs = args.jobs or max(1, (os.cpu_count() or 1) - JOB_HEADROOM)
    jobs = max(1, min(jobs, NCHUNK, len(origins)))
    chunks = [(i, [int(o) for o in c]) for i, c in
              enumerate(np.array_split(np.array(origins), NCHUNK))]
    chunks = [(i, c) for i, c in chunks if c]

    fp = world_fingerprint(w)
    results = [None] * len(chunks)
    state = {"done": 0}
    t0 = time.time()
    print("routing %d origins in %d chunks on %d process%s ..."
          % (len(origins), len(chunks), jobs, "" if jobs == 1 else "es"))

    def report():
        # Spawning the workers costs as much as several chunks, so the first
        # one is not worth extrapolating from -- start the clock again once it
        # lands and estimate off what the pool does at full speed.
        done = state["done"]
        el = time.time() - t0
        if done == 1:
            state["t1"] = time.time()
            print("   chunk  1/%d  %4.0fs elapsed" % (len(chunks), el))
        else:
            per = (time.time() - state["t1"]) / (done - 1)
            print("   chunk %2d/%d  %4.0fs elapsed, ~%.0fs left"
                  % (done, len(chunks), el, per * (len(chunks) - done)))
        sys.stdout.flush()

    if jobs == 1:
        for ci, part in chunks:
            results[ci] = route_origins(w, part)
            state["done"] += 1
            report()
    else:
        import multiprocessing as mp
        pool = mp.Pool(jobs, initializer=_worker_init)
        try:
            for ci, res in pool.imap_unordered(_worker_chunk, chunks):
                if res["fp"] != fp:
                    raise RuntimeError(
                        "a worker built a different timetable than this "
                        "process (%r vs %r) -- the routing tables would not "
                        "line up" % (res["fp"], fp))
                results[ci] = res
                state["done"] += 1
                report()
        finally:
            pool.close()
            pool.join()

    acc = merge_chunks(w, results)
    boardings, alightings = acc["boardings"], acc["alightings"]
    wait_deltas = acc["wait_deltas"]
    routed, unrouted, late = acc["routed"], acc["unrouted"], acc["late"]
    unrouted_by_hour, xtab = acc["unrouted_by_hour"], acc["xtab"]
    net, patterns, complexes = w.net, w.patterns, w.complexes
    coord, code_cx = w.coord, w.code_cx

    ran, saved = acc["searched"], acc["reused"]
    print("\nRAPTOR searches %s, reused %s (%.0f%% of spawns answered from "
          "the previous search)"
          % (format(ran, ","), format(saved, ","),
             100.0 * saved / max(ran + saved, 1)))
    print("\nrouted %s riders, %s unrouted (%.2f%%)"
          % (format(int(routed), ","), format(int(unrouted), ","),
             100.0 * unrouted / max(routed + unrouted, 1)))
    print("   of those routed, %s fell back to the last scheduled train (%.2f%%)"
          % (format(int(late), ","), 100.0 * late / max(routed, 1)))
    # Sanity check worth keeping: a rider should board within an hour or so of
    # spawning. A large figure here means people are being put on trains that
    # left long before they arrived -- which is how the 00:00:00 null-time bug
    # showed itself, as 15% of all boardings landing on the first train of the
    # day.
    tot_x = sum(xtab.values())
    far = sum(v for (sh, dh), v in xtab.items() if abs(sh - dh) > 1)
    print("   boarding a train more than an hour off the spawn hour: "
          "%s of %s (%.2f%%)"
          % (format(int(far), ","), format(int(tot_x), ","),
             100.0 * far / max(tot_x, 1)))
    if unrouted_by_hour:
        print("unrouted by departure hour:")
        for h in sorted(unrouted_by_hour):
            print("   %02d:00  %10s" % (h % 24,
                                        format(int(unrouted_by_hour[h]), ",")))

    # Stamped into both outputs so the page can say which build it is
    # drawing, and so a sampled one can say so out loud rather than looking
    # like a quiet Sunday.
    stamp = {
        "built": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "sample": args.sample,
        "origins": len(origins),
        "riders": int(round(routed)),
        "unrouted": int(round(unrouted)),
        "shaped": not args.no_shapes,
    }
    write_output(net, patterns, boardings, alightings, complexes, coord,
                 wait_deltas, code_cx, stamp)

    if args.no_shapes:
        print("\nskipping build_shapes (--no-shapes): trains will cut corners")
    else:
        print("")
        import build_shapes
        build_shapes.main(out_paths(args.sample)[0])


def write_output(net, patterns, boardings, alightings, complexes, coord,
                 wait_deltas, code_cx, stamp):
    print("building output ...")
    trains = []
    absurd = []
    for (pi, ti), bd in boardings.items():
        p = patterns[pi]
        al = alightings.get((pi, ti), {})
        vdep = p["vdep"][ti]
        onboard = 0.0
        timeline = []
        for si, code in enumerate(p["stops"]):
            onboard += bd.get(si, 0.0) - al.get(si, 0.0)
            onboard = max(0.0, onboard)
            xy = coord.get(code)
            if not xy:
                continue
            timeline.append([int(vdep[si]), round(xy[0], 5),
                             round(xy[1], 5), round(onboard, 1),
                             round(bd.get(si, 0.0), 1)])
        if not timeline or max(r[3] for r in timeline) < 0.5:
            continue
        # A single bad stop time used to stretch the page's time slider across
        # nine days, because the slider spans the last train it is given. The
        # longest real run on the network is line 1 at just under three hours,
        # so anything past MAX_RUN_S is a parse failure upstream and the run is
        # dropped rather than allowed to set the scale for everything else.
        if timeline[-1][0] - timeline[0][0] > MAX_RUN_S or                 timeline[-1][0] > MAX_END_S:
            absurd.append((p["line"], timeline[0][0], timeline[-1][0]))
            continue
        trains.append({
            "route": p["line"],
            "color": LINE_COLORS.get(p["line"], "#888"),
            "express": bool(p["express"]),
            "timeline": timeline,
        })

    if absurd:
        print("   DROPPED %d runs with impossible times -- the timetable parse "
              "is wrong upstream, run validate.py:" % len(absurd))
        for line, t0, t1 in sorted(absurd, key=lambda x: -x[2])[:6]:
            print("      line %-4s %.1fh -> %.1fh" % (line, t0 / 3600.0,
                                                      t1 / 3600.0))

    per_line = collections.Counter(t["route"] for t in trains)
    print("   %d trains carrying riders" % len(trains))
    for lid in sorted(per_line, key=LR.order_key):
        print("      %-4s %-14s %5d" % (lid, LR.DISPLAY.get(lid, lid),
                                        per_line[lid]))
    quiet = [l for l in LR.ALL_IDS if l not in per_line]
    if quiet:
        print("      NO TRAINS WITH RIDERS: %s" % ",".join(quiet))

    lines = []
    for line, ways in net["geometry"].items():
        lines.append({"route": line,
                      "color": LINE_COLORS.get(line, "#888"),
                      "ways": [[[round(a, 5), round(b, 5)] for a, b in w]
                               for w in ways]})
    line_meta = net.get("line_meta") or dict(
        (l.id, {"display": l.display, "display_en": l.display_en,
                "color": l.color, "capacity": l.capacity}) for l in LR.LINES)

    waiting = build_waiting_timelines(wait_deltas, len(complexes))
    stations = [{"name": c["name"], "name_en": c.get("name_en", ""),
                 "lat": c["lat"], "lon": c["lon"],
                 "lines": sorted(set(p["line"] for p in c["platforms"])),
                 "boardings": c["od_boardings"],
                 "measured": c["measured"],
                 "wait": waiting[i]}
                for i, c in enumerate(complexes)]

    stamp = dict(stamp, trains=len(trains))
    out = {"date": OD_DATE, "day": DAY_NAME, "build": stamp, "stations": stations,
           "lines": lines, "line_meta": line_meta, "trains": trains}
    path = out_paths(stamp["sample"])[0]
    write_json(path, out)
    print("wrote %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))

    write_stats(patterns, boardings, alightings, complexes, coord, code_cx,
                line_meta, stamp)


# --------------------------------------------------------------------------
# the day in aggregate, for the static view
# --------------------------------------------------------------------------

STAT_HOURS = list(range(5, 26))       # 05:00 .. 25:00, the service day
NSH = len(STAT_HOURS)


def _hour_slot(secs):
    return min(NSH - 1, max(0, secs // 3600 - STAT_HOURS[0]))


def write_stats(patterns, boardings, alightings, complexes, coord, code_cx,
                line_meta, stamp):
    """Aggregate the same routing into per-station and per-segment totals.

    The animation answers "where is everyone right now"; this answers "how much
    moves through here over the day", which is the question you cannot get at by
    watching dots. Same numbers, summed instead of sampled -- so it costs one
    more pass, not another model.
    """
    print("aggregating the day for the static view ...")
    nb = len(complexes)
    board = collections.defaultdict(lambda: [0.0] * NSH)   # (ci, line) -> hrs
    alight = collections.defaultdict(lambda: [0.0] * NSH)
    seg = collections.defaultdict(lambda: [0.0] * NSH)     # (line, a, b) -> hrs

    for (pi, ti), bd in boardings.items():
        p = patterns[pi]
        line = p["line"]
        al = alightings.get((pi, ti), {})
        vdep = p["vdep"][ti]
        stops = p["stops"]
        onboard = 0.0
        for si, code in enumerate(stops):
            h = _hour_slot(int(vdep[si]))
            b, a = bd.get(si, 0.0), al.get(si, 0.0)
            ci = code_cx.get(code)
            if ci is not None:
                if b:
                    board[(ci, line)][h] += b
                if a:
                    alight[(ci, line)][h] += a
            onboard = max(0.0, onboard + b - a)
            if onboard > 0 and si + 1 < len(stops):
                seg[(line, code, stops[si + 1])][h] += onboard

    st_out = []
    by_station = collections.defaultdict(dict)
    for (ci, line), hrs in board.items():
        by_station[ci].setdefault(line, {})["b"] = [round(v, 1) for v in hrs]
    for (ci, line), hrs in alight.items():
        by_station[ci].setdefault(line, {})["a"] = [round(v, 1) for v in hrs]
    for ci, c in enumerate(complexes):
        rows = by_station.get(ci)
        if not rows:
            continue
        st_out.append({"name": c["name"], "name_en": c.get("name_en", ""),
                       "lat": c["lat"], "lon": c["lon"],
                       "lines": sorted(set(p["line"] for p in c["platforms"])),
                       "by_line": rows})

    seg_out = []
    for (line, a, b), hrs in seg.items():
        ca, cb = coord.get(a), coord.get(b)
        if not ca or not cb or max(hrs) < 0.5:
            continue
        ia, ib = code_cx.get(a), code_cx.get(b)
        seg_out.append({
            "line": line,
            "a": complexes[ia]["name"] if ia is not None else "",
            "b": complexes[ib]["name"] if ib is not None else "",
            "ae": complexes[ia].get("name_en", "") if ia is not None else "",
            "be": complexes[ib].get("name_en", "") if ib is not None else "",
            "p": [round(ca[0], 5), round(ca[1], 5),
                  round(cb[0], 5), round(cb[1], 5)],
            "h": [round(v, 1) for v in hrs],
        })

    out = {"date": OD_DATE, "day": DAY_NAME, "build": stamp, "hours": STAT_HOURS,
           "line_meta": line_meta, "stations": st_out, "segments": seg_out}
    path = out_paths(stamp["sample"])[1]
    write_json(path, out)
    print("   %d stations, %d segments -> %s (%.1f MB)"
          % (len(st_out), len(seg_out), path, os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
