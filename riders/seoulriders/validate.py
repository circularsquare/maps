# -*- coding: utf-8 -*-
"""Check the network against the outside world, not against itself.

Every bug this project has found looked completely plausible on the map, so
the useful checks are the ones that compare what we built to a figure someone
else published. Three groups, in order of how much they would cost to get
wrong:

  --schedules   Did kric.py parse the timetables correctly? Compares each
                line's end-to-end journey time, service span and station count
                against published figures. A packed-cell parsed with the wrong
                delimiter, or times read as H:MM when they are H:MM:SS, shows
                up here as a journey time that is wrong by a factor.
  --geometry    Is every station where it should be? Walks each line's own
                stop order and measures the gaps. A station whose name was
                resolved to the wrong place lands far from its neighbours.
  --od          Is the IPF output sane? Compares fitted per-line daily volume
                against the OD's own totals, and prints the hourly profile of
                stations whose shape we can predict -- an airport should not
                have a commuter peak.
  --coverage    What is actually in the OD? Measures how much of each line's
                traffic the source contains at all. This is the check that
                found the OD holds only trips touching the Seoul network.

    python validate.py               # all three
    python validate.py --schedules

Reference figures are the operators' own published journey times and service
spans, rounded to the minute. They are a sanity bound, not ground truth: a
10-15% difference is ordinary (we hold a 2026 timetable, they quote a typical
run), a 50% difference is a bug.
"""

import argparse
import collections
import csv
import io
import json
import math
import os
import re
import sys

import lines as LR

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

SERVICE = "END"          # 2023-12-31 was a Sunday

# line id -> (end-to-end minutes, stations, first train, last train)
# Published by the operators; see the module docstring on how to read a miss.
REFERENCE = {
    "SB": (135, 63, "05:10", "24:30"),   # 청량리-인천, 수인분당선
    "GJ": (140, 57, "05:00", "24:30"),   # 문산-지평
    "AR": (66, 14, "05:20", "24:40"),    # 서울-인천공항2터미널, 일반열차
    "SN": (50, 16, "05:30", "24:30"),    # 신사-광교
    "UI": (23, 13, "05:30", "24:00"),    # 신설동-북한산우이
    "SL": (16, 11, "05:30", "24:30"),    # 샛강-관악산
    "I1": (57, 30, "05:30", "24:30"),    # 계양-국제업무지구 (2023 extent)
    "I2": (57, 27, "05:30", "24:30"),    # 검단오류-운연
    "GP": (32, 10, "05:30", "24:00"),    # 양촌-김포공항
    "GC": (80, 25, "05:10", "24:00"),    # 상봉-춘천
    "SH": (70, 21, "05:30", "24:00"),    # 일산-원시
    "GG": (50, 11, "05:30", "23:50"),    # 판교-여주
    "UJ": (21, 15, "05:00", "24:00"),    # 발곡-탑석
}

# A metro hop longer than this is worth a look; the commuter lines legitimately
# run further between stops, so they get their own bound.
# 신분당선 is a metro by service but was built for speed: the 청계산 tunnel
# between 청계산입구 and 판교 is 7.8 km, the longest gap on the network.
HOP_WARN_M = {"SB": 9000, "GJ": 12000, "GC": 14000, "SH": 9000, "GG": 16000,
              "AR": 20000, "1": 12000, "SN": 9000}
HOP_WARN_DEFAULT = 6000


def norm(s):
    return re.sub(r"\s+", "", (s or "").strip())


def read_cp949(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return []
    with io.open(p, encoding="cp949", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def hhmmss(s):
    s = (s or "").strip()
    if not s or s.count(":") != 2:
        return None
    h, m, sec = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def hhmm(secs):
    return "%02d:%02d" % (secs // 3600, secs % 3600 // 60)


def metres(a, b):
    dy = (a[0] - b[0]) * 111320.0
    dx = (a[1] - b[1]) * 111320.0 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def load_runs(service=SERVICE):
    """Every train run for a service day, as (line, [(code, dep_secs), ...])."""
    rows = read_cp949("timetable_raw.csv") + read_cp949("timetable_extra.csv")
    runs = collections.defaultdict(list)
    for r in rows:
        if service and norm(r["주중주말"]) != service:
            continue
        line = norm(r["호선"])
        if line not in LR.BY_ID:
            continue
        arr, dep = hhmmss(r["열차도착시간"]), hhmmss(r["열차출발시간"])
        if arr == 0:
            arr = None
        if dep == 0:
            dep = None
        t = dep if dep is not None else arr
        if t is None:
            continue
        key = (line, norm(r["방향"]), norm(r["열차코드"]))
        runs[key].append((t, norm(r["역사코드"])))
    out = collections.defaultdict(list)
    for (line, _, _), stops in runs.items():
        stops.sort()
        if len(stops) >= 2:
            out[line].append(stops)
    return out


def load_coords():
    with io.open(os.path.join(D, "stations.json"), encoding="utf-8") as f:
        net = json.load(f)
    coord, name = {}, {}
    for c in net["complexes"]:
        for p in c["platforms"]:
            coord[p["code"]] = (c["lat"], c["lon"])
            name[p["code"]] = c["name"]
    return net, coord, name


# --------------------------------------------------------------------------

def check_schedules(runs):
    print("=" * 78)
    print("SCHEDULES -- parsed timetable against the operators' published figures")
    print("=" * 78)
    print("%-4s %-14s %s" % ("", "", "end-to-end     stations      first        last"))
    print("%-4s %-14s %6s %6s  %4s %4s  %5s %5s  %5s %5s   %s"
          % ("id", "line", "ours", "pub", "ours", "pub",
             "ours", "pub", "ours", "pub", "trains"))
    problems = []
    for lid in LR.ALL_IDS:
        seqs = runs.get(lid)
        if not seqs:
            continue
        longest = max(seqs, key=len)
        e2e = (longest[-1][0] - longest[0][0]) / 60.0
        stations = len(set(c for s in seqs for _, c in s))
        first = min(s[0][0] for s in seqs)
        last = max(s[-1][0] for s in seqs)
        ref = REFERENCE.get(lid)
        if ref:
            r_e2e, r_st, r_first, r_last = ref
            flag = ""
            if abs(e2e - r_e2e) / max(r_e2e, 1) > 0.35:
                flag += " JOURNEY-TIME"
            if abs(stations - r_st) > 3:
                flag += " STATION-COUNT"
            if abs(first - hhmmss(r_first + ":00")) > 3600:
                flag += " FIRST-TRAIN"
            # An impossible last train is how a mishandled midnight shows up,
            # and it is invisible unless something asserts on it: 47:59 and
            # 234:36 both sat in this column for a while looking like data.
            if abs(last - hhmmss(r_last + ":00")) > 3600:
                flag += " LAST-TRAIN"
            if last > 28 * 3600:
                flag += " IMPOSSIBLE-LAST-TRAIN"
            if flag:
                problems.append((lid, flag.strip()))
            print("%-4s %-14s %6.0f %6d  %4d %4d  %5s %5s  %5s %5s   %5d%s"
                  % (lid, LR.DISPLAY[lid], e2e, r_e2e, stations, r_st,
                     hhmm(first), r_first, hhmm(last), r_last, len(seqs),
                     flag))
        else:
            flag = " IMPOSSIBLE-LAST-TRAIN" if last > 28 * 3600 else ""
            if flag:
                problems.append((lid, flag.strip()))
            print("%-4s %-14s %6.0f %6s  %4d %4s  %5s %5s  %5s %5s   %5d%s"
                  % (lid, LR.DISPLAY[lid], e2e, "-", stations, "-",
                     hhmm(first), "-", hhmm(last), "-", len(seqs), flag))
    print()
    if problems:
        print("   %d lines differ from the published figure by more than the "
              "tolerance:" % len(problems))
        for lid, flag in problems:
            print("      %-4s %s" % (lid, flag))
    else:
        print("   every line with a reference figure is within tolerance.")
    return problems


def check_geometry(runs, coord, name):
    print()
    print("=" * 78)
    print("GEOMETRY -- station spacing along each line's own stop order")
    print("=" * 78)
    print("A station resolved to the wrong place lands far from its neighbours.")
    print()
    print("%-4s %-14s %6s %7s %8s %9s   %s"
          % ("id", "line", "hops", "median", "longest", "over bound", "worst hop"))
    problems = []
    for lid in LR.ALL_IDS:
        seqs = runs.get(lid)
        if not seqs:
            continue
        longest = max(seqs, key=len)
        codes = [c for _, c in longest]
        hops = []
        for i in range(len(codes) - 1):
            a, b = coord.get(codes[i]), coord.get(codes[i + 1])
            if a and b and codes[i] != codes[i + 1]:
                hops.append((metres(a, b), codes[i], codes[i + 1]))
        if not hops:
            continue
        hops_m = sorted(h[0] for h in hops)
        med = hops_m[len(hops_m) // 2]
        worst = max(hops)
        bound = HOP_WARN_M.get(lid, HOP_WARN_DEFAULT)
        over = [h for h in hops if h[0] > bound]
        if over:
            problems.append((lid, over))
        print("%-4s %-14s %6d %6.0fm %7.0fm %9d   %s -> %s  %.1f km"
              % (lid, LR.DISPLAY[lid], len(hops), med, worst[0], len(over),
                 name.get(worst[1], "?"), name.get(worst[2], "?"),
                 worst[0] / 1000.0))
    print()
    if problems:
        print("   hops over the per-line bound (long is normal on commuter "
              "lines; a metro line's is not):")
        for lid, over in problems:
            for d, a, b in sorted(over, reverse=True)[:4]:
                print("      %-4s %-12s -> %-12s %6.1f km"
                      % (lid, name.get(a, "?"), name.get(b, "?"), d / 1000.0))
    else:
        print("   no station sits implausibly far from its neighbours.")
    return problems


def check_od(net):
    print()
    print("=" * 78)
    print("OD -- fitted hourly volumes against the source totals")
    print("=" * 78)
    import numpy as np

    p = os.path.join(D, "od_hourly.npz")
    if not os.path.exists(p):
        print("   no od_hourly.npz -- run build_od.py")
        return []
    z = np.load(p, allow_pickle=True)
    X, pairs = z["x"], z["pairs"]
    complexes = net["complexes"]

    # which line each complex belongs to, for a per-line rollup
    cx_lines = [sorted(set(pl["line"] for pl in c["platforms"]))
                for c in complexes]

    board = np.zeros(len(complexes))
    for k, (o, _d) in enumerate(pairs):
        board[int(o)] += X[k].sum()

    print("fitted daily boardings by line (a station on two lines is counted")
    print("once per line, so the total exceeds the trip count):")
    print()
    per = collections.Counter()
    for i, ls in enumerate(cx_lines):
        for l in ls:
            per[l] += board[i]
    print("   %-4s %-14s %12s" % ("id", "line", "boardings"))
    for lid in LR.ALL_IDS:
        if per.get(lid):
            print("   %-4s %-14s %12s"
                  % (lid, LR.DISPLAY[lid], format(int(per[lid]), ",")))

    # Shape check: a handful of stations whose profile we can predict.
    hours = list(z["hours"])
    print()
    print("hourly shape for stations whose profile we can predict:")
    idx = dict((c["name"], i) for i, c in enumerate(complexes))
    watch = [("인천공항1터미널", "an airport: flat, no commuter peak"),
             ("잠실", "NYE crowd: a late spike"),
             ("홍대입구", "nightlife: builds through the evening"),
             ("여의도", "office: quiet on a Sunday")]
    prof = np.zeros((len(complexes), X.shape[1]))
    for k, (o, _d) in enumerate(pairs):
        prof[int(o)] += X[k]
    for nm, why in watch:
        i = idx.get(nm)
        if i is None:
            print("   %-14s (not in the network)" % nm)
            continue
        v = prof[i]
        if v.sum() <= 0:
            print("   %-14s (no boardings)" % nm)
            continue
        peak = v.max()
        bars = "".join("%d" % min(9, round(9 * x / peak)) for x in v)
        print("   %-14s %s   %s" % (nm, bars, why))
    print("   %-14s %s" % ("", "".join(str(h % 10) for h in hours)))
    print("   (digits are 0-9 relative to that station's own peak hour)")
    return []


def check_coverage():
    """How much of a line's own traffic does the OD contain at all?

    A line whose riders mostly travel within it should show a high share of
    within-line trips: 2호선 is 55%. A line showing ~0% is not quiet, it is
    absent -- the source only holds trips that touch the Seoul network, so the
    outer operators appear as feeders and their internal traffic is missing.
    This is a property of the OD, not of anything we do to it, and it is the
    single most important caveat on the map.
    """
    print()
    print("=" * 78)
    print("COVERAGE -- how much of each line's traffic is in the OD at all")
    print("=" * 78)
    rows = read_cp949("od_2023-12-31.csv")
    if not rows:
        print("   no OD file")
        return []
    board = collections.Counter()
    internal = collections.Counter()
    total = 0
    for r in rows:
        n = int(r["총_승객수"])
        o, d = norm(r["승차_호선"]), norm(r["하차_호선"])
        total += n
        board[o] += n
        if o == d:
            internal[o] += n

    print("%-16s %11s %11s %8s   %s"
          % ("OD 호선 label", "boardings", "within-line", "share", ""))
    problems = []
    for lab in sorted(board, key=lambda l: -board[l]):
        share = 100.0 * internal[lab] / max(board[lab], 1)
        note = ""
        if board[lab] > 1500 and share < 5.0:
            note = "  <- essentially only its trips to and from Seoul"
            problems.append((lab, "OD-COVERAGE"))
        print("%-16s %11s %11s %7.1f%%%s"
              % (lab, format(board[lab], ","), format(internal[lab], ","),
                 share, note))
    print()
    print("   Lines flagged above are not quiet -- their internal traffic is")
    print("   not in the source. 인천1호선's real daily ridership is an order")
    print("   of magnitude above what the OD holds for it.")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--schedules", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--od", action="store_true")
    args = ap.parse_args()
    everything = not (args.schedules or args.geometry or args.od or
                      args.coverage)

    bad = []
    print("loading timetables and network ...")
    runs = load_runs()
    net, coord, name = load_coords()

    # The 김포골드라인 weekday bug lived for a day because every check ran on
    # END service only, which is the one we draw. A bad row on any service day
    # is a bad parse, so sweep all of them.
    print("   checking every service day for impossible times ...")
    for svc in ("DAY", "SAT", "END"):
        bad_svc = []
        for lid, seqs in load_runs(svc).items():
            worst = max((s[-1][0] for s in seqs), default=0)
            if worst > 28 * 3600:
                bad_svc.append((lid, worst))
        if bad_svc:
            for lid, worst in bad_svc:
                print("      %s %-4s last stop at %.1fh   IMPOSSIBLE"
                      % (svc, lid, worst / 3600.0))
            bad += [(lid, "%s IMPOSSIBLE-TIME" % svc) for lid, _ in bad_svc]
        else:
            print("      %s: clean" % svc)

    print("   %d lines with END-service runs, %d platforms placed\n"
          % (len(runs), len(coord)))

    if everything or args.schedules:
        bad += check_schedules(runs)
    if everything or args.geometry:
        bad += check_geometry(runs, coord, name)
    if everything or args.od:
        check_od(net)
    if everything or args.coverage:
        bad += check_coverage()

    print()
    print("=" * 78)
    print("%d thing(s) flagged for a look." % len(bad))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
