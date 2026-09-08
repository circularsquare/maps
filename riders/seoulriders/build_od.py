# -*- coding: utf-8 -*-
"""Re-level the measured OD onto a chosen day, split it into hours, and write
data/od_hourly.npz.

Seoul publishes station-to-station pairs for exactly one date, 2023-12-31 --
and that date is a Sunday. It is not, as it first appeared, distorted by being
New Year's Eve: measured against the other 51 Sundays of 2023 its hourly
boarding profile is indistinguishable, peaking at 16-17 like 48 of them, and
its total sits 5% above the Sunday mean. The only genuinely New Year thing
about it is the post-midnight bin. **Sunday is the problem, not New Year.** A
Sunday has one broad afternoon hump; a Wednesday has 8% of its boardings in
07-08 and 12% in 18-19. Nothing that reweights the hours of a Sunday can
produce that, because the trips are not there to move.

So `--day weekday` does two fits, not one.

**First, re-level the pairs (`furness`).** Take the measured pair totals as a
seed and scale them so each station's daily origin and destination totals match
what that station actually did on a typical weekday -- measured, from
`card_daily_*.csv`, which covers all 27 lines. Classic doubly-constrained
Furness: `od[o,d] * a[o] * b[d]`. The seed supplies the interaction structure
(who plausibly travels to whom, and how far people will go), the marginals
supply the volumes. This is the modelled step and the honest limit of the
result -- see "Getting off New Year's Eve" in README.md.

**Then split into hours (`ipf`), as before.** The hourly file gives boardings
and alightings per station per hour, for every day of 2023 and 2024, so the
weekday shape is measured too. Fit over three constraint families:

  (a) pair total   sum_h  X[o,d,h]      = OD[o,d]        every pair
  (b) origin-hour  sum_d  X[o,d,h]      = B[o,h]         measured origins
  (c) dest-hour    sum_o  X[o,d,h']     = A[d,h']        measured destinations

where h' is the hour the rider *arrives*, h + traveltime(o,d). That shift is
what makes this more than a textbook IPF: the destination constraint sits in a
different time index from the origin constraint, so the fit couples adjacent
hours and the travel-time matrix does real work. Without it the evening comes
out as the morning at lower volume rather than genuinely reversed.

Stations with no measured hourly counts take part in (a) only. Their hours are
inferred through their measured partners: a trip into 강남 still has to arrive
in an hour consistent with 강남's measured alighting profile. That is most of
the network now -- the hourly file is 서울교통공사's own, covering lines 1-8
within their operating boundary, so all of line 9, the Korail through-running
sections and every line kric.py added are fitted this way.

A station is found by **name and 호선 together**, never by name alone. Seoul
reuses names: 양평 on line 5 and 양평 on 경의중앙선 are 27 km apart, and before
both lines were carried the ambiguity could not arise. build_stations.py records
which OD 호선 labels belong to each complex; this file uses them.

    python build_od.py                  # a typical weekday
    python build_od.py --day sunday     # a typical Sunday
    python build_od.py --day nye        # 2023-12-31 exactly, nothing re-levelled
    python build_od.py --month 202411   # a second opinion on the reference days
"""

import argparse
import collections
import csv
import datetime
import io
import json
import os
import re
import sys

import numpy as np

import airport
import daytype
import outside

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

STATIONS = os.path.join(D, "stations.json")
OD_CSV = os.path.join(D, "od_2023-12-31.csv")
OUT = os.path.join(D, "od_hourly.npz")

# Daily × hourly gate counts, 서울교통공사 lines 1-8 only, one file per year.
HOURLY_CSVS = [os.path.join(D, "hourly_2023_raw.csv"),
               os.path.join(D, "daily_hourly_raw.csv")]

# Daily gate counts, no hours, but all 27 lines. fetch_ridership.py pulls one
# file per month; the OD month is needed as well as the reference month
# because the re-levelling is a per-station ratio between the two.
def card_csv(month):
    return os.path.join(D, "card_daily_%s.csv" % month)


OD_DATE = "2023-12-31"
OD_MONTH = "202312"

# Service day. The hourly file bins everything before 06:00 into one column and
# everything after midnight into another, so we run 05:00 to 01:00 next day.
HOURS = list(range(5, 25))
NH = len(HOURS)
H_INDEX = dict((h, i) for i, h in enumerate(HOURS))

HOUR_COLS = (
    [("06시이전", 5)]
    + [("%02d-%02d시간대" % (h, h + 1), h) for h in range(6, 24)]
    + [("24시이후", 24)]
)

TRANSFER_SEC = 180          # penalty for changing lines within a complex
DWELL_SEC = 30              # station dwell, added to each inter-station hop
IPF_ROUNDS = 40
FURNESS_ROUNDS = 60
EPS = 1e-9


# The card file names Korail's routes rather than the through-service the
# riders think they are on: 경부선/경인선/경원선/장항선 are all 1호선 track,
# 과천선/안산선 are 4호선, 일산선 is 3호선. stations.json keys on the OD's
# labels, so translate. Only 양평 is a name shared by two complexes, and this
# is what tells 5호선's apart from 경의중앙선's.
CARD_LINE_ALIAS = {
    "경부선": "1호선", "경인선": "1호선", "경원선": "1호선", "장항선": "1호선",
    "과천선": "4호선", "안산선": "4호선",
    "일산선": "3호선",
    "경의선": "경의중앙선", "중앙선": "경의중앙선",
    "9호선2~3단계": "9호선",
    "공항철도1호선": "공항철도1호선",
}


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
# travel-time matrix
# --------------------------------------------------------------------------

def median_hops(tt_rows, lines):
    """Median run time between consecutive stops, from the trains themselves."""
    runs = collections.defaultdict(list)
    for r in tt_rows:
        line = norm(r["호선"])
        if line not in lines:
            continue
        arr, dep = hhmmss(r["열차도착시간"]), hhmmss(r["열차출발시간"])
        # 00:00:00 is the file's null marker for a terminus that only arrives or
        # only departs, not a train at midnight; after-midnight times are
        # written 24:xx. Left as zero it scrambles the stop order.
        if arr == 0:
            arr = None
        if dep == 0:
            dep = None
        t = dep if dep is not None else arr
        if t is None:
            continue
        key = (line, norm(r["주중주말"]), norm(r["방향"]), norm(r["열차코드"]))
        runs[key].append((t, norm(r["역사코드"]), arr, dep))

    hops = collections.defaultdict(list)
    for key, stops in runs.items():
        stops.sort()
        for i in range(len(stops) - 1):
            _, ca, _, dep_a = stops[i]
            _, cb, arr_b, _ = stops[i + 1]
            if dep_a is None or arr_b is None:
                continue
            dt = arr_b - dep_a
            if 20 <= dt <= 1800:
                hops[(ca, cb)].append(dt)
                hops[(cb, ca)].append(dt)
    return dict((k, float(np.median(v))) for k, v in hops.items())


def travel_matrix(complexes, hops):
    """Dijkstra over platforms, collapsed to complex-to-complex seconds."""
    import heapq

    code_cx = {}
    cx_codes = []
    for i, c in enumerate(complexes):
        codes = [p["code"] for p in c["platforms"]]
        cx_codes.append(codes)
        for code in codes:
            code_cx[code] = i

    adj = collections.defaultdict(list)
    for (a, b), dt in hops.items():
        if a in code_cx and b in code_cx:
            adj[a].append((b, dt + DWELL_SEC))
    # transfers inside a complex
    for codes in cx_codes:
        for a in codes:
            for b in codes:
                if a != b:
                    adj[a].append((b, TRANSFER_SEC))

    n = len(complexes)
    TT = np.full((n, n), np.inf, dtype=np.float32)
    for i, codes in enumerate(cx_codes):
        dist = {}
        pq = [(0.0, c) for c in codes]
        heapq.heapify(pq)
        for c in codes:
            dist[c] = 0.0
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for v, w in adj.get(u, ()):
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        for code, d in dist.items():
            j = code_cx[code]
            if d < TT[i, j]:
                TT[i, j] = d
    np.fill_diagonal(TT, 0.0)
    return TT


# --------------------------------------------------------------------------
# daily marginals, all 27 lines
# --------------------------------------------------------------------------

def read_card(month):
    """CARD_SUBWAY_MONTH rows. UTF-8 with a BOM, unlike everything else here."""
    path = card_csv(month)
    if not os.path.exists(path):
        raise SystemExit(
            "missing %s.\nRun: python fetch_ridership.py %s"
            % (os.path.basename(path), month))
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def card_totals(rows, dates, find, n):
    """Mean daily boardings/alightings per complex over `dates`.

    `dates` are YYYYMMDD strings. Returns (B, A, have) where `have` marks the
    complexes that appeared at all -- 인천 1/2, 신분당, 김포골드, 서해, 의정부 and
    the 진접 branch are outside Seoul's card settlement and are simply absent.
    """
    B = np.zeros(n)
    A = np.zeros(n)
    have = np.zeros(n, dtype=bool)
    dates = set(dates)
    seen_dates = set()
    for r in rows:
        d = norm(r["사용일자"])
        if d not in dates:
            continue
        seen_dates.add(d)
        i = find(base(r["역명"]), norm(r["노선명"]))
        if i is None:
            continue
        B[i] += float(r["승차총승객수"] or 0)
        A[i] += float(r["하차총승객수"] or 0)
        have[i] = True
    if not seen_dates:
        raise SystemExit("none of %s are in the card file" % sorted(dates)[:4])
    B /= len(seen_dates)
    A /= len(seen_dates)
    return B, A, have, sorted(seen_dates)


def furness(od, po, pd, row_t, col_t, fit, rounds=FURNESS_ROUNDS):
    """Doubly-constrained scaling of a pair vector onto new station totals.

    `fit` marks the stations with a target; the rest float, taking whatever
    the other end of their trips implies. That is not a cop-out -- for a
    인천1호선 rider bound for 강남, the Seoul end is the one that knows how much
    busier a weekday is, and letting it do the whole job is better than
    inventing a factor for the Incheon end.
    """
    x = od.copy()
    for it in range(rounds):
        before = x.copy()
        for idx, target in ((po, row_t), (pd, col_t)):
            cur = np.zeros(len(row_t))
            np.add.at(cur, idx, x)
            f = np.ones(len(row_t))
            m = fit & (cur > EPS)
            f[m] = target[m] / cur[m]
            x *= f[idx]
        delta = np.abs(x - before).sum() / max(x.sum(), 1.0)
        if delta < 1e-7:
            return x, it, delta
    return x, rounds - 1, delta


def deterrence(pair_tot, TT, n, inside, width=180.0):
    """How readily a trip of a given length happens, measured off this network.

    For every travel-time bin, the measured trips in it over the origin x
    destination mass available to it -- so it is a rate, not a count, and the
    fact that there are more short pairs than long ones does not shape it. Bins
    are `width` seconds wide and the last one absorbs the tail.

    **Fitted over pairs with a 서울교통공사 gate at both ends, and nowhere else.**
    Fitting it over every pair looks more general and is badly wrong. The
    denominator is demand read off the measured OD, and for an outside station
    that demand is exactly the broken number this file exists to repair: at
    long travel times nearly every pair has an outside end, so the denominator
    comes out too small and the curve too high, while at short times the
    outside-outside pairs contribute demand and no trips and pull it down. The
    two biases together flatten the decay -- measured, it stopped falling past
    about 40 minutes and ran level to two hours, so a 100-minute journey scored
    as readily as a 50-minute one. Seeded with that, the invented trips came
    out at a 41-minute median against the measured 20, and the extra length
    turned into transfers: 1.82 legs a journey, and Incheon carrying 122% of
    its published 수송인원.

    Inside-inside pairs have a consistent numerator and denominator. They run
    out past about 70 minutes, which is what `_smooth_tail` is for.
    """
    b = np.zeros(n)
    a = np.zeros(n)
    for (o, d), v in pair_tot.items():
        if not (inside[o] and inside[d]):
            continue
        b[o] += v
        a[d] += v
    nb = int(np.nanmax(TT[np.isfinite(TT)]) // width) + 1
    num = np.zeros(nb)
    den = np.zeros(nb)
    fin = np.isfinite(TT)
    idx = np.zeros(TT.shape, dtype=np.int32)
    idx[fin] = np.minimum((TT[fin] / width).astype(np.int32), nb - 1)
    mass = np.outer(b, a)
    np.fill_diagonal(mass, 0.0)
    mass = np.where(np.outer(inside, inside), mass, 0.0)
    np.add.at(den, idx[fin], mass[fin])
    for (o, d), v in pair_tot.items():
        if inside[o] and inside[d]:
            num[idx[o, d]] += v
    f = np.where(den > 0, num / np.maximum(den, EPS), 0.0)
    return _smooth_tail(f, width), width


def _smooth_tail(f, width, min_share=1e-4):
    """Carry the fitted decay past where the measured pairs run out.

    Inside-inside pairs thin out past about 70 minutes and the raw bins there
    are noise -- a couple of long pairs with a handful of riders between them.
    Left as they are the curve wanders back up, which is what the seed then
    over-produces. So: fit a straight line to log f against log t where the
    bins are solid, use the fit beyond that, and require the whole thing to be
    non-increasing. A longer journey is never readier than a shorter one.
    """
    nb = len(f)
    t = (np.arange(nb) + 0.5) * width / 60.0
    solid = f > f.max() * min_share
    if solid.sum() >= 4:
        k = np.flatnonzero(solid)
        A = np.polyfit(np.log(t[k]), np.log(f[k]), 1)
        fit = np.exp(A[1] + A[0] * np.log(t))
        last = k[-1]
        f = np.where(np.arange(nb) <= last, np.where(solid, f, fit), fit)
    else:
        f = np.where(f > 0, f, f.max() if f.max() > 0 else 1.0)
    return np.minimum.accumulate(np.maximum(f, EPS))


# The card file names the operator of every station it settles, and that is
# what says whether the OD can see a station's own trips: 서울교통공사's lines
# are numbered, everyone else's are named -- Korail's route (경부선, 경인선,
# 일산선, 안산선 ...), 공항철도 1호선, 신림선, 우이신설선. 1호선 appears under
# both, because 서울역-청량리 is 서울교통공사's and the rest of the corridor is
# Korail's, which is exactly the distinction wanted.
SEOUL_METRO_CARD = re.compile("^[1-9]호선(2~3단계)?$")


def card_gates(card_rows, dates, find, n):
    """Gate counts at each complex, split by who settles the gate.

    Returns (seoul_b, seoul_a, other_b, other_a) as mean daily counts over
    `dates`. The OD holds a journey if either end passed a 서울교통공사 gate and
    drops it otherwise, and the file bears that out: over the complexes with a
    numbered-line row the OD carries 97.0% of their gate count -- the
    pipeline's own keep rate -- against 37.7% over the rest.

    **The split is per gate, not per complex, and that distinction is the
    whole point.** 강남 has 2호선 gates that 서울교통공사 settles and 신분당선
    gates that 네오트랜스 does, in one building. A 강남 to 양재 trip on 신분당
    touches neither 서울교통공사 gate and is missing from the OD exactly as a
    수원 to 안산 trip is. Treating 강남 as simply "inside" left it at 77.5% of
    its gate count while every pure-Korail station was fixed.
    """
    sb, sa, ob, oa = (np.zeros(n) for _ in range(4))
    dates = set(dates)
    seen = set()
    for r in card_rows:
        d = norm(r["사용일자"])
        if d not in dates:
            continue
        seen.add(d)
        raw = norm(r["노선명"])
        i = find(base(r["역명"]), raw)
        if i is None:
            continue
        b, a = float(r["승차총승객수"] or 0), float(r["하차총승객수"] or 0)
        if SEOUL_METRO_CARD.match(raw):
            sb[i] += b
            sa[i] += a
        else:
            ob[i] += b
            oa[i] += a
    k = max(len(seen), 1)
    return sb / k, sa / k, ob / k, oa / k


def seed_outside(pair_tot, TT, n, gate_b, gate_a, kept_b, kept_a, keep_frac,
                 has_outside, keep_share=0.995, verbose=True):
    """Invent the pairs the OD structurally cannot contain, at the right level.

    The OD holds only journeys that touch 서울교통공사's network, so a trip with
    both ends anywhere else -- 계양 to 인천시청, 부천 to 부평, 수원 to 안산 -- is
    not in the file at all. Scaling what is there cannot fix that, because
    Furness multiplies and a cell that is zero stays zero: without this step,
    giving 인천1호선 its true weekday total would pile all of it onto the handful
    of Seoul-bound pairs that do exist and draw Incheon as a commuter funnel
    with no local life in it.

    So the block is built rather than scaled. Each outside station's *missing*
    boardings -- its measured gate count less what the OD already carries for
    it -- are distributed over the other outside stations by the deterrence
    function above, then balanced against the missing alightings so both ends
    agree. Volumes are measured on both ends; only the split between them is
    modelled, and the hourly constraints in the IPF below reshape it again
    wherever an hourly profile exists. This is the one invented thing here.

    **It is not only the 도시철도 operators.** The first version of this covered
    the lines with no card row at all -- 인천, 신분당, 김포골드, 의정부, 진접 --
    and left the Korail 광역 lines alone because they *are* in the card file.
    Being in the card file was never the point. 수원 held 14.6% of its gate
    count and 동인천 19.5%, for the same reason and with the same fix; Incheon's
    own 유입 came out at 27% against a measured 33%, because the riders
    transferring in at 부평 off 경인선 were among the missing.

    Pairs are pruned to the `keep_share` of seeded volume that matters. The
    full outside block is ~127,000 pairs and its tail is 수원 to 동두천 at two
    riders a day; carrying those to `build.py` costs routing time and changes
    nothing that can be seen.
    """
    f, width = deterrence(pair_tot, TT, n, ~has_outside)
    miss_b = np.where(has_outside,
                      np.maximum(gate_b * keep_frac - kept_b, 0.0), 0.0)
    miss_a = np.where(has_outside,
                      np.maximum(gate_a * keep_frac - kept_a, 0.0), 0.0)

    idx = np.flatnonzero(has_outside & ((miss_b > 0) | (miss_a > 0)))
    grid_o, grid_d = np.meshgrid(idx, idx, indexing="ij")
    po = grid_o.ravel()
    pd_ = grid_d.ravel()
    ok = (po != pd_) & np.isfinite(TT[po, pd_])
    po, pd_ = po[ok], pd_[ok]
    if not len(po):
        return 0.0, 0
    bin_i = np.minimum((TT[po, pd_] / width).astype(np.int32), len(f) - 1)
    seed = miss_b[po] * miss_a[pd_] * f[bin_i]
    if seed.sum() <= 0:
        return 0.0, 0

    # balance the block against both sets of missing marginals
    fit = (miss_b > 0) & (miss_a > 0)
    seed, rounds, delta = furness(seed, po, pd_, miss_b, miss_a, fit)

    order = np.argsort(-seed)
    cut = np.searchsorted(np.cumsum(seed[order]), keep_share * seed.sum()) + 1
    keep = order[:cut]
    added = {}
    for k in keep:
        key = (int(po[k]), int(pd_[k]))
        pair_tot[key] += float(seed[k])
        added[key] = added.get(key, 0.0) + float(seed[k])
    if verbose:
        print("   %d complexes with non-서울교통공사 gates, %s pairs in the block"
              % (has_outside.sum(), format(len(po), ",")))
        print("   block furness: %d rounds, last change %.2e"
              % (rounds + 1, delta))
        print("   kept %s pairs (%.1f%% of the volume), %s trips added"
              % (format(len(keep), ","), 100 * seed[keep].sum() / seed.sum(),
                 format(int(seed[keep].sum()), ",")))
    return added


def dates_in(month, dows, exclude=()):
    """Every YYYYMMDD in `month` whose weekday is in `dows`."""
    y, m = int(month[:4]), int(month[4:])
    out = []
    d = datetime.date(y, m, 1)
    while d.month == m:
        if d.weekday() in dows and d.isoformat() not in exclude:
            out.append(d.strftime("%Y%m%d"))
        d += datetime.timedelta(days=1)
    return out


# --------------------------------------------------------------------------

def main(day_name, ref_month):
    day = daytype.get(day_name)
    print("day: %s -- %s%s" % (day_name, day["label"],
                               "" if ref_month == daytype.REF_MONTH
                               else "  (reference month %s)" % ref_month))
    print("loading network ...")
    with io.open(STATIONS, encoding="utf-8") as f:
        net = json.load(f)
    complexes = net["complexes"]
    lines = set(net["lines"])
    n = len(complexes)

    # (name, 호선) -> complex, with a name-only fallback for the overwhelming
    # majority of names that belong to exactly one station.
    by_name_label = {}
    by_name = collections.defaultdict(list)
    for i, c in enumerate(complexes):
        by_name[c["name"]].append(i)
        for lbl in c.get("od_labels", ()):
            by_name_label[(c["name"], lbl)] = i

    def find(name, label):
        label = CARD_LINE_ALIAS.get(label, label)
        i = by_name_label.get((name, label))
        if i is not None:
            return i
        cands = by_name.get(name)
        return cands[0] if cands and len(cands) == 1 else None

    ambiguous = [k for k, v in by_name.items() if len(v) > 1]
    print("   %d complexes, %d platforms%s"
          % (n, sum(len(c["platforms"]) for c in complexes),
             (", %d names shared by more than one station (%s)"
              % (len(ambiguous), ",".join(ambiguous))) if ambiguous else ""))

    print("deriving inter-station run times from the timetables ...")
    tt_rows = read_cp949(os.path.join(D, "timetable_raw.csv"))
    extra = os.path.join(D, "timetable_extra.csv")
    if os.path.exists(extra):
        tt_rows += read_cp949(extra)
    hops = median_hops(tt_rows, lines)
    print("   %d directed hops" % len(hops))

    print("building travel-time matrix ...")
    TT = travel_matrix(complexes, hops)
    reach = np.isfinite(TT).sum()
    print("   %d of %d pairs reachable (%.1f%%)"
          % (reach, n * n, 100.0 * reach / (n * n)))
    finite = TT[np.isfinite(TT)]
    print("   median %.0f min, 95th pct %.0f min"
          % (np.median(finite) / 60.0, np.percentile(finite, 95) / 60.0))

    # ----------------------------------------------------------------------
    print("loading OD (%s) ..." % OD_DATE)
    od_rows = read_cp949(OD_CSV)

    # the OD names a complex by its 2023 name; stations.json keys on the same
    pair_tot = collections.Counter()
    dropped = 0
    kept = 0
    for r in od_rows:
        o = find(base(r["승차_역"]), norm(r["승차_호선"]))
        d = find(base(r["하차_역"]), norm(r["하차_호선"]))
        cnt = int(r["총_승객수"])
        if o is None or d is None:
            dropped += cnt
            continue
        if o == d:
            dropped += cnt          # same-complex round trips carry no journey
            continue
        if not np.isfinite(TT[o, d]):
            dropped += cnt
            continue
        pair_tot[(o, d)] += cnt
        kept += cnt
    print("   kept %s trips over %d pairs, dropped %s (%.1f%%)"
          % (format(kept, ","), len(pair_tot), format(dropped, ","),
             100.0 * dropped / (kept + dropped)))

    # ----------------------------------------------------------------------
    # the operators Seoul does not settle
    # ----------------------------------------------------------------------
    print("\nseeding the journeys the OD cannot hold ...")
    names = set(by_name)
    off_b0, off_a0, _ = outside.gate(OD_MONTH, OD_DATE, find, n, names)
    od_card = read_card(OD_MONTH)
    s_b, s_a, o_b, o_a = card_gates(
        od_card, [OD_DATE.replace("-", "")], find, n)
    # KRIC's operators are all outside 서울교통공사 by construction
    other_b, other_a = o_b + off_b0, o_a + off_a0
    gate0_b, gate0_a = s_b + other_b, s_a + other_a
    has_outside = (other_b > 0) | (other_a > 0)

    seed_b0 = np.zeros(n)
    seed_a0 = np.zeros(n)
    for (o, d), v in pair_tot.items():
        seed_b0[o] += v
        seed_a0[d] += v

    # What share of the OD's own trips the pipeline carries -- everything not
    # dropped above for having an end off the network, being a same-complex
    # round trip, or being unroutable. About 98%. Off-card stations are put on
    # the same footing rather than at their raw gate total.
    #
    # **Measured against the file, never against gate counts.** The obvious
    # version of this -- trips kept over the gate count at the stations we
    # settle -- comes out at 80%, and the missing fifth is not the pipeline
    # dropping anything. It is the source's own coverage hole on the Korail
    # lines, where a 수인선 complex holds 7.4% of its gate count because a trip
    # that stays on 수인선 never touches 서울교통공사 and so was never in the
    # file. Folding that into this ratio would shrink every off-card line by a
    # fifth on the strength of a different line's missing data -- 인천1호선
    # built at 163k against a measured 203k. It did, until this was caught.
    keep_frac = kept / float(kept + dropped)
    print("   the OD's trips reach the network at %.1f%%" % (100 * keep_frac))
    pure = ~has_outside
    print("   %d complexes are 서울교통공사 gates only; the OD carries %.1f%% of "
          "their gate count against %.1f%% where another operator has gates too"
          % (pure.sum(),
             100 * seed_b0[pure].sum() / max(gate0_b[pure].sum(), 1),
             100 * seed_b0[has_outside].sum()
             / max(gate0_b[has_outside].sum(), 1)))

    seeded_vol = seed_outside(pair_tot, TT, n, gate0_b, gate0_a,
                              seed_b0, seed_a0, keep_frac, has_outside)

    pairs = np.array(sorted(pair_tot), dtype=np.int32)
    od = np.array([pair_tot[tuple(p)] for p in pairs], dtype=np.float64)
    # What share of each pair was seeded rather than measured. Furness scales
    # both halves of a pair by the same factor, so this fraction survives it.
    modelled = np.array([seeded_vol.get(tuple(p), 0.0) for p in pairs],
                        dtype=np.float64)
    modelled = np.divide(modelled, np.maximum(od, EPS))
    po, pd = pairs[:, 0], pairs[:, 1]
    npairs = len(pairs)

    # arrival-hour offset per pair, in whole hours
    shift = np.rint(TT[po, pd] / 3600.0).astype(np.int32)
    print("   arrival-hour offset: %d pairs same hour, %d +1, %d +2 or more"
          % ((shift == 0).sum(), (shift == 1).sum(), (shift >= 2).sum()))

    kept_b = np.zeros(n)
    kept_a = np.zeros(n)
    np.add.at(kept_b, po, od)
    np.add.at(kept_a, pd, od)

    # ----------------------------------------------------------------------
    # re-level the pairs onto the chosen day
    # ----------------------------------------------------------------------
    if day["dows"] is None:
        ref_dates = [OD_DATE]
        print("\nno re-levelling: --day %s is the measured date itself"
              % day_name)
    else:
        print("\nre-levelling the pairs onto %s ..." % day["label"])
        od_rows_card = read_card(OD_MONTH)
        ref_rows_card = read_card(ref_month)

        gate_b0, gate_a0, have0, _ = card_totals(
            od_rows_card, [OD_DATE.replace("-", "")], find, n)
        want = dates_in(ref_month, day["dows"], daytype.EXCLUDE_DATES)
        gate_b1, gate_a1, have1, got = card_totals(ref_rows_card, want, find, n)
        print("   reference days: %d of %s (%s .. %s)"
              % (len(got), ref_month, got[0], got[-1]))

        # The off-card operators' own gates, added to the card's rather than
        # replacing them: 부평 counts 경인선 boardings in the card file and
        # 인천1호선 boardings in KRIC, and those are two arrays of gates in one
        # building. A complex can be in both, in one, or in neither.
        off_b1, off_a1, off_have1 = outside.gate(
            ref_month, day["dows"], find, n, names, verbose=True)
        G0_b = np.where(have0, gate_b0, 0.0) + off_b0
        G0_a = np.where(have0, gate_a0, 0.0) + off_a0
        G1_b = np.where(have1, gate_b1, 0.0) + off_b1
        G1_a = np.where(have1, gate_a1, 0.0) + off_a1

        # Per-station ratio between the reference day and the measured date,
        # applied to the trips we actually keep. Doing it as a ratio rather
        # than an absolute means the OD's own coverage -- which trips reach a
        # station we carry, and which are dropped -- carries over untouched.
        # The seeded off-card pairs entered at the measured date's level, so
        # the same ratio lifts them onto the chosen day with everything else.
        fit = (G0_b > 0) & (G0_a > 0) & (G1_b > 0) & (G1_a > 0)
        row_t = np.zeros(n)
        col_t = np.zeros(n)
        row_t[fit] = kept_b[fit] * (G1_b[fit] / G0_b[fit])
        col_t[fit] = kept_a[fit] * (G1_a[fit] / G0_a[fit])
        fit_o = fit & (kept_b > 0)
        fit_d = fit & (kept_a > 0)
        print("   %d of %d complexes have a gate total on both days (%d of "
              "them off-card)" % (fit.sum(), n, (fit & off_have1).sum()))
        print("   day/OD-date ratio over those: boardings x%.2f, alightings x%.2f"
              % (G1_b[fit].sum() / G0_b[fit].sum(),
                 G1_a[fit].sum() / G0_a[fit].sum()))

        od, rounds, delta = furness(
            od, po, pd, row_t, col_t, fit_o & fit_d)
        print("   furness: %d rounds, last change %.2e" % (rounds + 1, delta))

        cur_b = np.zeros(n)
        np.add.at(cur_b, po, od)
        err = np.abs(cur_b[fit_o] - row_t[fit_o]).sum() / max(row_t[fit_o].sum(), 1)
        print("   origin totals hit to %.3f%%; total trips %s -> %s"
              % (100.0 * err, format(int(kept), ","), format(int(od.sum()), ",")))

        kept_b = np.zeros(n)
        kept_a = np.zeros(n)
        np.add.at(kept_b, po, od)
        np.add.at(kept_a, pd, od)
        ref_dates = [d[:4] + "-" + d[4:6] + "-" + d[6:] for d in got]

    # ----------------------------------------------------------------------
    print("\nloading hourly counts for %d day(s) ..." % len(ref_dates))
    B = np.zeros((n, NH))
    A = np.zeros((n, NH))
    have_b = np.zeros(n, dtype=bool)
    have_a = np.zeros(n, dtype=bool)

    wanted = set(ref_dates)
    seen = set()
    for path in HOURLY_CSVS:
        if not os.path.exists(path):
            continue
        for r in read_cp949(path):
            d = norm(r["수송일자"])
            if d not in wanted:
                continue
            seen.add(d)
            i = find(base(r["역명"]), norm(r["호선"]))
            if i is None:
                continue
            kind = norm(r["승하차구분"])
            tgt, flag = (B, have_b) if kind == "승차" else (A, have_a)
            for col, hour in HOUR_COLS:
                v = (r.get(col) or "").strip().replace(",", "")
                if v:
                    tgt[i, H_INDEX[hour]] += float(v)
            flag[i] = True
    if not seen:
        raise SystemExit("none of %s are in the hourly files" % ref_dates[:3])
    if len(seen) < len(wanted):
        print("   only %d of %d dates present in the hourly files"
              % (len(seen), len(wanted)))
    # Only the shape is used below -- every station is rescaled to its kept
    # total -- so summing several days and not dividing would be harmless.
    # Divide anyway, so the printed numbers mean what they say.
    B /= len(seen)
    A /= len(seen)

    # 인천교통공사 publishes the same thing for its own 68 stations -- station x
    # hour, 승차 and 하차 -- monthly since 2015, on its 사전정보 공표목록 board.
    # See outside.py for the month-to-weekday correction that goes with it.
    # Until this went in, every Incheon station was fitted only through its
    # Seoul-bound partners, so its hours were Seoul's rather than its own.
    ict_month = OD_MONTH if day["dows"] is None else ref_month
    ict_dows = day.get("date", OD_DATE) if day["dows"] is None else day["dows"]
    HB, HA, ict = outside.hourly(ict_month, ict_dows, find, n, names,
                                 verbose=True)
    B[ict] = HB[ict]
    A[ict] = HA[ict]
    have_b |= ict
    have_a |= ict

    measured = have_b & have_a
    print("   measured complexes: %d of %d (%d of them from 인천교통공사)"
          % (measured.sum(), n, ict.sum()))

    # The hourly counts include trips to destinations we dropped, so rescale
    # each measured station's profile to the total we actually carry. Shape is
    # what we want from them; level comes from the OD.
    for arr, kept_v in ((B, kept_b), (A, kept_a)):
        s = arr.sum(axis=1)
        ok = s > 0
        arr[ok] *= (kept_v[ok] / s[ok])[:, None]

    # ----------------------------------------------------------------------
    # seed: measured origin profile where we have one, else the system profile
    # ----------------------------------------------------------------------
    sys_profile = B[measured].sum(axis=0)
    sys_profile = sys_profile / sys_profile.sum()

    # An airport is not a commuter station, and nothing in this fit would ever
    # discover that. Every 공항철도 complex is unmeasured, so a 서울역 ->
    # 인천공항 pair has an hourly constraint at neither end and whatever it is
    # seeded with survives untouched into the output. Seeded with sys_profile
    # it came out with an 08:00/18:00 commuter double peak on the 직통, which
    # is not a service any commuter can use -- it stops only at 서울역 and the
    # two terminals. See "The airport train has a commuter rush hour" in
    # README.md, and `airport.py` for what replaces it.
    air = airport.profiles(ict_month)
    air_set = set()
    from_air, to_air = None, {}
    if air is not None:
        board_at, reach = air
        air_set = set(by_name[nm][0] for nm in airport.AIRPORT_COMPLEXES
                      if by_name.get(nm))
    if air_set:
        # Both in origin-departure time, which is what `seed` is indexed by.
        # Leaving the airport reads straight off; going to it has to be pulled
        # back by the train ride, which is the same per-pair `shift` the
        # destination constraint uses.
        from_air = np.array([board_at[h % 24] for h in HOURS])
        from_air = from_air / from_air.sum()
        for sh in np.unique(shift):
            v = np.array([reach[(h + int(sh)) % 24] for h in HOURS])
            to_air[int(sh)] = v / v.sum()

    seed = np.empty((npairs, NH))
    n_air = 0
    for k in range(npairs):
        o, d = po[k], pd[k]
        if d in air_set:
            seed[k] = to_air[int(shift[k])]
            n_air += 1
        elif o in air_set:
            seed[k] = from_air
            n_air += 1
        elif measured[o] and B[o].sum() > 0:
            seed[k] = B[o] / B[o].sum()
        else:
            seed[k] = sys_profile
    if air_set:
        print("   %d pairs have an 인천공항 end, seeded from the airport's "
              "own hours rather than the network's" % n_air)
    X = seed * od[:, None]

    # index helpers for the marginal scalings
    arrive = (shift[:, None] + np.arange(NH)[None, :])
    np.clip(arrive, 0, NH - 1, out=arrive)          # after 01:00 stays in the last bin

    print("\nfitting (%d rounds) ..." % IPF_ROUNDS)
    for it in range(IPF_ROUNDS):
        before = X.copy()

        # (b) origin-hour
        cur = np.zeros((n, NH))
        np.add.at(cur, po, X)
        f = np.ones((n, NH))
        m = measured[:, None] & (cur > EPS)
        f[m] = B[m] / cur[m]
        X *= f[po]

        # (c) destination-hour, in arrival time
        cur = np.zeros((n, NH))
        np.add.at(cur, (pd[:, None], arrive), X)
        f = np.ones((n, NH))
        m = measured[:, None] & (cur > EPS)
        f[m] = A[m] / cur[m]
        X *= f[pd[:, None], arrive]

        # (a) pair totals -- applied last so it always holds exactly
        s = X.sum(axis=1)
        good = s > EPS
        X[good] *= (od[good] / s[good])[:, None]
        X[~good] = (od[~good] / NH)[:, None]

        delta = np.abs(X - before).sum() / max(od.sum(), 1.0)
        if it % 5 == 0 or it == IPF_ROUNDS - 1:
            print("   round %2d  mean abs change %.5f" % (it, delta))
        if delta < 1e-6:
            print("   converged at round %d" % it)
            break

    # ----------------------------------------------------------------------
    # how well did we do?
    # ----------------------------------------------------------------------
    fit_b = np.zeros((n, NH))
    np.add.at(fit_b, po, X)
    fit_a = np.zeros((n, NH))
    np.add.at(fit_a, (pd[:, None], arrive), X)

    def rel_err(fit, target, mask):
        f, t = fit[mask], target[mask]
        denom = t.sum()
        return np.abs(f - t).sum() / denom if denom else float("nan")

    print("\nfit quality on measured stations:")
    print("   boardings  mean abs error %.2f%% of volume"
          % (100.0 * rel_err(fit_b, B, measured)))
    print("   alightings mean abs error %.2f%% of volume"
          % (100.0 * rel_err(fit_a, A, measured)))
    print("   pair totals preserved to %.6f"
          % (np.abs(X.sum(axis=1) - od).sum() / od.sum()))

    prof = X.sum(axis=0)
    print("\nsystem profile by hour (thousands):")
    for i, h in enumerate(HOURS):
        bar = "#" * int(60 * prof[i] / prof.max())
        print("   %02d:00  %7.1f  %s" % (h % 24, prof[i] / 1000.0, bar))

    np.savez_compressed(
        OUT,
        pairs=pairs, hours=np.array(HOURS, dtype=np.int32),
        x=X.astype(np.float32), shift=shift,
        modelled=modelled.astype(np.float32),
        tt=TT[po, pd].astype(np.float32),
        names=np.array([c["name"] for c in complexes]),
        # build.py and validate.py take the day from here rather than from a
        # flag of their own, so the timetable cannot disagree with the riders.
        day=np.array(day_name),
        ref_dates=np.array(ref_dates),
    )
    print("\nwrote %s (%d pairs x %d hours, day=%s)"
          % (OUT, npairs, NH, day_name))
    print("Next: python build.py")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=daytype.DEFAULT,
                    choices=sorted(daytype.DAYS),
                    help="which day to draw (default %s)" % daytype.DEFAULT)
    ap.add_argument("--month", default=daytype.REF_MONTH,
                    help="YYYYMM the reference days come from; needs a "
                         "matching data/card_daily_<month>.csv and the same "
                         "year in the hourly files (default %s)"
                         % daytype.REF_MONTH)
    a = ap.parse_args()
    main(a.day, a.month)
