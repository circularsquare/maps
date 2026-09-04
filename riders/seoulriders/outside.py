# -*- coding: utf-8 -*-
"""The operators Seoul's card file does not settle, made ready for build_od.py.

`fetch_outside_seoul.py` writes three raw files; this turns them into the two
things the fit actually wants, on the same terms as the Seoul sources:

  gate(month, dows, find, n)   mean daily boardings/alightings per complex
  hourly(month, find, n)       station x hour shape, weekday-corrected

103 of the 626 complexes have no row in `card_daily_*.csv` at all, because
their fares settle outside Seoul. Until now they took whatever their Seoul-bound
trips implied, which for 인천1호선 meant about a sixth of its real traffic
pointed entirely at Seoul. See "What the OD actually contains" in README.md.

## Two conversions, and both are measured

**A month is not a day.** KRIC publishes monthly totals, so a typical weekday
has to be solved out of one. Per station, from `incheon_daily.csv`, take the
mean of each day of the week relative to the Tue-Wed-Thu mean; the month is
then `sum over days of (that day's factor)` times the weekday mean, which
inverts to give the weekday mean. Public holidays count as Sundays. Incheon's
own factors are used for Incheon; the rest of the off-card operators have no
daily file and fall back to the Seoul network's factors for the same month,
which is the one borrowed number here. 신분당선 commutes harder than the network
mean and is the place that borrow is most likely to cost something.

**A month's hours are not a weekday's hours.** 인천교통공사's 시간대별 file is a
monthly aggregate, so its peaks are flattened by the weekends inside it, where
서울교통공사's OA-12921 is per day. The correction is measured rather than
assumed: for the same month, build Seoul's hourly shape twice -- once over
Tue/Wed/Thu only, once over every day -- and the ratio between them is what a
month does to a weekday. Applied per hour to Incheon's shape, separately for
boardings and alightings.

Only the shape of an hourly profile is used downstream (`build_od.py` rescales
every station to its own kept total), so this correction moves the peak, not
the level.

## Names

KRIC and the hourly file disagree with `stations.json` in three ways, all
resolved here rather than in the fetcher, which keeps its sources verbatim:

- 용인경전철 and 신분당선 stations carry a trailing 역 in KRIC (`강남역`,
  `정자역`). Stripped only when the stripped form is a complex and the full one
  is not, so `서울역` survives.
- KRIC splits a transfer complex per line with a bracketed suffix
  (`고속터미널`, `고속터미널(7)`, `고속터미널(9)`). `base()` drops the bracket and
  the rows sum, which is what makes the totals agree with the card file.
- Three names are simply different, and one is doubled in the KRIC source:
  see `ALIAS`.

    python outside.py                # coverage and reconciliation
    python outside.py --month 202312
"""

import argparse
import collections
import csv
import datetime
import io
import os
import re
import sys

import numpy as np

import daytype

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

KRIC_CSV = os.path.join(D, "kric_station_monthly.csv")
ICT_HOURLY_CSV = os.path.join(D, "incheon_hourly.csv")
ICT_DAILY_CSV = os.path.join(D, "incheon_daily.csv")
CARD_CSV = os.path.join(D, "card_daily_%s.csv")
SEOUL_HOURLY = [os.path.join(D, "hourly_2023_raw.csv"),
                os.path.join(D, "daily_hourly_raw.csv")]

HOURS = list(range(5, 25))
NH = len(HOURS)
H_INDEX = dict((h, i) for i, h in enumerate(HOURS))

# Seoul's hourly file bins by range, Incheon's by start hour; they land on the
# same 20 bins. 06시이전 and 05시 are both "before 06:00".
SEOUL_HOUR_COLS = (
    [("06시이전", 5)]
    + [("%02d-%02d시간대" % (h, h + 1), h) for h in range(6, 24)]
    + [("24시이후", 24)]
)

# KRIC names an operator and a line; the OD names a through-service. 인천's
# 1/2/7호선 are the collision that matters -- without the operator they would
# read as Seoul's.
KRIC_LINE = {
    (u"인천교통공사", u"1호선"): u"인천1호선",
    (u"인천교통공사", u"2호선"): u"인천2호선",
    (u"인천교통공사", u"7호선"): u"7호선(인천)",
    (u"공항철도", u"공항철도"): u"공항철도1호선",
    (u"서울메트로9", u"9호선"): u"9호선",
    (u"의정부경전철", u"의정부경전철"): u"의정부선",
    (u"네오트랜스(주)", u"신분당선"): u"신분당선",
    (u"경기철도", u"신분당선"): u"신분당선",
    (u"새서울철도", u"신분당선"): u"신분당선(연장2)",
    (u"김포골드라인", u"김포골드"): u"김포골드라인",
    (u"남서울경전철", u"신림선"): u"신림선",
    (u"우이신설도시철도", u"우이신설"): u"우이신설선",
    (u"남양주도시공사", u"4호선"): u"진접선",
}

# 용인경전철 is in KRIC but not on this map -- stations.json carries no
# 에버라인선 complex -- so its 15 stations are dropped rather than warned about.
KRIC_SKIP = (u"용인경량전철",)

ICT_LINE = {u"1호선": u"인천1호선", u"2호선": u"인천2호선",
            u"7호선": u"7호선(인천)"}

ALIAS = {
    u"부평삼거리부평삼거리": u"부평삼거리",   # doubled in the KRIC source
    u"가정시장": u"가정중앙시장",            # the hourly file's short forms,
    u"아시아드": u"아시아드경기장",          # used in some months and not others
    u"서해구청": u"서구청",                 # renamed after 2023; the daily file
    u"7호선부평구청": u"부평구청",           # is the only source using the new
}                                          # name, and the only one that has to
                                           # tell the two 부평구청 apart

# Korean public holidays that fall on a Mon-Fri in the windows these files
# cover. A holiday behaves like a Sunday, and counting it as a weekday would
# drag the weekday mean down.
HOLIDAYS = set("""
2023-01-23 2023-01-24 2023-03-01 2023-05-01 2023-05-05 2023-05-29
2023-06-06 2023-08-15 2023-09-28 2023-09-29 2023-10-03 2023-10-09
2023-12-25 2024-02-09 2024-02-12 2024-03-01 2024-04-10 2024-05-01
2024-05-06 2024-05-15 2024-06-06 2024-08-15 2024-09-16 2024-09-17
2024-09-18 2024-10-01 2024-10-03 2024-10-09 2024-12-25 2025-01-01
2025-01-28 2025-01-29 2025-01-30 2025-03-03 2025-05-01 2025-05-05
2025-05-06 2025-06-03 2025-06-06 2025-08-15 2025-10-03 2025-10-06
2025-10-07 2025-10-08 2025-10-09 2025-12-25 2026-01-01 2026-02-16
2026-02-17 2026-02-18 2026-03-02 2026-05-01 2026-05-05 2026-05-25
2026-06-03 2026-06-08 2026-08-17 2026-09-24 2026-09-25 2026-10-05
2026-10-09 2026-12-25
""".split())

WEEKDAY_DOWS = daytype.TUE_WED_THU


def norm(s):
    return re.sub(r"\s+", "", (s or "").strip())


def base(s):
    return re.sub(r"\(.*?\)$", "", norm(s))


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit("missing %s.\nRun: python fetch_outside_seoul.py"
                         % os.path.basename(path))
    with io.open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_cp949(path):
    with io.open(path, encoding="cp949", newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def day_class(d):
    """'wd' (Tue-Thu), 'mon', 'fri', 'sat' or 'sun'; holidays count as Sunday."""
    if d.isoformat() in HOLIDAYS:
        return "sun"
    w = d.weekday()
    if w in WEEKDAY_DOWS:
        return "wd"
    return {0: "mon", 4: "fri", 5: "sat", 6: "sun"}[w]


def month_days(month):
    y, m = int(month[:4]), int(month[4:])
    d = datetime.date(y, m, 1)
    out = []
    while d.month == m:
        out.append(d)
        d += datetime.timedelta(days=1)
    return out


# --------------------------------------------------------------------------
# name resolution
# --------------------------------------------------------------------------

def resolver(find, complex_names):
    """(raw name, OD label) -> complex index, or None.

    `find` is build_od.py's, so a name that already works is untouched; this
    only adds the three ways these two sources spell things differently.
    """
    misses = collections.Counter()

    def resolve(raw, label):
        n = base(raw)
        for cand in (n, ALIAS.get(n), ALIAS.get(norm(raw))):
            if not cand:
                continue
            i = find(cand, label)
            if i is not None:
                return i
        # a trailing 역, but only where dropping it names a complex and
        # keeping it does not -- 서울역 must survive
        if n.endswith(u"역") and n not in complex_names:
            i = find(n[:-1], label)
            if i is not None:
                return i
        misses[(raw, label)] += 1
        return None

    resolve.misses = misses
    return resolve


# --------------------------------------------------------------------------
# day-of-week factors
# --------------------------------------------------------------------------

def incheon_dow_factors(resolve, n):
    """Per complex, mean of each day class over the Tue-Thu mean.

    From `incheon_daily.csv`, which is the current 12 months rather than 2023 --
    the levels are useless to us but the day-type ratios are stable and are all
    this is taken from. cp949, straight from data.go.kr.
    """
    rows = read_cp949(ICT_DAILY_CSV)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    for r in rows:
        d = datetime.date(*[int(x) for x in r[u"통행일자"].split("-")])
        label = ICT_LINE.get(norm(r[u"호선"]) + u"호선")
        if label is None:
            continue
        i = resolve(r[u"역명"], label)
        if i is None:
            continue
        seen.add(d)
        acc[i][day_class(d)].append((float(r[u"승차인원"] or 0),
                                     float(r[u"하차인원"] or 0)))
    fb = np.full((n, 5), np.nan)
    fa = np.full((n, 5), np.nan)
    classes = ["wd", "mon", "fri", "sat", "sun"]
    for i, per in acc.items():
        if not per.get("wd"):
            continue
        wb = np.mean([x[0] for x in per["wd"]])
        wa = np.mean([x[1] for x in per["wd"]])
        for k, cls in enumerate(classes):
            if per.get(cls) and wb > 0 and wa > 0:
                fb[i, k] = np.mean([x[0] for x in per[cls]]) / wb
                fa[i, k] = np.mean([x[1] for x in per[cls]]) / wa
    return fb, fa, sorted(seen)


def card_dow_factors(month):
    """The same factors from `card_daily_<month>.csv`, for the Seoul network.

    Used as the fallback for the off-card operators that have no daily file of
    their own: 신분당, 김포골드, 의정부, 우이신설, 신림, 진접.
    """
    path = CARD_CSV % month
    if not os.path.exists(path):
        raise SystemExit("missing %s" % os.path.basename(path))
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    per = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        d = r[u"사용일자"]
        d = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:]))
        cls = day_class(d)
        p = per[cls]
        p[0] += float(r[u"승차총승객수"] or 0)
        p[1] += float(r[u"하차총승객수"] or 0)
    counts = collections.Counter(day_class(d) for d in month_days(month))
    classes = ["wd", "mon", "fri", "sat", "sun"]
    fb = np.full(5, np.nan)
    fa = np.full(5, np.nan)
    if counts["wd"]:
        wb = per["wd"][0] / counts["wd"]
        wa = per["wd"][1] / counts["wd"]
        for k, cls in enumerate(classes):
            if counts[cls] and wb > 0:
                fb[k] = (per[cls][0] / counts[cls]) / wb
                fa[k] = (per[cls][1] / counts[cls]) / wa
    return fb, fa


# --------------------------------------------------------------------------
# which lines the card file already settles
# --------------------------------------------------------------------------

def offcard_labels(month=None):
    """OD labels with no row at all in `card_daily_<month>.csv`.

    Worked out from the file rather than listed, because getting it wrong is a
    silent double count. KRIC covers 13 operators and the card file already
    settles most of them: joining the two for 2023-11 gives a KRIC/card ratio
    of 0.999 on 9호선, 신림선 and 우이신설선 -- the same gates counted twice --
    while 인천1/2, 7호선(인천), 신분당, 김포골드, 의정부 and 진접 have no card row
    at all. Only the second group is taken from KRIC.

    A complex can be in both groups at once and that is not a conflict:
    부평 counts 864,819 card boardings a month through 경인선's gates and a
    further 190,941 through 인천1호선's, which are a different array of gates in
    the same building. Those add.
    """
    from build_od import CARD_LINE_ALIAS      # deferred; build_od imports us
    month = month or daytype.REF_MONTH
    path = CARD_CSV % month
    if not os.path.exists(path):
        raise SystemExit("missing %s" % os.path.basename(path))
    seen = set()
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            lbl = norm(r[u"노선명"])
            seen.add(CARD_LINE_ALIAS.get(lbl, lbl))
    return set(KRIC_LINE.values()) - seen


# --------------------------------------------------------------------------
# daily marginals
# --------------------------------------------------------------------------

def gate(month, dows, find, n, complex_names, verbose=False):
    """Mean daily boardings/alightings per complex for `dows` in `month`.

    Returns (B, A, have). `dows` is a tuple of weekday numbers as in daytype.py,
    or a single ISO date string for one named day.
    """
    resolve = resolver(find, complex_names)
    offcard = offcard_labels()
    fb_i, fa_i, _ = incheon_dow_factors(resolve, n)
    fb_s, fa_s = card_dow_factors(daytype.REF_MONTH)

    y, m = int(month[:4]), int(month[4:])
    days = month_days(month)
    counts = collections.Counter(day_class(d) for d in days)
    classes = ["wd", "mon", "fri", "sat", "sun"]

    if isinstance(dows, str):
        want = [day_class(datetime.date(*[int(x) for x in dows.split("-")]))]
    else:
        want = sorted(set(day_class(d) for d in days
                          if d.weekday() in dows
                          and d.isoformat() not in daytype.EXCLUDE_DATES))

    B = np.zeros(n)
    A = np.zeros(n)
    have = np.zeros(n, dtype=bool)

    for r in read_csv(KRIC_CSV):
        if r["operator"] in KRIC_SKIP:
            continue
        if int(r["year"]) != y or int(r["month"]) != m:
            continue
        label = KRIC_LINE.get((r["operator"], r["line"]))
        if label is None or label not in offcard:
            continue
        i = resolve(r["station"], label)
        if i is None:
            continue
        for arr, col, fi, fs in ((B, "boardings", fb_i, fb_s),
                                 (A, "alightings", fa_i, fa_s)):
            total = float(r[col] or 0)
            if total <= 0:
                continue
            f = fi[i] if not np.isnan(fi[i]).any() else fs
            # month = sum over days of (that day's factor) x weekday mean
            denom = sum(counts[c] * f[k] for k, c in enumerate(classes)
                        if counts[c] and not np.isnan(f[k]))
            if denom <= 0:
                continue
            wd_mean = total / denom
            arr[i] += wd_mean * np.mean(
                [f[classes.index(c)] for c in want])
        have[i] = True

    if verbose:
        miss = resolve.misses
        print("   %d complexes from KRIC %s, %d names unresolved%s"
              % (have.sum(), month, len(miss),
                 (": " + ", ".join("%s/%s" % k for k in
                                   list(miss)[:8])) if miss else ""))
    return B, A, have


# --------------------------------------------------------------------------
# hourly shape
# --------------------------------------------------------------------------

def seoul_month_to_weekday(month):
    """Per hour, what a whole month does to a weekday's shape, from Seoul.

    Returns (cb, ca), each a length-20 multiplier normalised to mean 1.
    """
    days = month_days(month)
    wd = set(d.strftime("%Y-%m-%d") for d in days
             if day_class(d) == "wd"
             and d.isoformat() not in daytype.EXCLUDE_DATES)
    alld = set(d.strftime("%Y-%m-%d") for d in days)
    tot = {"wd": [np.zeros(NH), np.zeros(NH)],
           "all": [np.zeros(NH), np.zeros(NH)]}
    for path in SEOUL_HOURLY:
        if not os.path.exists(path):
            continue
        for r in read_cp949(path):
            d = norm(r[u"수송일자"])
            if d not in alld:
                continue
            k = 0 if norm(r[u"승하차구분"]) == u"승차" else 1
            for col, hour in SEOUL_HOUR_COLS:
                v = (r.get(col) or "").strip().replace(",", "")
                if not v:
                    continue
                v = float(v)
                tot["all"][k][H_INDEX[hour]] += v
                if d in wd:
                    tot["wd"][k][H_INDEX[hour]] += v
    out = []
    for k in (0, 1):
        w = tot["wd"][k] / max(tot["wd"][k].sum(), 1.0)
        a = tot["all"][k] / max(tot["all"][k].sum(), 1.0)
        c = np.where(a > 0, w / np.maximum(a, 1e-12), 1.0)
        out.append(c)
    return out


def hourly(month, find, n, complex_names, correct=True, verbose=False):
    """Station x hour boardings and alightings for the Incheon operator.

    Level is the month's; only the shape is used downstream. With `correct`,
    the month's flattened peak is pulled back to a weekday's using Seoul.
    """
    resolve = resolver(find, complex_names)
    B = np.zeros((n, NH))
    A = np.zeros((n, NH))
    have_b = np.zeros(n, dtype=bool)
    have_a = np.zeros(n, dtype=bool)
    for r in read_csv(ICT_HOURLY_CSV):
        if norm(r["month"]) != month:
            continue
        label = ICT_LINE.get(norm(r["line"]))
        if label is None:
            continue
        i = resolve(r["station"], label)
        if i is None:
            continue
        h = int(r["hour"])
        if h not in H_INDEX:
            continue
        if norm(r["kind"]) == u"승차":
            B[i, H_INDEX[h]] += float(r["count"] or 0)
            have_b[i] = True
        else:
            A[i, H_INDEX[h]] += float(r["count"] or 0)
            have_a[i] = True
    if correct:
        cb, ca = seoul_month_to_weekday(month)
        B *= cb[None, :]
        A *= ca[None, :]
        if verbose:
            print("   month->weekday correction, boardings: "
                  + " ".join("%d:%.2f" % (h, cb[H_INDEX[h]])
                             for h in (7, 8, 9, 12, 18, 19)))
    if verbose:
        print("   %d complexes with an hourly profile" % (have_b & have_a).sum())
    return B, A, have_b & have_a



# --------------------------------------------------------------------------

def _check(month):
    import json
    with io.open(os.path.join(D, "stations.json"), encoding="utf-8") as f:
        net = json.load(f)
    complexes = net["complexes"]
    n = len(complexes)
    by_name_label = {}
    by_name = collections.defaultdict(list)
    for i, c in enumerate(complexes):
        by_name[c["name"]].append(i)
        for lbl in c.get("od_labels", ()):
            by_name_label[(c["name"], lbl)] = i

    def find(name, label):
        i = by_name_label.get((name, label))
        if i is not None:
            return i
        cands = by_name.get(name)
        return cands[0] if cands and len(cands) == 1 else None

    names = set(by_name)
    print("reference month %s, weekday = Tue/Wed/Thu\n" % month)
    off = offcard_labels()
    print("off-card labels (%d): %s\n" % (len(off), ", ".join(sorted(off))))

    print("daily marginals from KRIC:")
    B, A, have = gate(month, WEEKDAY_DOWS, find, n, names, verbose=True)
    print("   weekday boardings over those complexes: %s"
          % format(int(B.sum()), ","))

    B0, A0, have0 = gate("202312", "2023-12-31", find, n, names)
    print("   the OD date (2023-12-31, a Sunday): %s"
          % format(int(B0.sum()), ","))
    m = have & have0 & (B0 > 0)
    print("   weekday / that Sunday over %d complexes: x%.2f"
          % (m.sum(), B[m].sum() / B0[m].sum()))

    print("\nhourly shape from 인천교통공사:")
    HB, HA, hh = hourly(month, find, n, names, verbose=True)
    tot = HB[hh].sum(axis=0)
    tot = tot / tot.sum()
    print("   corrected boarding shape: "
          + " ".join("%02d:%4.1f%%" % (h, 100 * tot[H_INDEX[h]])
                     for h in (6, 7, 8, 9, 12, 17, 18, 19)))

    print("\nper line:")
    per = collections.defaultdict(lambda: [0.0, 0])
    for i, c in enumerate(complexes):
        if not have[i]:
            continue
        for lbl in c.get("od_labels", ()):
            per[lbl][0] += B[i] / max(len(c.get("od_labels", ())), 1)
            per[lbl][1] += 1
    for lbl in sorted(per, key=lambda k: -per[k][0]):
        v, cnt = per[lbl]
        print("   %-14s %3d complexes  %10s weekday boardings"
              % (lbl, cnt, format(int(v), ",")))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=daytype.REF_MONTH)
    a = ap.parse_args()
    _check(a.month)
