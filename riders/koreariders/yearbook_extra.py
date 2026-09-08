# -*- coding: utf-8 -*-
"""The yearbook sheets the rest of the pipeline never opened.

`lines.py` reads the passenger workbook's sheets 4, 5, 8 and 9-13 and the
facility workbook's 2 and 4. The passenger workbook has nineteen sheets, and
the zip holds two further sections -- `2.도시철도` and `3.광역철도` -- that
nothing in this project had ever looked at. Three of those sheets are evidence
this model was missing, and one that looks like evidence is not:

    passenger sheet 14   거리별 여객 수송실적 -- passengers AND 인거리 by
                         distance band and train type. The 인거리 here is a
                         plain published total, unlike the per-line 인거리 that
                         probe_ingeori.py rejected, so summing load x length
                         over our own output is a direct external check on the
                         *level* -- the axis nothing else tests.

    도시철도 sheet 10     통행거리별 여객 승차실적 -- boardings and 인거리 by
                         distance band, per city line. The mean trip length
                         falls straight out of it, which is the number the five
                         gravity models assume rather than measure.

    도시철도 sheet 13     연도별 최대 혼잡도 -- the busiest segment on each city
                         line, named, per year. A model that puts its peak
                         somewhere else is wrong outright.

    passenger sheet 15   노선간 여객환승 실적 -- 1,137 line-to-line transfer
                         counts. It looks like the junction steps solve.py
                         fits, and it is not: the network total is 6,290 a day
                         against steps of millions a year. It counts ticketed
                         platform transfers; a tau is a through train whose
                         passengers never get off. Read it as confirmation that
                         a junction's 승하차 is almost all local, not as a way
                         to size a step.

Sheet 3 (선별 여객수송 by train type) was also examined and is *not* usable as
a sharper 통과인원: 전라선 reads 3.27M against a published 7.71M line total and
a 7.88M rebuild, so it attributes journeys to lines by some rule that is not
"touched this line's metals". Left alone deliberately.

    python yearbook_extra.py            # print everything this module reads
"""

import functools
import io
import os
import sys

import lines as LN

URBAN = "2.도시철도/도시철도-3.수송실적_완.xlsx"

TOTAL_ROW = "합계"

# Distance-band upper bounds, in km, for the two band tables. The last band is
# open-ended in both.
INTERCITY_BANDS = [50.0, 80.0, 100.0, 200.0, 300.0, 400.0, None]
URBAN_BANDS = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, None]


# The yearbook is a static file, so every read here is pure. Callers ask for
# the same table several times -- build_small_cities.py once per city, check.py
# alongside its own resolve() -- and reopening a 30 MB zip each time is the
# whole cost of these checks.
@functools.lru_cache(maxsize=None)
def distance_bands():
    """거리별 여객 수송실적 -> (passengers, 인거리) by train type.

    The sheet stacks two blocks with the same row labels: 수송인원 first, then
    인거리. Both are read, and the 인거리 block is the one that matters -- it is
    published rather than inferred from band midpoints, so the total is exact.
    """
    wb = LN._book(LN.PASSENGER)
    ws = wb["14"]
    blocks = []
    for row in ws.iter_rows(values_only=True):
        label, vals = row[0], list(row[1:8])
        if not isinstance(label, str):
            continue
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
        blocks.append((label.replace(" ", ""), [float(v) for v in vals]))
    # Each block ends with its own 합계 row. Keep it aside as the published
    # total -- it is the authoritative figure and it also catches a train type
    # going missing from the type rows.
    totals = [v for k, v in blocks if k == TOTAL_ROW]
    blocks = [(k, v) for k, v in blocks if k != TOTAL_ROW]
    wb.close()
    # The two blocks repeat the same train types in the same order, but not the
    # same *set* -- the 인원 block carries a 기타 row the 인거리 block omits. So
    # split where the first label comes round again rather than at the halfway
    # point, and intersect afterwards.
    if not blocks:
        raise SystemExit("sheet 14: no numeric rows")
    first = blocks[0][0]
    cut = next((i for i in range(1, len(blocks)) if blocks[i][0] == first), None)
    if cut is None:
        raise SystemExit("sheet 14: only one block found, expected 인원 + 인거리")
    pax, pkm = dict(blocks[:cut]), dict(blocks[cut:])
    if len(totals) != 2:
        raise SystemExit("sheet 14: expected a 합계 row in each block, got %d"
                         % len(totals))
    return pax, pkm, (sum(totals[0]), sum(totals[1]))


def intercity_pkm():
    """Published 2022 intercity 인거리, and the journeys behind it.

    The sheet's own 합계 rows, not a sum of the type rows -- the 인원 block
    carries a 통근 row the 인거리 block omits, so adding the types up loses
    33.6M passenger-km that 합계 keeps.
    """
    _, _, totals = distance_bands()
    return totals


@functools.lru_cache(maxsize=None)
def urban_trip_lengths():
    """통행거리별 여객 승차실적 -> {(operator, line): (pax, pkm, bands)}.

    Units in the sheet are 천명/년 and 천인-km/년; both are scaled to units here
    so callers do not have to remember. Operators that report nothing -- Seoul
    Metro and Busan both write "공사에서 관리하지 않는 데이터임" -- are skipped,
    which is why Busan lines 1-4 still have no calibration.
    """
    wb = urban_book()
    ws = wb["10"]
    out, org, line, pend = {}, None, None, None
    for row in ws.iter_rows(values_only=True):
        cells = list(row[:13])
        if len(cells) < 13:
            cells += [None] * (13 - len(cells))
        o, ln, kind = cells[1], cells[2], cells[3]
        if o:
            org = str(o).split("\n")[0].strip()
        if ln:
            line = " ".join(str(ln).split())
        if not isinstance(kind, str):
            continue
        vals = [float(c) * 1000.0 if isinstance(c, (int, float)) else 0.0
                for c in cells[5:13]]
        kind = kind.strip()
        if kind == "승차인원":
            pend = (org, line, vals)
        elif kind == "연인거리" and pend:
            o2, l2, pax = pend
            pend = None
            if sum(pax) <= 0 or sum(vals) <= 0:
                continue
            out[(o2, l2)] = (sum(pax), sum(vals), (pax, vals))
    wb.close()
    return out


def urban_mean_trip(operator, line, max_km=None):
    """Published mean trip length in km for one city line, or None.

    `max_km` is the longest journey the network can actually carry. Some
    operators report passengers in distance bands longer than their own line --
    Gwangju puts 8.4 % of its 인거리 beyond 20.5 km on a 20.5 km railway, which
    is either integrated bus-and-rail journeys or a filing error, and either way
    it is not a trip on the metals this model draws. Bands whose *lower* bound
    is already past the network are dropped; a band straddling the end is kept,
    since most of it is real.

    Returns (mean_km, dropped_fraction_of_pkm).
    """
    rec = urban_trip_lengths().get((operator, line))
    if rec is None:
        return None
    pax_total, pkm_total, (pax, pkm) = rec
    if max_km is None:
        return pkm_total / pax_total, 0.0
    lo = 0.0
    keep_p = keep_k = drop_k = 0.0
    for hi, p, k in zip(URBAN_BANDS, pax, pkm):
        if lo >= max_km:
            drop_k += k
        else:
            keep_p += p
            keep_k += k
        lo = hi if hi is not None else float("inf")
    if keep_p <= 0:
        return None
    return keep_k / keep_p, drop_k / pkm_total if pkm_total else 0.0


@functools.lru_cache(maxsize=None)
def urban_peak_segments(year=2022):
    """연도별 최대 혼잡도 -> {(operator, line): (segment, congestion_pct)}.

    The busiest segment on each city line, named, with the half-hour it applies
    to folded into the same cell. Five of this project's city models have no
    check on their output at all; this is one.
    """
    wb = urban_book()
    ws = wb["13"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # Find the year's column off the header row.
    col = None
    for r in rows[:6]:
        for i, c in enumerate(r):
            if isinstance(c, (int, float)) and int(c) == year:
                col = i
        if col is not None:
            break
    if col is None:
        raise SystemExit("sheet 13: no column for %d" % year)

    out, org, line, seg = {}, None, None, None
    for r in rows:
        cells = list(r[:col + 1])
        if len(cells) <= col:
            continue
        o, ln, kind = cells[1], cells[3], cells[4]
        if o:
            org = str(o).split("\n")[0].split("※")[0].strip()
        if ln not in (None, ""):
            line = " ".join(str(ln).split())
        v = cells[col]
        if not isinstance(kind, str):
            continue
        if kind.startswith("구간"):
            seg = str(v).replace("\n", " ").strip() if v else None
        elif kind.strip() == "혼잡도" and seg and isinstance(v, (int, float)):
            out[(org, line)] = (seg, float(v))
            seg = None
    return out


# The 도시철도 and 광역철도 volumes are `.xlsb` from the 2023 bundle on, and
# openpyxl cannot open those at all -- it raises "File contains no valid
# workbook part", which reads as a corrupt download rather than a format change.
# So this module is pinned to 2022 rather than following `lines.YEARBOOK`. It
# was not, and after the map moved to 2023 on 2026-09-08 `check_cities.py` died
# on import for a fortnight of sessions before anyone ran it.
#
# Pinned to the file, not to `year()`: a later edition that goes back to xlsx
# should be adopted deliberately, by changing this line, not silently.
BUNDLE_2022 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "korail_yearbook_2022_excel.zip")


def urban_book():
    """The 도시철도 수송실적 workbook, read-only and without its stylesheet."""
    return LN._book(URBAN, BUNDLE_2022)


def main():
    pax, pkm, _ = distance_bands()
    print("passenger sheet 14 -- 거리별 여객 수송실적 (2022)\n")
    print("%-12s %14s %16s %9s" % ("type", "인원", "인거리", "mean km"))
    print("-" * 55)
    for k in pax:
        if k not in pkm:
            print("%-12s %14.0f %16s" % (k, sum(pax[k]), "(no 인거리 row)"))
            continue
        p, d = sum(pax[k]), sum(pkm[k])
        print("%-12s %14.0f %16.0f %9.1f" % (k, p, d, d / p))
    tp, td = intercity_pkm()
    print("-" * 55)
    print("%-12s %14.0f %16.0f %9.1f  (the sheet's own 합계)"
          % ("TOTAL", tp, td, td / tp))
    print("\n%.2f billion passenger-km, %.1fM journeys, mean trip %.0f km"
          % (td / 1e9, tp / 1e6, td / tp))

    print("\n\n도시철도 sheet 10 -- 통행거리별 여객 승차실적 (2022)\n")
    print("%-24s %-12s %13s %16s %9s"
          % ("operator", "line", "승차인원", "연인거리", "mean km"))
    print("-" * 80)
    for (o, l), (p, d, _) in urban_trip_lengths().items():
        print("%-24s %-12s %13.0f %16.0f %9.2f" % (o, l, p, d, d / p))

    print("\n\n도시철도 sheet 13 -- 최대 혼잡도 (2022)\n")
    print("%-24s %-12s %-34s %8s"
          % ("operator", "line", "busiest segment", "혼잡도"))
    print("-" * 82)
    for (o, l), (seg, pct) in urban_peak_segments().items():
        print("%-24s %-12s %-34s %7.1f%%" % (o, l, seg[:34], pct))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
