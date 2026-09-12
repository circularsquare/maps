# -*- coding: utf-8 -*-
"""인천국제공항공사's 시간대별통계 -> data/incheon_airport_hourly.csv.

Why this file exists: the airport train had a commuter rush hour it should not
have. Every 공항철도 complex is unmeasured, so `build_od.py` seeded its pairs
with `sys_profile` -- the summed hourly shape of 서울교통공사 lines 1-8 -- and
for 서울역 -> 인천공항 neither end is measured, so nothing in the IPF ever
reshaped it. The 직통 came out with an 08:00/18:00 double peak. See "The
airport train has a commuter rush hour" in README.md.

**The line-level fix does not work and was tried first.** KRIC's
`citytimepassList.jsp` gives 공항철도's own measured hourly shape for free, and
it correlates **0.987** with the lines 1-8 shape we were already using --
because at the operator level AREX *is* a commuter railway, the 일반 service
through 김포공항/계양/검암/청라 carrying about five times the 직통's riders. The
artefact is confined to the airport journeys themselves, which are invisible in
any line-level aggregate. So the shape has to come from the airport.

## What this downloads

`statisticCategoryOfTime.do` on airport.kr, one GET per month per passenger
class, no login and no key. The table is 여객 by hour, split 도착 / 출발, which
is exactly the two directions an airport rail trip can take.

**환승승객 are fetched separately and netted out downstream.** A transit
passenger never leaves the airport, so they are not rail demand at all, and at
인천 they are 10.6% of everyone -- large enough to bend the shape, because
transit banks do not sit at the same hours as origin-destination traffic.

## What it does not give

Hourly by day of week. `outside.py`'s month-to-weekday correction cannot be
borrowed here: it is measured off Seoul's commuter pattern, and applying a
commuter correction to airport traffic would put back the very thing this is
removing. The hourly shape of an airport is set by the airline schedule, which
is close to the same every day, so the month aggregate is used uncorrected.
That is an assumption, and it is the weakest one in this file.

    python fetch_airport.py                # 2023, all twelve months
    python fetch_airport.py --years 2023 2024
    python fetch_airport.py --force        # re-download
"""

import argparse
import csv
import io
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

OUT = os.path.join(D, "incheon_airport_hourly.csv")

BASE = "https://www.airport.kr"
STATS = BASE + "/fsmFsn/co_ko/statisticCategoryOfTime.do"
# The nav wraps every statistics page in this layout token. Without it the
# page still renders but the search form comes back empty.
LAYOUT = "636f5f6b6f40403635314040666e637431"
LANDING = BASE + "/co_ko/651/subview.do"

# "" is everybody; the transit class is fetched separately so it can be
# subtracted. The other two classes (유임/무임) are not needed -- they split
# the same people by whether they paid.
CLASSES = [("all", ""), ("transit", "TRANSIT_PSNGER")]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

ROW = re.compile(r"^(\d\d):\d\d\s*~")


def session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"})
    # The stats servlet wants the cookie the landing page sets.
    s.get(LANDING, timeout=60)
    return s


def fetch_month(s, year, month):
    """(class, hour) -> (arrivals, departures), passengers."""
    out = {}
    for label, pas in CLASSES:
        params = {"layout": LAYOUT, "firstYn": "", "nvgSe": "", "arplnSe": "",
                  "routeSe": "", "bag": "", "terminalId": "", "pas": pas,
                  "stYear": str(year), "stMonth": "%02d" % month,
                  "edYear": str(year), "edMonth": "%02d" % month}
        r = s.get(STATS, params=params, timeout=90)
        r.raise_for_status()
        got = parse(r.text)
        if not got:
            print("   %d-%02d %-8s no table" % (year, month, label))
            continue
        for h, (a, d) in got.items():
            out[(label, h)] = (a, d)
        time.sleep(0.4)
    return out


def parse(html):
    """Pull the 여객 도착/출발 columns out of the hourly table.

    Read off the raw HTML rather than pandas.read_html: the table has a
    two-level header with merged cells, and the column positions are stable
    (구분, then 운항 x3, 여객 x3, 화물 x3) where the parsed MultiIndex is not.
    """
    out = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = [re.sub(r"<[^>]+>", "", c).replace(",", "").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if not cells:
            continue
        m = ROW.match(cells[0])
        if not m or len(cells) < 8:
            continue
        try:
            # 0 구분 | 1-3 운항 도착/출발/합계 | 4-6 여객 도착/출발/합계
            out[int(m.group(1))] = (float(cells[4]), float(cells[5]))
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    have = set()
    rows = []
    if os.path.exists(OUT) and not args.force:
        with io.open(OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                have.add((int(r["month"]) // 100, int(r["month"]) % 100))

    s = session()
    for year in args.years:
        for month in range(1, 13):
            if (year, month) in have:
                continue
            got = fetch_month(s, year, month)
            if not got:
                continue
            n = 0
            for (label, h), (a, d) in sorted(got.items()):
                rows.append({"month": "%d%02d" % (year, month), "hour": h,
                             "pas": label, "arr": "%.0f" % a,
                             "dep": "%.0f" % d})
                n += 1
            tot = sum(a + d for (lab, _), (a, d) in got.items() if lab == "all")
            print("   %d-%02d  %3d rows, %s passengers"
                  % (year, month, n, format(int(tot), ",")))

    if not rows:
        raise SystemExit("nothing downloaded -- the site may have changed")
    rows.sort(key=lambda r: (int(r["month"]), r["pas"], int(r["hour"])))
    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["month", "hour", "pas", "arr", "dep"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (OUT, len(rows)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
