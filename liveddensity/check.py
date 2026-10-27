"""Checks on data/hist.json for one epoch (results go in NOTES.md by hand).

    python -I check.py 2020 path/to/owid_population.csv

The CSV is Our World in Data's population grapher (UN WPP 2024 for 1950 on):
https://ourworldindata.org/grapher/population.csv?v=1&csvType=full&useColumnShortNames=true
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = ["CHN", "IND", "USA", "IDN", "BRA", "NGA", "RUS", "JPN", "DEU", "EGY", "COD", "MEX",
          "GBR", "AUS", "ARG", "KEN", "CAN", "PHL", "SGP", "FJI", "ISL", "NZL", "FRA", "NOR"]


def main():
    year, csv_path = sys.argv[1], sys.argv[2]
    with open(os.path.join(HERE, "data", "hist.json"), encoding="utf-8") as f:
        d = json.load(f)
    with open(os.path.join(HERE, "logs", f"summary_{year}.json"), encoding="utf-8") as f:
        s = json.load(f)
    wpp = {}
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["year"] == year and len(r["code"]) == 3:
                wpp[r["code"]] = float(r[[k for k in r if k not in ("entity", "code", "year")][0]])
    lo, step = d["bins"]["lo"], d["bins"]["step"]
    print(f"years {d['years']}, {len(d['countries'])} countries, "
          f"{os.path.getsize(os.path.join(HERE, 'data', 'hist.json')) / 1e6:.2f} MB")
    print(f"raster total {s['total']['total']:,.0f}; sum of country pops "
          f"{sum(c['pop'].get(year, 0) for c in d['countries'].values()):,.0f}; "
          f"world (WPP) {wpp.get('OWID_WRL', 0):,.0f}")
    for g in ("100m", "1km"):
        tot = sum(sum(h["v"]) for h in d["hist"][g][year].values())
        print(f"  sum of {g} bins {tot:,.0f}")
    print(f"reassigned {s['total']['near'] + s['total']['far']:,.0f} "
          f"(far {s['total']['far']:,.0f})")

    print("\ncountry       GHSL          WPP     diff   peak100m   peak1km   median100m median1km")
    for k in SAMPLE:
        c = d["countries"].get(k)
        if not c:
            print(k, "missing")
            continue
        p = c["pop"].get(year, 0)
        w = wpp.get(k)
        cols = []
        for g in ("100m", "1km"):
            h = d["hist"][g][year][k]
            v = h["v"]
            i = max(range(len(v)), key=lambda j: v[j])
            peak = 10 ** (lo + (h["o"] + i + 0.5) * step)
            half, acc, med = sum(v) / 2, 0, None
            for j, x in enumerate(v):
                acc += x
                if acc >= half:
                    med = 10 ** (lo + (h["o"] + j + 0.5) * step)
                    break
            cols.append((peak, med))
        diff = f"{(p / w - 1) * 100:+.1f}%" if w else "  n/a"
        print(f"{k} {c['name'][:10]:10} {p:13,.0f} {w or 0:13,.0f} {diff:>7} "
              f"{cols[0][0]:9,.0f} {cols[1][0]:9,.0f} {cols[0][1]:10,.0f} {cols[1][1]:9,.0f}")

    # worst mismatches against WPP among countries over a million
    rows = []
    for k, c in d["countries"].items():
        w, p = wpp.get(k), c["pop"].get(year, 0)
        if w and w > 1e6:
            rows.append(((p / w - 1) * 100, k, c["name"], p, w))
    rows.sort()
    print("\nlargest gaps vs WPP (countries over 1M):")
    for r in rows[:8] + rows[-8:]:
        print(f"  {r[1]} {r[2]}: {r[0]:+.1f}% ({r[3]:,.0f} vs {r[4]:,.0f})")

    # US 1km, smoothed peak over 0.1 log units
    h = d["hist"]["1km"][year]["USA"]
    v = h["v"]
    sm = [sum(v[max(0, j - 2):j + 3]) for j in range(len(v))]
    i = max(range(len(v)), key=lambda j: sm[j])
    print(f"\nUSA 1km smoothed peak at {10 ** (lo + (h['o'] + i + 0.5) * step):,.0f}/km2")
    print("USA 1km people by decade of density:")
    dec = {}
    for j, x in enumerate(v):
        e = int((lo + (h["o"] + j) * step) // 1)
        dec[e] = dec.get(e, 0) + x
    for e in sorted(dec):
        print(f"  10^{e}-10^{e + 1}: {dec[e]:,.0f}")


if __name__ == "__main__":
    main()
