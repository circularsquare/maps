"""Read the 旅客地域流動調査 FY2024 prefecture matrices for ships.

表2 sheet 旅客船 and 表3 sheet 航送船: 発 prefecture rows x 着 prefecture
columns, thousands of passengers. Writes data/flows_fy2024.csv in long form at
prefecture level (Hokkaido's four sub-regions dropped in favour of 北海道, the
全国 totals dropped), then prints each prefecture's departures beside the port
statistics' domestic boardings, to show whether 航送船 passengers sit on top of
旅客船 or inside it.
"""
import csv
import os
import sys
from collections import defaultdict

import openpyxl

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
DATA = os.path.join(HERE, "data")
SHEETS = [("ship", "flow_fy2024_t2.xlsx", "旅客船"), ("car_ferry", "flow_fy2024_t3.xlsx", "航送船")]
SKIP = {"道北", "道東", "道央", "道南", "全国"}


def read_matrix(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = [list(r) for r in wb[sheet].iter_rows(values_only=True)]
    header = rows[2]
    cols = {i: name for i, name in enumerate(header) if isinstance(name, str) and i >= 2}
    cells = {}
    for r in rows[5:]:
        origin = r[0]
        if not isinstance(origin, str) or r[1] != "発" or origin in SKIP:
            continue
        for i, dest in cols.items():
            if dest in SKIP:
                continue
            v = r[i]
            if isinstance(v, (int, float)) and v:
                cells[(origin, dest)] = float(v)
    return cells


def main():
    mats = {mode: read_matrix(os.path.join(RAW, f), s) for mode, f, s in SHEETS}

    path = os.path.join(DATA, "flows_fy2024.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "origin", "destination", "thousand_passengers"])
        for mode, cells in mats.items():
            for (o, d), v in sorted(cells.items()):
                w.writerow([mode, o, d, round(v, 3)])
    for mode, cells in mats.items():
        print(f"{mode}: {len(cells)} non-zero cells, total {sum(cells.values()):,.0f} thousand")
    print(f"wrote {path}")

    boarding = defaultdict(int)
    with open(os.path.join(DATA, "ports_2024.csv"), encoding="utf-8") as f:
        for p in csv.DictReader(f):
            boarding[p["prefecture"]] += int(p["dom_boarding"])

    dep = {mode: defaultdict(float) for mode in mats}
    for mode, cells in mats.items():
        for (o, _), v in cells.items():
            dep[mode][o] += v

    print("\nprefecture   port boardings (k)   旅客船 dep (k)   航送船 dep (k)   port/旅客船   port/(both)")
    for pref in sorted(boarding, key=lambda p: -boarding[p]):
        b = boarding[pref] / 1000
        s, c = dep["ship"][pref], dep["car_ferry"][pref]
        r1 = f"{b / s:.2f}" if s else "-"
        r2 = f"{b / (s + c):.2f}" if s + c else "-"
        print(f"{pref:<6} {b:>14,.0f} {s:>16,.0f} {c:>16,.0f} {r1:>12} {r2:>12}")
    tb = sum(boarding.values()) / 1000
    ts, tc = sum(dep["ship"].values()), sum(dep["car_ferry"].values())
    print(f"{'total':<6} {tb:>14,.0f} {ts:>16,.0f} {tc:>16,.0f} {tb / ts:>12.2f} {tb / (ts + tc):>12.2f}")

    print("\nlargest cross-prefecture 旅客船 flows (both directions, thousand):")
    pairs = defaultdict(float)
    for (o, d), v in mats["ship"].items():
        if o != d:
            pairs[tuple(sorted((o, d)))] += v
    for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {v:>8,.0f}  {a} - {b}")


if __name__ == "__main__":
    main()
