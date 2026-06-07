#!/usr/bin/env python3
"""scan_partial.py — find PARTIAL coverage holes: airports with some observed
FR24 flights but far fewer than their scheduled-route network implies. Signal =
observed 2-day flight count / OpenFlights scheduled-route degree. Well-covered
hubs sit near the median; high-degree airports with a very low ratio are
suspects (FR24 seeing only a sliver of their real traffic).
"""
import json, os, csv, collections

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MIN_DEG = 40   # only judge airports with a real route network

iata2icao, meta = {}, {}
for row in csv.reader(open("data/airports.dat", encoding="utf-8")):
    if len(row) < 8:
        continue
    icao, iata = row[5].strip(), row[4].strip()
    if len(icao) != 4 or icao == "\\N":
        continue
    if len(iata) == 3 and iata != "\\N":
        iata2icao.setdefault(iata, icao)
    meta[icao] = (iata if len(iata) == 3 else icao, row[2], row[3])

out = collections.defaultdict(int)
status = {}
for d in ["pull/2026-05-15", "pull/2026-05-16"]:
    for fn in os.listdir(d):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        ic = fn[:-5]
        doc = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        status[ic] = status.get(ic, True) and bool(doc.get("ok"))
        out[ic] += len(doc.get("records", []))

deg = collections.Counter()
for row in csv.reader(open("data/routes.dat", encoding="utf-8")):
    if len(row) < 5:
        continue
    s, t = row[2].strip(), row[4].strip()
    if len(s) == 3 and len(t) == 3:
        deg[s] += 1
        deg[t] += 1

rows = []
for iata, dg in deg.items():
    if dg < MIN_DEG:
        continue
    ic = iata2icao.get(iata)
    if not ic or not status.get(ic):
        continue
    o = out[ic]
    if o == 0:                       # total holes -> handled by scan_holes.py
        continue
    rows.append((o / dg, o, dg, ic, meta[ic][0], meta[ic][1], meta[ic][2]))

rows.sort()
allratios = sorted(r[0] for r in rows)
med = allratios[len(allratios) // 2]
print(f"obs-per-scheduled-route ratio: median {med:.2f} across {len(rows)} airports (deg>={MIN_DEG})")
print(f"\nlowest ratios = partial-coverage suspects (obs flights far below network size):\n")
print("ratio   obs  deg   airport                       country")
for ratio, o, dg, ic, iata, city, country in rows[:35]:
    flag = "  <<" if ratio < med * 0.15 else ""
    print(f"{ratio:5.2f} {o:5d} {dg:5d}   {iata} {ic} {city[:18]:18s} {country}{flag}")
