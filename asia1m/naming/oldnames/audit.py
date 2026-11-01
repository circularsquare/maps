"""Flag map-read names that likely sit on the wrong region -> oldnames/audit.csv

    python asia1m/naming/oldnames/audit.py

A China row is flagged when its characters appear in a neighbour's county/township names
but not in its own region's (the Longwan / Rui'an swap near Wenzhou was this kind).
Picks and fixes are skipped.
"""
import os, csv, json, sys
sys.stdout.reconfigure(encoding="utf-8")
N = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
regions = {r["key"]: r for r in json.load(open(f"{N}/regions.json", encoding="utf-8"))}
def names(r):
    s = set()
    for u in r.get("units", []):
        for v in (u.get("local"), u.get("prefecture_local")):
            if v: s.add(v)
    for t in r.get("towns", []):
        for v in (t.get("town"), t.get("county")):
            if v: s.add(v)
    return s
def has(zh, pool): return any(zh in p for p in pool)
out = []
for row in csv.DictReader(open(f"{N}/all_names.csv", encoding="utf-8-sig")):
    if row["source"] != "map" or row["country"] != "China" or len(row["zh"]) < 2: continue
    r = regions.get(row["key"])
    if not r or has(row["zh"], names(r)): continue
    hits = [nb["key"] for nb in r["neighbours"] if nb["key"] in regions and has(row["zh"], names(regions[nb["key"]]))]
    if hits:
        out.append(dict(key=row["key"], name=row["name"], zh=row["zh"], found_in_neighbour=";".join(hits),
                        neighbour_name=";".join(next((x["name"] for x in csv.DictReader(open(f"{N}/all_names.csv", encoding="utf-8-sig")) if x["key"] == h), "") for h in hits)))
with open(f"{N}/oldnames/audit.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["key", "name", "zh", "found_in_neighbour", "neighbour_name"]); w.writeheader(); w.writerows(out)
print(len(out), "flagged")
for o in out[:40]: print(o)
