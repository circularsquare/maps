"""How China's plan rows relate to prefecture-level divisions (GADM 4.1 CHN level 2, 368 features). Read-only.
Each plan unit goes to the prefecture holding most of its area. Prints the summary and writes rows.csv / prefs.csv."""
import collections
import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\notes\T-4364\kit")
import plan_edit as pe  # noqa: E402
import shared_lock  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import shape  # noqa: E402
import world_plan as wp  # noqa: E402

g = json.load(open(os.path.join(HERE, "gadm", "gadm41_CHN_2.json"), encoding="utf-8"))
prefs = []
for f in g["features"]:
    p = f["properties"]
    prefs.append((f"{p['NAME_1']}/{p['NAME_2']}", wp.to_moll(shapely.make_valid(shape(f["geometry"])))))
tree = shapely.STRtree([q for _, q in prefs])
by = pe.load_units("CHN")
doc = shared_lock.read_json(pe.PLAN)
rows = {r["id"]: r for r in doc["metros"] if (r.get("units") or {}).get("CHN")}
pref_of = {}
for u in by.values():
    best, ba = None, 0.0
    for j in tree.query(u.geom):
        a = u.geom.intersection(prefs[j][1]).area
        if a > ba:
            best, ba = prefs[j][0], a
    pref_of[u.id] = best
row_pref = {}
out_rows = []
one = 0
pp = collections.defaultdict(collections.Counter)  # prefecture -> row -> people
for k, r in rows.items():
    c = collections.Counter()
    for u in r["units"]["CHN"]:
        if u in by:
            c[pref_of[u.id] if hasattr(u, "id") else pref_of[u]] += by[u].pop
            pp[pref_of[u]][k] += by[u].pop
    tot = sum(c.values()) or 1
    top, tp = c.most_common(1)[0] if c else ("-", 0)
    share = tp / tot
    one += share >= 0.9
    row_pref[k] = top
    out_rows.append([k, r["name"], r.get("kind", "metro"), round(tot), len(c), top, f"{share:.2f}",
                     "; ".join(f"{p} {v / tot:.2f}" for p, v in c.most_common()[1:4])])
with open(os.path.join(HERE, "rows.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["id", "name", "kind", "people", "prefectures", "main", "main_share", "others"]] + out_rows)
split = 0
out_p = []
for p, c in pp.items():
    tot = sum(c.values()) or 1
    big = [(k, v / tot) for k, v in c.most_common() if v / tot >= 0.1]
    split += len(big) > 1
    out_p.append([p, round(tot), len(c), len(big), "; ".join(f"{k} {s:.2f}" for k, s in big)])
with open(os.path.join(HERE, "prefs.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["prefecture", "people", "rows", "rows_10pct", "rows"]] + sorted(out_p, key=lambda x: -x[1]))
zones = [x for x in out_rows if x[2] == "zone"]
print(f"rows {len(out_rows)} (zones {len(zones)}); rows with 90% of people in one prefecture: {one} "
      f"(zones {sum(1 for x in zones if float(x[6]) >= 0.9)})")
print(f"prefectures {len(pp)}; split over 2+ rows each holding 10%: {split}")
print("rows spanning prefectures most (main share under 0.6):")
for x in sorted(out_rows, key=lambda x: float(x[6]))[:25]:
    print("  ", x)
