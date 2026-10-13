"""T-181: New York's school input, NYC DOE buildings (before) against NCES (after), in the five
boroughs. Read-only: the old and new audits (data/work/pre-T181/nyc.schools.audit.json and
data/work/nyc.schools.audit.json).

    ..\\venv\\Scripts\\python.exe trainworld/notes/t181_schools.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

import shapefile
from shapely.geometry import Point, shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402

BOROUGHS = {"36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan", "36081": "Queens", "36085": "Staten Island"}


def main():
    old = json.load(open(os.path.join(b.WORK, "pre-T181", "nyc.schools.audit.json")))
    new = json.load(open(os.path.join(b.WORK, "nyc.schools.audit.json")))
    reader = shapefile.Reader(b.fetch(b.COUNTIES))
    polys = {r.record.as_dict()["GEOID"]: shape(r.shape.__geo_interface__) for r in reader.iterShapeRecords()
             if r.record.as_dict()["GEOID"] in BOROUGHS}

    def borough(s):
        p = Point(s["lon"], s["lat"])
        return next((BOROUGHS[g] for g, poly in polys.items() if poly.covers(p)), None)

    doe = [s for s in old["schools"] if s.get("source") == "NYC DOE"]
    nces_in = [s for s in new["schools"] if borough(s)]
    print(f"five boroughs: DOE {len(doe):,} buildings, {sum(s['enrollment'] for s in doe):,} pupils; "
          f"NCES {len(nces_in):,} schools, {sum(s['enrollment'] for s in nces_in):,} pupils")
    print(f"outside: old NCES {sum(s['enrollment'] for s in old['schools'] if s.get('source') != 'NYC DOE'):,}, "
          f"new {sum(s['enrollment'] for s in new['schools'] if not borough(s)):,}")
    by = defaultdict(lambda: [0, 0, 0, 0])
    for s in doe:
        k = borough(s) or "?"
        by[k][0] += 1; by[k][1] += s["enrollment"]
    for s in nces_in:
        k = borough(s)
        by[k][2] += 1; by[k][3] += s["enrollment"]
    for k, v in sorted(by.items()):
        print(f"  {k:<14} DOE {v[0]:>5,} bldgs {v[1]:>9,}   NCES {v[2]:>5,} schools {v[3]:>9,}  ({v[3] / max(v[1], 1) - 1:+.1%})")
    charter = [s for s in nces_in if "CHARTER" in s["name"].upper()]
    print(f"NCES charter schools (name) in the boroughs: {len(charter):,}, {sum(s['enrollment'] for s in charter):,} pupils")
    virt = Counter(s.get("virtual") for s in nces_in)
    print(f"NCES virtual flag in the boroughs: {dict(virt)}")
    # several schools per building: schools sharing a cell
    cells = Counter(s["cell"] for s in nces_in)
    shared = [c for c, n in cells.items() if n > 1]
    print(f"NCES cells with more than one school: {len(shared):,} of {len(cells):,} cells "
          f"({sum(cells[c] for c in shared):,} schools); most in one cell {max(cells.values())}")
    old_cells = Counter(s["cell"] for s in doe)
    e_old, e_new = defaultdict(float), defaultdict(float)
    for s in doe:
        e_old[s["cell"]] += s["enrollment"]
    for s in nces_in:
        e_new[s["cell"]] += s["enrollment"]
    only_old = set(e_old) - set(e_new)
    only_new = set(e_new) - set(e_old)
    both = set(e_old) & set(e_new)
    print(f"cells with school pupils: DOE {len(e_old):,}, NCES {len(e_new):,}; DOE only {len(only_old):,} "
          f"({sum(e_old[c] for c in only_old):,.0f} pupils), NCES only {len(only_new):,} ({sum(e_new[c] for c in only_new):,.0f}), "
          f"both {len(both):,} (DOE {sum(e_old[c] for c in both):,.0f}, NCES {sum(e_new[c] for c in both):,.0f})")
    ratio = sorted(((e_new[c] - e_old[c]), c) for c in both)
    print("  largest falls in a shared cell:", [(int(d), c) for d, c in ratio[:5]])
    print("  largest rises in a shared cell:", [(int(d), c) for d, c in ratio[-5:]])
    big_new = sorted(((e_new[c], c) for c in only_new), reverse=True)[:8]
    names = defaultdict(list)
    for s in nces_in:
        names[s["cell"]].append(s["name"])
    for e, c in big_new:
        print(f"  NCES-only cell {c}: {e:,.0f} pupils, {names[c][:4]}")
    big_old = sorted(((e_old[c], c) for c in only_old), reverse=True)[:5]
    dnames = {s["cell"]: s.get("name") for s in doe}
    for e, c in big_old:
        print(f"  DOE-only cell {c}: {e:,.0f} pupils, {dnames[c]}")
    print(f"most schools in one DOE building cell: {max(old_cells.values())}")


if __name__ == "__main__":
    main()
