"""T-1541: market rows for the runs. The packs moved on after T-1521 round 3 (Saanich joined, T-1377),
so its rows no longer match them; this takes the main tree's market-rows.json and adds round 3's
`built` fields (`fw`, `fg`, `cu`, per country, from rows_built.json).

    python trainworld/notes/T-1541/t1541_rows.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", ".."))
MAPS = os.path.dirname(TW)
R3 = os.path.join(MAPS, "artifacts", "trainworld", "T-1521", "corridor")
OUT = os.path.join(MAPS, "artifacts", "trainworld", "T-1541")
os.makedirs(OUT, exist_ok=True)
rows = json.load(open(os.path.join(TW, "app", "src", "game", "market-rows.json"), encoding="utf-8"))
built = json.load(open(os.path.join(R3, "rows_built.json"), encoding="utf-8"))
by_c = {}
for r in built.values():
    by_c.setdefault(r["country"], {k: r[k] for k in ("fw", "fg", "cu") if k in r})
missing = set()
for r in rows.values():
    f = by_c.get(r["country"])
    if f is None:
        missing.add(r["country"])
        continue
    r.update(f)
json.dump(rows, open(os.path.join(OUT, "rows_built.json"), "w"))
print(len(rows), "rows; countries with no built fields:", sorted(missing))
