"""Per-state and per-division tables from dry_us2's outputs."""
import collections
import json
import os
import pickle
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import world_plan as wp  # noqa: E402,F401  (Unit for the pickle)

floor = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
st = pickle.load(open(os.path.join(HERE, "dry_state.pkl"), "rb"))
units = st["units"]
by = {u.id: u for u in units}
rows = json.load(open(os.path.join(HERE, "dry2_rows.json")))["rows"]
zd = json.load(open(os.path.join(HERE, f"dry2_zones_{floor}.json")))
zones = zd["zones"]
ABBR = {u.parents[0][0]: u.name.rsplit(", ", 1)[1] for u in units}
DIV = {"New England": "09 23 25 33 44 50", "Middle Atlantic": "34 36 42",
       "East North Central": "17 18 26 39 55", "West North Central": "19 20 27 29 31 38 46",
       "South Atlantic": "10 11 12 13 24 37 45 51 54", "East South Central": "01 21 28 47",
       "West South Central": "05 22 40 48", "Mountain": "04 08 16 30 32 35 49 56", "Pacific": "02 06 15 41 53",
       "Puerto Rico": "72"}
div_of = {s: d for d, ss in DIV.items() for s in ss.split()}
ghsl = collections.Counter()
for u in units:
    if u.iso != "TERR":
        ghsl[u.parents[0][0]] += u.pop
mrows = collections.defaultdict(list)
for r in rows:
    mrows[r["lead_state"]].append(r)
zs = collections.defaultdict(list)
for z in zones:
    zs[z["state"]].append(z)
print("| State | GHSL M | Metro rows (M) | Zones | Median zone M | Over cap | Under 0.4M |")
print("|---|---|---|---|---|---|---|")
tot = collections.Counter()
for s in sorted(ghsl, key=lambda s: -ghsl[s]):
    ps = [z["people"] / 1e6 for z in zs[s]]
    m = ", ".join(f"{r['name'].split(' [')[0]} {r['people'] / 1e6:.1f}" for r in mrows[s])
    over = sum(p > (1.5 if floor == 500000 else 2.5) for p in ps)
    under = sum(p < 0.4 for p in ps)
    print(f"| {ABBR[s]} | {ghsl[s] / 1e6:.1f} | {m} | {len(ps)} | {statistics.median(ps) if ps else 0:.2f} | {over} | {under} |")
print()
print("by division: rows, metros, zones, GHSL M")
for d, ss in DIV.items():
    ss = ss.split()
    nm = sum(len(mrows[s]) for s in ss)
    nz = sum(len(zs[s]) for s in ss)
    print(f"  {d}: {nm} metros, {nz} zones, {sum(ghsl[s] for s in ss) / 1e6:.1f}M")
ps = [z["people"] for z in zones]
print(f"all: {len(zones)} zones, {sum(ps) / 1e6:.1f}M, median {statistics.median(ps) / 1e6:.2f}M")
print("over cap:", [(z["name"], round(z["people"] / 1e6, 2), len(z["units"])) for z in zones if z["people"] > (1.5e6 if floor == 500000 else 2.5e6)])
print("under 0.4M:", [(z["name"], round(z["people"] / 1e6, 3), z["km2"]) for z in zones if z["people"] < 4e5])
big = sorted(zones, key=lambda z: -z["km2"])[:12]
print("largest by area:", [(z["name"], z["km2"], round(z["people"] / 1e6, 2)) for z in big])
xs = [ln for ln in zd["lines"] if "another region" in ln or "stays" in ln]
print("cross-region or alone:")
for ln in xs:
    print("  ", ln)
