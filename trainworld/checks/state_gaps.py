"""T-2695, T-2702: people in no plan unit and no pack. For each country, the geoBoundaries units of its plan level that
no plan row (any status) names and the country's zones did not leave as dead space, with the GHS-POP people of each
that lie outside every pack's outline. Before T-2695 `world_plan.country_units` dropped a unit whose representative
point lay outside the simplified first-level coast (Buenos Aires's comunas, Avellaneda, Ushuaia; Brazil's Guaruja,
T-2702), and no row could take it. A unit listed here with 10,000+ people outside packs is a hole in the map. Reads only.

    python checks/state_gaps.py ISO3[,ISO3...] [--min N]      (from trainworld/, ..\\venv's Python; N default 1000)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import shapely  # noqa: E402
import world_plan as wp  # noqa: E402
from shapely import STRtree, make_valid  # noqa: E402
from shapely.geometry import shape  # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MIN = float(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 1000.0
TW = Path(__file__).resolve().parents[1]
plan = json.loads(wp.OUT.read_text(encoding="utf-8"))
reg = json.loads((TW / "app/src/game/registry.json").read_text(encoding="utf-8"))
paths = [str(TW / "data/packs" / f"{c['id']}.boundary.geojson") for c in reg["cities"]]
paths = [p for p in paths if os.path.exists(p)]
import osm_units  # noqa: E402

boxes = osm_units.pack_boxes(paths)
meta = wp.europe_isos()[1]
pop = wp.Pop()
_packs = {}


def packs_near(ll_bounds):
    w, s, e, n = ll_bounds
    out = []
    for p in paths:
        a = boxes[p]
        if a[0] > e or a[2] < w or a[1] > n or a[3] < s:
            continue
        if p not in _packs:
            with open(p, encoding="utf-8") as f:
                fc = json.load(f)
            g = shapely.union_all([make_valid(shape(ft["geometry"])) for ft in fc["features"]])
            _packs[p] = wp.polygonal(wp.to_moll(g))
        out.append((os.path.basename(p).split(".")[0], _packs[p]))
    return out


grand = 0.0
for iso in sys.argv[1].split(","):
    src = plan["unit_sources"].get(iso)
    if not src or src.get("kind") != "geoBoundaries":
        print(iso, "no geoBoundaries source in the plan")
        continue
    if iso not in plan.get("zones", {}):
        print(iso, "metros only, no zone rows: land outside them is expected")
        continue
    named = {u for r in plan["metros"] for u in (r.get("units") or {}).get(iso, [])}
    z = plan["zones"][iso]
    dead = {d["unit"] for d in z.get("dead", [])} | {k if isinstance(k, str) else k.get("unit") for k in z.get("kept", [])}
    # the file as fetched: wp.gb_units' gap fill takes over an hour on Canada's ADM3 when not cached
    gb = json.load(open(wp.RAW / "geoboundaries" / f"{iso}_{src['level']}.geojson", encoding="utf-8"))
    units = [f for f in gb["features"] if f["geometry"]]
    cand, other = [], 0
    for f in units:
        uid = f["properties"]["shapeID"]
        if uid in named or uid in dead or any(u.startswith(uid + "~") for u in named):
            continue
        g = wp.polygonal(wp.to_moll(shape(f["geometry"])))
        if g.is_empty or g.bounds[2] - g.bounds[0] >= 2.0e7:
            continue
        if wp.left_out(iso, g) or wp.de_facto(iso, g) is None:  # held by another side (occupied Ukraine, Gilgit)
            other += 1
            continue
        cand.append((uid, f["properties"].get("shapeName") or "", g))
    lines, tot = [], 0.0
    if cand:
        bb = shapely.total_bounds([g for _, _, g in cand])
        ll = wp.to_ll(shapely.box(*bb)).bounds
        near = packs_near(ll)
        tree = STRtree([g for _, g in near]) if near else None
        for uid, name, g in cand:
            hit = tree.query(g, predicate="intersects") if tree is not None else []
            free = g.difference(shapely.union_all([near[i][1] for i in hit])) if len(hit) else g
            n_all = float(pop.inside(g)[2].sum())
            n = float(pop.inside(free)[2].sum()) if not free.is_empty else 0.0
            if n < MIN:
                continue
            tot += n
            c = wp.to_ll(g).representative_point()
            lines.append((n, f"  {uid} {name}: {n:,.0f} people outside packs (of {n_all:,.0f}), {free.area / 1e6:,.0f} of "
                             f"{g.area / 1e6:,.0f} km2 ({c.x:.2f}, {c.y:.2f}); packs touching: "
                             f"{','.join(near[i][0] for i in hit) or '-'}"))
    grand += tot
    print(f"{iso}: {len(units)} units at {src['level']}, {len(cand)} in no row and not dead space ({other} held by another side, left out); {len(lines)} with "
          f"{MIN:,.0f}+ people outside packs, {tot:,.0f} people", flush=True)
    for _, ln in sorted(lines, reverse=True):
        print(ln)
print(f"all: {grand:,.0f} people")
