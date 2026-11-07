"""T-2695: land and people in no plan unit. For each country, the geoBoundaries units (each level its plan source
names) that the cached plan state (data/work/worldplan/state.pkl) lacks and that its state units cover under half of,
with GHS-POP people. Before T-2695 `world_plan.country_units` dropped a unit whose representative point lay outside the
simplified first-level coast (Buenos Aires's comunas, Avellaneda, Ushuaia; Brazil's Guaruja, T-2702). Reads only.

    python checks/state_gaps.py ISO3[,ISO3...]        (from trainworld/, ..\\venv's Python)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import shapely  # noqa: E402
import world_plan as wp  # noqa: E402
from shapely import STRtree  # noqa: E402
from shapely.geometry import shape  # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
plan = json.loads(wp.OUT.read_text(encoding="utf-8"))
pop = wp.Pop()
for iso in sys.argv[1].split(","):
    src = plan["unit_sources"].get(iso)
    if not src or src.get("kind") != "geoBoundaries":
        print(iso, "no geoBoundaries source in the plan")
        continue
    st = wp.state_units(iso)
    have = {u for u, _, _ in st}
    sg = [g for _, _, g in st]
    tree = STRtree(sg)
    tot, lines = 0, []
    for adm in sorted(src["urls"]):
        gb = json.load(open(wp.RAW / "geoboundaries" / f"{iso}_{adm}.geojson", encoding="utf-8"))
        for f in gb["features"]:
            if f["properties"]["shapeID"] in have or not f["geometry"]:
                continue
            g = wp.polygonal(wp.to_moll(shape(f["geometry"])))
            if g.is_empty:
                continue
            near = tree.query(g, predicate="intersects")
            un = g.difference(shapely.union_all([sg[i] for i in near])) if len(near) else g
            if un.area < 0.5 * g.area:
                continue
            n = float(pop.inside(un)[2].sum())
            tot += n
            c = shape(f["geometry"]).representative_point()
            lines.append(f"  {adm} {f['properties']['shapeID']} {f['properties']['shapeName']}: {n:,.0f} people uncovered, "
                         f"{un.area / 1e6:,.0f} of {g.area / 1e6:,.0f} km2 ({c.x:.2f}, {c.y:.2f})")
    print(f"{iso}: {len(st)} state units; {len(lines)} file units mostly outside them, {tot:,.0f} people")
    for ln in lines:
        print(ln)
