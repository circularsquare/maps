"""T-1982: a country's preflight on its dry plan, in one call (docs/adding-a-country.md 3), with a picture.

    python checks/country_preflight.py <ISO> <dry plan.json> [--png <out.png>]

<ISO>: ISO3 or ISO2 of the country; the dry plan is `world_plan.py zones --countries XX --out <dry plan.json>`'s,
its extents beside it (<dry plan>.extents.geojson; for the live plan, data/work/worldplan's). From the maps root, with ..\\venv's Python. Reads only.

FLAG lines need a fix before the build:
- scattered: a row with parts over SCATTER_KM from all its others, so no chain of such steps reaches them from
  its main part, holding SCATTER_PEOPLE (a unit of far towns: "Unincorporated NT"; an archipelago's islands
  chain; emptier far islets are a NOTE)
- origin: a zone's origin over ORIGIN_KM from the unit it leads with (T-1762)
- lone: a zone under LONE_PEOPLE people (parks on its own figures; world_plan.ZONE_ALONE_MIN)
- cut city: a GHS-FUA over FUA_PEOPLE whose main urban centre is split between rows, two each holding
  world_plan.FUA_CUT_SHARE of its people (Adelaide)
NOTE lines are for the run: built packs of other countries within BORDER_KM (borders.json, the joins); a
planned row whose data/terrain/<id>.terrain is older than the dry plan (an earlier row's, which the runner
clears, T-1986); far islets with next to no one in them.
The picture (default ../artifacts/trainworld/preflight/<ISO>.png): the rows' extents, flagged rows in red, far
parts circled, other countries' packs within BORDER_KM in grey. Exit 1 when anything is flagged."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
TW = HERE.parent
sys.path.insert(0, str(TW / "pipeline"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pyproj  # noqa: E402
import shapely  # noqa: E402
from shapely import contains_xy  # noqa: E402
from shapely.geometry import Point, shape  # noqa: E402
from shapely.ops import transform  # noqa: E402

import world_plan as wp  # noqa: E402

SCATTER_KM = wp.SCATTER_KM
SCATTER_PEOPLE = wp.SCATTER_PEOPLE  # far parts with fewer (GHS-POP 2025) are empty islets: a NOTE
ORIGIN_KM = 25.0
LONE_PEOPLE = wp.ZONE_ALONE_MIN
FUA_PEOPLE = wp.FUA_WHOLE
BORDER_KM = 50.0
REGISTRY = TW / "app/src/game/registry.json"
PLAN = TW / "pipeline/world_plan.json"


def aeqd(g):
    """A function taking lon/lat geometries to metres around g's centre."""
    c = g.centroid
    t = pyproj.Transformer.from_crs(4326, f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +datum=WGS84", always_xy=True)
    return lambda h: transform(t.transform, h)


def units_of(iso, src):
    """{unit id: (name, lon/lat geometry)} from the plan's unit source (geoBoundaries, NUTS3, ITL3), or {}."""
    out = {}
    if src.get("kind") == "geoBoundaries":
        p = TW / f"data/raw/worldplan/geoboundaries/{iso}_{src['level']}.geojson"
        if p.exists():
            for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
                out[f["properties"]["shapeID"]] = (f["properties"].get("shapeName"), shape(f["geometry"]))
    elif src.get("kind") in ("NUTS3", "ITL3") and src.get("file"):
        for f in json.loads((TW / src["file"]).read_text(encoding="utf-8"))["features"]:
            pr = f["properties"]
            k = pr.get("ITL325CD") or pr.get("NUTS_ID") or pr.get("id")
            out[k] = (pr.get("ITL325NM") or pr.get("NAME_LATN") or pr.get("NUTS_NAME"), shape(f["geometry"]))
    return out


def main(argv):
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    planp = Path(argv[1])
    plan = json.loads(planp.read_text(encoding="utf-8"))
    extp = planp.with_suffix(".extents.geojson")
    if not extp.exists() and planp.resolve() == PLAN.resolve():
        extp = TW / "data/work/worldplan/extents.geojson"
    ext = {f["properties"]["id"]: f for f in json.loads(extp.read_text(encoding="utf-8"))["features"]} if extp.exists() else {}
    iso = argv[0].upper()
    if len(iso) == 2:
        iso = next((k for r in plan["metros"] if r.get("iso2") == iso for k in (r.get("units") or {})), iso)
    rows = [r for r in plan["metros"] if iso in (r.get("units") or {})]
    if not rows:
        print(f"{iso}: no rows hold its units in {planp}")
        return 2
    iso2 = next((r["iso2"] for r in rows if r.get("iso2")), None)
    zones = [r for r in rows if r.get("kind") == "zone"]
    geom = {r["id"]: shapely.make_valid(shape(ext[r["id"]]["geometry"])) for r in rows if r["id"] in ext}
    print(f"country_preflight {iso} {planp}: {len(rows)} rows ({len(zones)} zones), {len(geom)} with extents")
    flags, notes, red, circles = [], [], set(), []
    src = (plan.get("unit_sources") or {}).get(iso) or json.loads(PLAN.read_text(encoding="utf-8"))["unit_sources"].get(iso, {})
    units = units_of(iso, src)
    pop = wp.Pop()

    # scattered rows
    for r in rows:
        g = geom.get(r["id"])
        if g is None:
            continue
        m = aeqd(g)
        parts = sorted(shapely.get_parts(g), key=lambda p: -m(p).area)
        pm = [m(p) for p in parts]
        # the parts reached from the main one in steps of at most SCATTER_KM (an archipelago's islands chain)
        near = shapely.STRtree(pm)
        seen, front = {0}, [0]
        while front:
            front = [int(j) for i in front for j in near.query(pm[i], predicate="dwithin", distance=SCATTER_KM * 1000)
                     if int(j) not in seen]
            front = list(dict.fromkeys(front))
            seen.update(front)
        far = [(pm[0].distance(pm[k]) / 1000, parts[k]) for k in range(len(parts)) if k not in seen]
        if not far:
            continue
        holders = sorted({units[u][0] for u in r["units"][iso] if u in units
                          for _, p in far if units[u][1].buffer(1e-6).contains(p.representative_point())})
        people = sum(float(pop.inside(wp.to_moll(p))[2].sum()) for _, p in far)
        town = people >= SCATTER_PEOPLE
        (flags if town else notes).append(
            f"{'FLAG' if town else 'NOTE'} scattered: {r['id']} {r['name']}: {len(far)} parts {min(d for d, _ in far):,.0f}-"
            f"{max(d for d, _ in far):,.0f} km from its main part, over {SCATTER_KM:g} km from any other "
            f"({sum(m(p).area for _, p in far) / 1e6:,.0f} km2, {people:,.0f} people)"
            + (f", in {', '.join(holders)}" if holders else "")
            + (". Fix: that unit to dead space" if town else ": islets with next to no one, left as they are"))
        if town:
            red.add(r["id"])
        circles += [p.representative_point() for _, p in far]

    # origins
    for r in zones:
        if iso == "KOR" or not units or not r.get("origin"):
            continue
        names = (r.get("source") or {}).get("unit_names") or [r["name"]]
        lead = next((i for i in r["units"][iso] if i in units and units[i][0] == names[0]), None) \
            or next((i for i in r["units"][iso] if i in units), None)
        if lead is None:
            continue
        o = Point(*r["origin"])
        m = aeqd(o.buffer(1e-6))
        d = m(units[lead][1]).distance(m(o)) / 1000
        if d > ORIGIN_KM:
            flags.append(f"FLAG origin: {r['id']} {r['name']}: origin {r['origin']} is {d:,.0f} km from {names[0]}. "
                         f"Fix: checks/zone_new_origins.py")
            red.add(r["id"])

    # lone tiny zones
    for r in zones:
        if r.get("people", 0) < LONE_PEOPLE:
            flags.append(f"FLAG lone: {r['id']} {r['name']}: {r.get('people', 0):,} people, {len(r['units'][iso])} units. "
                         f"Fix: dead space (world_plan.ZONE_ALONE_MIN)")
            red.add(r["id"])

    # cities cut between rows: a big FUA's main urban centre split
    moll = {k: wp.to_moll(g) for k, g in geom.items()}
    every = shapely.union_all(list(moll.values())) if moll else None
    name = {r["id"]: r["name"] for r in rows}
    for _, fname, fp, cname, cg in (wp.fua_centres(iso, FUA_PEOPLE) if every is not None else []):
        if not cg.intersects(every):
            continue
        x, y, p = pop.inside(cg)
        held = sorted(((float(p[contains_xy(g, x, y)].sum()), k) for k, g in moll.items() if g.intersects(cg)), reverse=True)
        tot = sum(n for n, _ in held)
        big = [(n, k) for n, k in held if tot and n >= wp.FUA_CUT_SHARE * tot]
        if len(big) > 1:
            flags.append(f"FLAG cut city: {cname} ({tot / 1e6:.2f}M in the rows), the centre of {fname}'s functional urban "
                         f"area ({fp / 1e6:.2f}M in 2015), split: " + ", ".join(f"{k} {name[k]} {n / 1e6:.2f}M" for n, k in big)
                         + ". Fix: one zone or the metro takes it (world_plan.FUA_WHOLE)")
            red |= {k for _, k in big}

    # reused ids
    made = planp.stat().st_mtime
    for r in rows:
        t = TW / f"data/terrain/{r['id']}.terrain"
        if r.get("status", "planned") == "planned" and t.exists() and t.stat().st_mtime < made:
            import datetime
            notes.append(f"NOTE reused id: {r['id']} {r['name']}: data/terrain/{r['id']}.terrain of "
                         f"{datetime.date.fromtimestamp(t.stat().st_mtime)}, an earlier row's; the runner clears it "
                         f"with the row's other caches (world_run.ROW_CACHES, T-1986)")

    # built neighbours
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))["cities"]
    others = {c["id"]: c.get("country") for c in reg if c.get("country") and c.get("country") != iso2}
    near = {}
    if geom:
        allg = shapely.union_all(list(geom.values()))
        m = aeqd(allg)
        allm = m(allg).buffer(BORDER_KM * 1000)
        box = allg.buffer(BORDER_KM / 50.0).bounds
        for f in sorted((TW / "data/packs").glob("*.boundary.geojson")):
            pid = f.name.split(".")[0]
            if pid not in others:
                continue
            gj = json.loads(f.read_text(encoding="utf-8"))
            gs = [shape(ft["geometry"]) for ft in gj["features"]]
            b = shapely.GeometryCollection(gs).bounds
            if b[0] > box[2] or b[2] < box[0] or b[1] > box[3] or b[3] < box[1]:
                continue
            g = shapely.union_all([shapely.make_valid(h) for h in gs])
            if m(g).intersects(allm):
                near.setdefault(others[pid], []).append((pid, g))
    for c, ps in sorted(near.items()):
        notes.append(f"NOTE borders: {c} built within {BORDER_KM:g} km: {len(ps)} packs ({' '.join(sorted(p for p, _ in ps)[:20])}"
                     f"{' ...' if len(ps) > 20 else ''}); pipeline/borders.json has its pair factor or the national default")

    for ln in flags + notes:
        print(ln)

    # the picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    png = Path(argv[argv.index("--png") + 1]) if "--png" in argv else TW.parent / f"artifacts/trainworld/preflight/{iso}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 14))
    cmap = plt.get_cmap("tab20")
    for _, ps in near.items():
        for _, g in ps:
            for p in shapely.get_parts(g):
                for ring in [p.exterior]:
                    ax.fill(*ring.xy, color="#bbbbbb", lw=0)
    for n, r in enumerate(rows):
        g = geom.get(r["id"])
        if g is None:
            continue
        bad = r["id"] in red
        for p in shapely.get_parts(g):
            ax.fill(*p.exterior.xy, color=cmap(n % 20), alpha=0.75 if not bad else 0.9, lw=0)
            ax.plot(*p.exterior.xy, color="red" if bad else "#333333", lw=2.0 if bad else 0.3)
        c = g.representative_point()
        ax.text(c.x, c.y, r["id"], fontsize=8 if not bad else 11, color="red" if bad else "black",
                ha="center", va="center", weight="bold" if bad else "normal")
    for c in circles:
        ax.plot(c.x, c.y, "o", ms=22, mfc="none", mec="red", mew=2)
    ax.set_aspect(1 / max(0.2, np.cos(np.radians(np.mean([geom[k].centroid.y for k in geom]) if geom else 0))))
    ax.set_title(f"{iso} preflight: {len(rows)} rows, {len(flags)} flagged (red), other countries' packs in grey")
    fig.savefig(png, dpi=110, bbox_inches="tight")
    print(f"picture {png}")
    print(f"{len(flags)} flags, {len(notes)} notes")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
