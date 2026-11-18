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
- phantom: a unit with a 1 km GHSL pixel of PHANTOM_PIXEL people or more that is a spike (its eight neighbours
  average under PHANTOM_SPIKE of it) in no GHS-UCDB urban centre, or in one that first counted as urban in
  PHANTOM_YOB or later: villages GHSL piles people on (Cambodia's Trapeang Prasat, T-2000, T-2002). Pixels alone
  cannot tell a phantom from a dense small town (Turkey and the Philippines have both), so read each against a
  census (world_plan's admin_totals) before holding a row
- off town: a row whose origin has no pixel of ORIGIN_DENSE_SHARE of the row's densest within ORIGIN_TOWN_KM,
  or has a pixel of PHANTOM_PIXEL people there but lies over ORIGIN_UC_KM from every urban centre of a row that
  holds one (a village or an outskirt: Konya, Erzurum, T-2299). Fix: the town's centre by hand, as checks/zone_new_origins.py does for the densest pixel
- cut name: a row named, or led, by a unit whose name is as long as a pile-up of names in its source file
  (geoBoundaries cuts Russia's ADM2 names at 26 characters, Canada's and Cambodia's with a "*"). Fix: name the
  zone in zone-names.json
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
# a km2 of 20,000 people is a Phnom Penh or Manila core; in a village it is GHSL's error (Cambodia's phantoms hold
# 23,000-48,000)
PHANTOM_PIXEL = 20000.0
# a real core's neighbours hold a third to three quarters of it (Philippine cities 0.3-0.7); Cambodia's phantoms
# 0.06-0.20
PHANTOM_SPIKE = 0.2
# UCDB draws a centre round each of Cambodia's phantoms, all first urban in 2020 or 2025 (they grew in GHSL's
# newest epochs); Turkish and Philippine towns with such a pixel mostly date from 2015 or earlier
PHANTOM_YOB = 2020
# an origin off the town: the origin's pixel holds under a tenth of the row's densest; or sits on a hot pixel
# (PHANTOM_PIXEL) outside every urban centre of a row that has some (Konya's and Erzurum's old origins were both on
# one, GHSL's phantom villages). A lower pixel outside a centre is a small town in a village row: Indonesia,
# Russia and Vietnam have 40 such rows at origin pixels of 3,000-6,000
ORIGIN_DENSE_SHARE = 0.1
ORIGIN_TOWN_KM = 3.0
ORIGIN_UC_KM = 3.0
# source names piled up at one length (the file's longest) as a cut makes: at least this many, twice the next
# shorter length, an eighth of the next longer
CUT_MIN, CUT_RATIO, CUT_LONGER = 20, 2.0, 8.0
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
        # every level the plan read (Malaysia's Sabah is ADM2 under an ADM3 plan), split as gb_units splits
        pop = wp.Pop() if iso in wp.GB_SPLIT_FAR else None
        for level in sorted(set(src.get("urls", {})) | {src["level"]}):
            p = TW / f"data/raw/worldplan/geoboundaries/{iso}_{level}.geojson"
            if not p.exists():
                continue
            for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
                uid, name, g = f["properties"]["shapeID"], f["properties"].get("shapeName"), shape(f["geometry"])
                pieces = [wp.to_ll(q) for q in wp.far_parts(wp.to_moll(g), pop)] if pop is not None else [g]
                for k, q in enumerate(pieces):
                    out[uid if k == 0 else f"{uid}~{k}"] = (name, q)
    elif src.get("kind") in ("NUTS3", "ITL3") and src.get("file"):
        for f in json.loads((TW / src["file"]).read_text(encoding="utf-8"))["features"]:
            pr = f["properties"]
            k = pr.get("ITL325CD") or pr.get("NUTS_ID") or pr.get("id")
            out[k] = (pr.get("ITL325NM") or pr.get("NAME_LATN") or pr.get("NUTS_NAME"), shape(f["geometry"]))
    return out


_UC = []


def uc_centres():
    """(STRtree of the GHS-UCDB urban centre polygons, their 2025 people, centroid points, polygons, first year
    urban), Mollweide."""
    if not _UC:
        import pyogrio
        df = pyogrio.read_dataframe(wp.UCDB, layer="GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A",
                                    columns=["ID_UC_G0", "GC_POP_TOT_2025", "GC_UCB_YOB_2025"])
        gs = [wp.polygonal(g) for g in df.geometry]
        _UC.extend([shapely.STRtree(gs), df.GC_POP_TOT_2025.to_numpy(float), [g.centroid for g in gs], gs,
                    df.GC_UCB_YOB_2025.to_numpy(float)])
    return _UC


def cut_lengths(iso, src):
    """The name lengths a geoBoundaries file of the plan's levels piles names up at, as a cut does."""
    out = set()
    for level in sorted(set(src.get("urls", {})) | {src.get("level")}):
        p = TW / f"data/raw/worldplan/geoboundaries/{iso}_{level}.geojson"
        if src.get("kind") != "geoBoundaries" or not p.exists():
            continue
        import collections
        c = collections.Counter(len(f["properties"].get("shapeName") or "")
                                for f in json.loads(p.read_text(encoding="utf-8"))["features"])
        out |= {n for n in c if n >= 12 and c[n] >= CUT_MIN and c[n] >= CUT_RATIO * c[n - 1] and c[n + 1] * CUT_LONGER <= c[n]}
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
        # a unit's own islands holding under FAR_SHARE of its people (Gresik's Bawean): as world_plan.far_towns keeps them
        own = sum(float(pop.inside(wp.to_moll(units[u][1]))[2].sum()) for u in r["units"][iso] if u in units
                  and any(units[u][1].buffer(1e-6).contains(p.representative_point()) for _, p in far))
        islands = people >= SCATTER_PEOPLE and own > 0 and people < wp.FAR_SHARE * own
        town = people >= SCATTER_PEOPLE and not islands
        (flags if town else notes).append(
            f"{'FLAG' if town else 'NOTE'} scattered: {r['id']} {r['name']}: {len(far)} parts {min(d for d, _ in far):,.0f}-"
            f"{max(d for d, _ in far):,.0f} km from its main part, over {SCATTER_KM:g} km from any other "
            f"({sum(m(p).area for _, p in far) / 1e6:,.0f} km2, {people:,.0f} people)"
            + (f", in {', '.join(holders)}" if holders else "")
            + (". Fix: that unit to dead space" if town else
               f": the unit's own islands, {people / own:.0%} of its people, left as they are (T-1738)" if islands else
               ": islets with next to no one, left as they are"))
        if town:
            red.add(r["id"])
        circles += [p.representative_point() for _, p in far]

    # origins
    for r in zones:
        # a grouped zone (`group_level`: Korea, Malaysia) takes its origin in the group it is named after, not
        # its first unit
        if (r.get("source") or {}).get("groups") or not units or not r.get("origin"):
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

    # phantom people, origins off their town, cut names
    uc = uc_centres()
    cut = cut_lengths(iso, src)
    for r in rows:
        bad = []
        for uid in r["units"][iso]:
            if uid not in units:
                continue
            nm, ug = units[uid]
            x, y, p = pop.inside(wp.to_moll(ug))
            hot = np.nonzero(p >= PHANTOM_PIXEL)[0]
            for i in hot[np.argsort(-p[hot])]:
                hit = uc[0].query(shapely.points(x[i], y[i]), predicate="within")
                big = float(uc[1][hit].max()) if len(hit) else 0.0
                if len(hit) and uc[4][hit].max() < PHANTOM_YOB:
                    continue
                X, Y, P = pop.pixels((x[i] - 1500, y[i] - 1500, x[i] + 1500, y[i] + 1500))
                d = np.hypot(X - x[i], Y - y[i])
                nb = float(P[(d > 0) & (d < 1500)].sum()) / 8
                if nb < PHANTOM_SPIKE * p[i]:
                    bad.append((float(p[i]), nm, nb, big))
                    break
        if bad:
            flags.append(f"FLAG phantom: {r['id']} {r['name']}: " + "; ".join(
                f"{nm} has a 1 km pixel of {px:,.0f} people, its neighbours {nb:,.0f} each, in "
                + (f"an urban centre of {big:,.0f}" if big else "no urban centre") for px, nm, nb, big in bad)
                + ". Fix: compare with a census; hold the row or use census totals (T-2000)")
            red.add(r["id"])
        g = geom.get(r["id"])
        if g is None or not r.get("origin"):
            continue
        gm = wp.to_moll(g)
        x, y, p = pop.inside(gm)
        om = wp.to_moll(Point(*r["origin"]))
        if not len(p):
            continue
        near = np.hypot(x - om.x, y - om.y) <= ORIGIN_TOWN_KM * 1000
        here = float(p[near].max()) if near.any() else 0.0
        why = None
        if here < ORIGIN_DENSE_SHARE * float(p.max()):
            why = (f"the densest pixel within {ORIGIN_TOWN_KM:g} km of the origin holds {here:,.0f} people, "
                   f"the row's densest {p.max():,.0f}")
        else:
            inrow = [int(k) for k in uc[0].query(gm, predicate="intersects")
                     if gm.intersection(uc[3][int(k)]).area >= 0.25 * uc[3][int(k)].area]
            if (inrow and here >= PHANTOM_PIXEL
                    and not any(uc[3][k].distance(om) <= ORIGIN_UC_KM * 1000 for k in inrow)):
                why = (f"a pixel of {here:,.0f} people within {ORIGIN_TOWN_KM:g} km of the origin, in none of the row's "
                       f"{len(inrow)} urban centres")
        if why:
            flags.append(f"FLAG off town: {r['id']} {r['name']}: origin {r['origin']}: {why}. "
                         f"Fix: the town's centre by hand (planned status, rebuilt)")
            red.add(r["id"])
    for r in rows:
        lead = ((r.get("source") or {}).get("unit_names") or [None])[0]
        for nm in {r["name"], lead} - {None}:
            if nm.endswith("*") or len(nm) in cut:
                flags.append(f"FLAG cut name: {r['id']} {r['name']}: '{nm}' is {len(nm)} characters, "
                             f"the length its source file piles names up at, so cut unless it is a whole name. Fix: name the zone in zone-names.json")
                red.add(r["id"])
                break

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
                if p.geom_type == "Polygon":
                    ax.fill(*p.exterior.xy, color="#bbbbbb", lw=0)
    for n, r in enumerate(rows):
        g = geom.get(r["id"])
        if g is None:
            continue
        bad = r["id"] in red
        for p in (q for q in shapely.get_parts(g) if q.geom_type == "Polygon"):
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
