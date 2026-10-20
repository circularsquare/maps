"""T-117, T-276: the Northeast's sites and commutes between zones, data/packs/intercity.json.

Endpoint sites from the Northeast's zone packs (the registry's cities with a `market`: 600 m squares
in the New York equirectangular frame, `origin`; 6 km zones), each site's parking for an arriving
car (the local layer's rule), then the joint commute gravity, which reads these sites and writes
the commutes between zones into the same file. Mercer County is in New York's pack since T-276;
its market place (id 2) keeps its name and has no sites.

    python pipeline/intercity.py            the sites, the joint commute gravity (rewrites every
                                            Northeast pack's gravity, T-276), then every zone's
                                            market pack (pipeline/market_zones.py, T-377);
                                            --no-joint skips the gravity
    python pipeline/intercity.py --group auto [ids]
                                            every joint gravity group (T-539: packs whose outlines
                                            lie within GROUP_GAP_KM, chained), or the groups holding
                                            the ids, then their market packs; --no-market stops after
                                            the gravity; --dry writes only reports and factor dumps;
                                            --root DIR reads a copy's registry and packs, --work DIR
                                            takes the outputs (no market step then)
    python pipeline/intercity.py --groups [--plan]
                                            list the groups (with the world plan's planned metros
                                            marked +) and their solve units (T-786: a group
                                            spanning countries is solved a country at a time)
    python pipeline/intercity.py --solve a,b [--fixed c,d] --name n
                                            one solve unit, no market step (world_run.py's join)

The game no longer reads this file (T-377): pipeline/market_zones.py takes the Northeast's sites
and commutes from it into each zone's own market pack, `data/packs/<id>.market.json`.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from read_pack import load_pack
from build_city import CITIES, R_EARTH

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = CITIES["nyc"]["origin"]
# Stable IDs: saved city selections and registry entries use these indices.
REGISTRY = json.loads((ROOT / "app/src/game/registry.json").read_text(encoding="utf-8"))
# a city with market -1 is not in the Northeast market (Toronto, T-277, until T-276 places it)
CITY_MARKETS = tuple((c["market"], c["id"]) for c in REGISTRY["cities"] if c["market"] >= 0)
MARKETS = [p["name"] for p in REGISTRY["market"]["places"]]

# T-276: the market's trips a weekday per resident (fitted with its curve, research/intercity/market_fit.py)
TRIP_RATE = 0.107
GROUP_GAP_KM = 50.0
# joint_gravity's cut for pairs across packs: a neighbouring country's pack this near takes part in a
# country's solve, held at its factors (T-786)
CUT_KM = 150.0
# a time guard on one solve (a country's packs since T-786): about 6x Japan's 68k zones (notes/T-539.md)
MAX_GROUP_ZONES = 400_000
# the local layer's parking (sim kernel `Params` defaults: park_min, park_paid_min, park_paid_d0,
# park_paid_cap_min, park_free_d; `CELL_KM2`)
PARK_MIN, PARK_PAID_MIN, PARK_PAID_D0, PARK_PAID_CAP_MIN, PARK_FREE_D = 7.0, 15.0, 3000.0, 50.0, 8000.0
CELL_KM2 = 0.1053
# kernel `CORDONS` (congestion charges, perceived minutes a one-way trip): lon/lat rings, charge
CORDONS = [
    (13.5, [(-74.020, 40.700), (-74.016, 40.715), (-74.012, 40.730), (-74.009, 40.745), (-74.002, 40.760), (-73.994, 40.774), (-73.959, 40.759), (-73.968, 40.748), (-73.972, 40.738), (-73.974, 40.726), (-73.978, 40.711), (-73.999, 40.707), (-74.012, 40.701)]),
    (35.1, [(-0.159, 51.513), (-0.163, 51.520), (-0.126, 51.528), (-0.105, 51.530), (-0.088, 51.526), (-0.077, 51.524), (-0.072, 51.515), (-0.075, 51.5055), (-0.100, 51.495), (-0.125, 51.487), (-0.144, 51.495), (-0.152, 51.503)]),
]


def in_ring(ring, x, y):
    inside = np.zeros(len(x), bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        (xi, yi), (xj, yj) = ring[i], ring[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = (yi > y) != (yj > y)
            inside ^= cross & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        j = i
    return inside


def park_by_cell(h, a, lon, lat):
    """Each cell's parking for an arriving car, perceived minutes: the mean over drivers in its
    2 km zone. Mirrors the sim kernel's `Params::park_mean` over `park_parts` (T-380, T-440) with
    `cordon_by_zone`: 7 to park (T-637; the packs built before T-636 hold 3, T-638) and the zone's congestion charge for everyone, a paid space
    15 ln(d / 3,000) more (at most 50) for the share 1 - 1 / (1 + d / 8,000), d the zone's jobs
    per km² of land (res-9 cells, a merged res-8 cell counts 7, at most the square)."""
    nz, zone = h["zones"], a["zone"].astype(np.int64)
    jobs = a["jobs"].astype(np.float64)
    res8 = ((a["h3"] >> np.uint64(52)) & np.uint64(0xF)) == 8
    ncell = np.bincount(zone, np.where(res8, 7.0, 1.0), nz)
    land = np.minimum(np.maximum(ncell, 1) * CELL_KM2, (h["gravity"]["zone_m"] / 1000.0) ** 2)
    zj = np.bincount(zone, jobs, nz)
    d = zj / land
    paid = np.minimum(PARK_PAID_MIN * np.log(np.maximum(d, PARK_PAID_D0) / PARK_PAID_D0), PARK_PAID_CAP_MIN)
    free = 1.0 / (1.0 + d / PARK_FREE_D)
    charged = np.zeros(nz)
    for charge, ring in CORDONS:
        inside = in_ring(ring, lon, lat) & (jobs > 0)
        charged += np.bincount(zone[inside], jobs[inside] * charge, nz)
    cordon = np.where(zj > 0, charged / np.maximum(zj, 1e-12), 0.0)
    return (PARK_MIN + cordon + (1.0 - free) * paid)[zone]


def lonlat_to_market(lon, lat):
    return (R_EARTH * np.radians(lon - ORIGIN[0]) * math.cos(math.radians(ORIGIN[1])),
            R_EARTH * np.radians(lat - ORIGIN[1]))


def sites():
    """Sites ([x, y, people, jobs, market]) and each one's parking for an arriving car."""
    rows = []
    for market, key in CITY_MARKETS:
        h, a = load_pack(key)
        o = h["origin"]
        lat = o["lat"] + np.degrees(a["y_m"] / R_EARTH)
        lon = o["lon"] + np.degrees(a["x_m"] / (R_EARTH * math.cos(math.radians(o["lat"]))))
        park = park_by_cell(h, a, lon, lat)
        rows.append((market, lon, lat, a["pop"], a["jobs"], park))
    out, parks = [], []
    for market, lon, lat, pop, jobs, park in rows:
        x, y = lonlat_to_market(lon, lat)
        groups = {}
        for xx, yy, pp, jj, pk in zip(x, y, pop, jobs, park):
            if pp + jj <= 0: continue
            k = (int(xx // 600), int(yy // 600))
            r = groups.setdefault(k, [0., 0., 0., 0., 0., 0.])
            r[0] += xx * (pp + jj); r[1] += yy * (pp + jj)
            r[2] += pp; r[3] += jj
            r[4] += pk * (jj + 1e-6); r[5] += jj + 1e-6
        for r in groups.values():
            w = r[2] + r[3]
            out.append([round(float(r[0]/w), 1), round(float(r[1]/w), 1), round(float(r[2]), 3), round(float(r[3]), 3), market])
            parks.append(round(float(r[4] / r[5]), 2))
    return out, parks


def joint(root=ROOT, work=None, dry=False):
    """T-276: one commute gravity over every zone pack (`joint_gravity`): rewrites each pack's
    gravity factors and the market pack's commutes between zones. About a minute (T-297)."""
    work = work or root / "data/work"
    extra = ["--dry", "--dump", str(work / "joint_dump.json")] if dry else []
    cargo_bin("joint_gravity", "--root", str(root), "--threads", os.environ.get("OMP_NUM_THREADS", "3"),
              "--report", str(work / "joint_report.json"), *extra)


def solve_group(ids, name, root=ROOT, work=None, dry=False, fixed=()):
    """T-463, T-539: one joint commute gravity over packs outside the Northeast: each pack's
    factors, and the cross-pack commutes from their homes in <work>/<name>.joint_commutes.json for
    market_zones.py; `fixed` packs (T-786) take part at their own factors and are not written. A
    file no pack takes its commutes from any more (joint_files) is removed."""
    work = work or root / "data/work"
    out = work / f"{name}.joint_commutes.json"
    extra = ["--dry", "--dump", str(work / f"{name}.joint_dump.json")] if dry else []
    if fixed:
        extra += ["--fixed", ",".join(fixed)]
    # Japan's columns stall at about 8e-4 from step ~90, all in Okinawa's outlying islands (weakly
    # joined to anything; notes/T-463.md): the best step is kept, so 150 steps are as good as 500
    cargo_bin("joint_gravity", "--root", str(root), "--threads", os.environ.get("OMP_NUM_THREADS", "3"), "--max-iters", "150",
              "--packs", ",".join(ids), "--out", str(out), "--report", str(work / f"{name}.joint_report.json"), *extra)
    if not dry:
        files = joint_files(work)
        for p, f in files.items():
            if any(q != p and g["t"] > f["t"] and covers(g, f) for q, g in files.items()):
                print(f"removed {p.name}: a newer solve covers every pair of packs it covered")
                p.unlink()


def joint_files(work):
    """T-786: every joint commutes file as {path: {solved, packs, t}}, oldest first. A file covers
    the pairs of its packs with an end among its solved ones; market_zones.py takes each pair from
    the newest file covering it. A file from before T-786 solved every pack it lists."""
    files = {}
    for p in sorted(Path(work).glob("*.joint_commutes.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        files[p] = {"solved": set(g.get("solved") or g["packs"]), "packs": set(g["packs"]),
                    "t": float(g.get("solved_at") or p.stat().st_mtime)}
    return dict(sorted(files.items(), key=lambda kv: kv[1]["t"]))


def covers(a, b):
    """Whether joint file `a` covers every pack pair `b` covers."""
    return b["solved"] <= a["solved"] and b["packs"] <= a["packs"]


def country_of(reg, root=ROOT):
    """Each registered pack's country: its header's, else the registry's (T-590, T-779)."""
    out = {}
    for c in reg["cities"]:
        p = root / f"data/packs/{c['id']}.json"
        h = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        out[c["id"]] = (h.get("country") or c.get("country") or "US").upper()
    return out


def solve_units(reg, root=ROOT, only=(), gap=None):
    """T-786 (T-781): what each joint solve covers. A group of one country is solved whole; a group
    spanning countries is solved a country at a time, the group's packs of other countries within
    CUT_KM of that country's held at their factors (`fixed`), so a country's join never re-solves
    its neighbours and no solve grows past one country. [{name, members, fixed, country}] for every
    unit, or those holding a pack of `only`."""
    cc = country_of(reg, root)
    units = []
    for g in auto_groups(reg, root, gap=gap):
        by = {}
        for i in g:
            by.setdefault(cc[i], []).append(i)
        if len(by) == 1:
            units.append({"name": g[0], "members": g, "fixed": [], "country": cc[g[0]]})
            continue
        shapes = None
        for c, ms in by.items():
            if only and not set(ms) & set(only):
                continue
            if shapes is None:
                shapes = {i: outline(root / f"data/packs/{i}.boundary.geojson") for i in g}
            units.append({"name": f"{ms[0]}-{c.lower()}", "members": ms, "country": c,
                          "fixed": [j for j in g if cc[j] != c and near_any(shapes[j], [shapes[m] for m in ms], CUT_KM)]})
    return [u for u in units if not only or set(u["members"]) & set(only)]


def near_any(a, others, km):
    """Whether shape `a` lies within `km` of any of `others` (boxes first)."""
    dy = km / 111.0
    for b in others:
        lat = max(abs(a.bounds[1]), abs(a.bounds[3]), abs(b.bounds[1]), abs(b.bounds[3]))
        dx = dy / max(math.cos(math.radians(min(lat + dy, 89.0))), 0.01)
        if (b.bounds[0] - dx < a.bounds[2] and a.bounds[0] - dx < b.bounds[2] and b.bounds[1] - dy < a.bounds[3]
                and a.bounds[1] - dy < b.bounds[3] and gap_km(a, b) <= km):
            return True
    return False


def solve_auto(only=(), root=ROOT, work=None, dry=False, gap=None):
    """T-539: every joint gravity group of the registered packs (auto_groups), or those holding a
    pack of `only`, solved in turn. The Northeast is solved its own way (its sites in
    intercity.json, T-276), which needs every member to have a Northeast market place. Returns the
    ids of the packs solved."""
    reg = json.loads((root / "app/src/game/registry.json").read_text(encoding="utf-8"))
    work = work or root / "data/work"
    ne = {c["id"] for c in reg["cities"] if c.get("market", -1) >= 0}
    solved = []
    for u in solve_units(reg, root, only, gap=gap):
        g = u["members"]
        solve_unit(g, u["fixed"], u["name"], root, work, dry, ne)
        solved += g
    for i in set(only) - set(solved):
        print(f"{i}: no registered pack within {GROUP_GAP_KM:g} km, so no joint gravity")
    return solved


def solve_unit(g, fixed, name, root=ROOT, work=None, dry=False, ne=None):
    """One solve unit (solve_units): its members balanced, `fixed` held at their factors."""
    t0 = time.time()
    if ne is None:
        reg = json.loads((root / "app/src/game/registry.json").read_text(encoding="utf-8"))
        ne = {c["id"] for c in reg["cities"] if c.get("market", -1) >= 0}
    nz = sum(json.loads((root / f"data/packs/{i}.json").read_text(encoding="utf-8"))["zones"] for i in g)
    if nz > MAX_GROUP_ZONES:
        sys.exit(f"group {name} has {nz:,} gravity zones, over {MAX_GROUP_ZONES:,}: split it (notes/T-539.md)")
    if set(g) & ne:
        if set(g) != ne or fixed:
            sys.exit(f"the Northeast's group is {'+'.join(g)}: {', '.join(sorted(set(g) ^ ne))} "
                     "would need a Northeast market place (docs/adding-a-city.md 6), or the Northeast "
                     "solved as a --packs group")
        if not dry:
            # the sites come from this tree's packs (sites() reads the module's registry)
            assert root == ROOT, "the Northeast is solved for real only in this tree"
            write_sites()
        joint(root, work, dry)
    else:
        solve_group(g, name, root, work, dry, fixed)
    print(f"group {name} ({len(g)} packs{f', {len(fixed)} fixed' if fixed else ''}: {' '.join(g)}) solved in "
          f"{time.time() - t0:.0f} s", flush=True)


def outline(path):
    """A GeoJSON file's polygons as one shape, lon/lat."""
    import shapely
    from shapely.geometry import shape
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = fc["features"] if fc.get("type") == "FeatureCollection" else [fc]
    return shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in feats])


def gap_km(a, b):
    """Shortest distance between two lon/lat shapes, in an equirectangular frame at their mean
    latitude (under 1% off at the few hundred km this is asked for)."""
    from shapely.ops import transform
    k = math.cos(math.radians((a.centroid.y + b.centroid.y) / 2))
    f = lambda x, y, z=None: (np.radians(x) * k * R_EARTH / 1000.0, np.radians(y) * R_EARTH / 1000.0)  # noqa: E731
    return transform(f, a).distance(transform(f, b))


def components(shapes, gap=None):
    """Groups of packs whose outlines lie within `gap` km of each other, chained (T-539)."""
    gap = GROUP_GAP_KM if gap is None else gap
    import shapely
    ids = list(shapes)
    tree = shapely.STRtree([shapes[i] for i in ids])
    parent = list(range(len(ids)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(ids):
        g = shapes[a]
        lat = max(abs(g.bounds[1]), abs(g.bounds[3]))
        dy = gap / 111.0
        dx = dy / max(math.cos(math.radians(min(lat + dy, 89.0))), 0.01)
        x0, y0, x1, y1 = g.bounds
        for j in tree.query(shapely.box(x0 - dx, y0 - dy, x1 + dx, y1 + dy)):
            if j > i and root(i) != root(j) and gap_km(g, shapes[ids[j]]) <= gap:
                parent[root(i)] = root(j)
    out = {}
    for i, a in enumerate(ids):
        out.setdefault(root(i), []).append(a)
    return list(out.values())


def auto_groups(reg, root=ROOT, plan=False, gap=None):
    """T-539: the joint gravity groups of the registered packs (registry order), each two or more
    packs; with `plan`, the world plan's planned metros join the grouping too (to see which
    groups a batch will grow or join), marked with a leading '+'."""
    shapes = {c["id"]: outline(root / f"data/packs/{c['id']}.boundary.geojson") for c in reg["cities"]}
    if plan:
        import shapely
        from shapely.geometry import shape
        ext = json.loads((root / "data/work/worldplan/extents.geojson").read_text(encoding="utf-8"))
        for f in ext["features"]:
            shapes.setdefault("+" + f["properties"]["id"], shapely.make_valid(shape(f["geometry"])))
    order = {c["id"]: k for k, c in enumerate(reg["cities"])}
    groups = [sorted(g, key=lambda i: (order.get(i, len(order)), i)) for g in components(shapes, gap)]
    groups = [g for g in groups if len(g) >= 2]
    return sorted(groups, key=lambda g: (order.get(g[0], len(order)), g[0]))


def cargo_bin(name, *args):
    env = dict(os.environ)
    env["PATH"] = os.path.join(os.path.expanduser("~"), ".cargo", "bin") + os.pathsep + env.get("PATH", "")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    # TW_SIM_DIR: a copy of sim/ to build from while the tree's own is mid-edit (as build_city.py),
    # with its own TW_CARGO_TARGET_DIR so the shared binaries are not overwritten
    env["CARGO_TARGET_DIR"] = os.environ.get("TW_CARGO_TARGET_DIR", str(ROOT / "data/work/target"))
    cargo = shutil.which("cargo", path=env["PATH"])
    if not cargo:
        sys.exit("cargo not found: install Rust (rustup) or put ~/.cargo/bin on PATH")
    sim = Path(os.environ.get("TW_SIM_DIR", ROOT / "sim"))
    subprocess.run([cargo, "run", "--quiet", "--release", "--features", "demand", "--bin", name,
                    "--manifest-path", str(sim / "Cargo.toml"), "--", *args], env=env, check=True)


def frame():
    """The frame the sites are in (T-128), as the registry writes frames."""
    return {"kind": "equirect", "lon": ORIGIN[0], "lat": ORIGIN[1]}


def write_sites():
    path = ROOT / "data/packs/intercity.json"
    site_rows, site_park = sites()
    data = {"format": 2, "origin": {"lon": ORIGIN[0], "lat": ORIGIN[1]}, "frame": frame(),
            "markets": MARKETS, "zone_m": 6000, "trip_rate": TRIP_RATE,
            "description": "The Northeast's endpoint sites and commutes between zones; the game reads each zone's market pack instead (T-377, pipeline/market_zones.py).",
            "sources": ["2020 Census block population less institutional group quarters (T-234)", "LODES 2023/2022 jobs", "registered zone-pack cells"],
            "sites": site_rows, "site_park": site_park}
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(path, len(site_rows), "sites", path.stat().st_size, "bytes")


def opt(name):
    return Path(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else None


if __name__ == "__main__":
    if "--groups" in sys.argv:
        for g in auto_groups(REGISTRY, plan="--plan" in sys.argv):
            print(f"{len(g)} packs: {' '.join(g)}")
        if "--plan" not in sys.argv:
            for u in solve_units(REGISTRY):
                print(f"solve {u['name']} ({u['country']}): {len(u['members'])} packs, {len(u['fixed'])} fixed")
        sys.exit()
    if "--solve" in sys.argv:  # one unit as world_run.py hands it: --solve a,b [--fixed c,d] --name n
        arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else ""  # noqa: E731
        solve_unit([i for i in arg("--solve").split(",") if i], [i for i in arg("--fixed").split(",") if i],
                   arg("--name"), dry="--dry" in sys.argv)
        sys.exit()
    if "--group" in sys.argv:
        i = sys.argv.index("--group")
        assert sys.argv[i + 1] == "auto", "--group takes auto (T-539)"
        rest = sys.argv[i + 2:]
        only = [a for k, a in enumerate(rest) if not a.startswith("--") and (k == 0 or rest[k - 1] not in ("--root", "--work"))]
        root = (opt("--root") or ROOT).resolve()
        dry = "--dry" in sys.argv
        ids = solve_auto(only, root, opt("--work"), dry)
        if not dry and root == ROOT and "--no-market" not in sys.argv:
            import market_zones
            market_zones.main(ids)
        sys.exit()
    write_sites()
    if "--no-joint" not in sys.argv[1:]:
        joint()
    import market_zones
    market_zones.main([])

