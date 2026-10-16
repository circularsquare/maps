"""T-420 stage 3: Japan's prefecture zones, planned, fitted and built as one batch.

    python pipeline/japan_zones.py plan               write pipeline/japan_zones.json (needs the fetched inputs)
    python pipeline/japan_zones.py build <id>         one zone: outline, curve, pack, places, routes, merge, checks
    python pipeline/japan_zones.py run [<id> ...] [--jobs 3]    build zones in parallel, one process each
    python pipeline/japan_zones.py fold               the fitted curves into japan_zones.json

**Zones.** Every prefecture outside the seven metro packs is one zone; Hokkaido's subprefectures
outside Sapporo's pack (Ishikari and Otaru) are units of which any under 100,000 people joins the
neighbouring zone it shares the longest border with (docs/zones.md 2), smallest first. A zone is
named after its most populous unit; its origin is the centre of its most populous 500 m mesh.

**Per zone.** As the metro packs (notes/T-420.md): census mesh population, economic census jobs
scaled to the census's workers by place of work, P29 schools, OSM from the Geofabrik region
extract whose .poly holds the outline (water from every region near the box). Work at home: the
census's at-home share by municipality of work plus MLIT's 16.8% telework outside the three big
metros at 2 days in 5 (`wfh` in the JSON is the zone's mean, for reference). The
curve is fitted on the census's municipality flows with the pack's own margins over a fixed grid
(fit below), the lowest misplaced share, ties within 0.3 points going to the own-municipality
share closest to the census's. A box over 150M pixels at 25 m takes 50 m water pixels.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_city as b  # noqa: E402
import japan  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "japan_zones.json")
ART = os.path.join(ROOT, "..", "artifacts", "trainworld")
PY = sys.executable

PREFS = {"02": "Aomori", "03": "Iwate", "05": "Akita", "06": "Yamagata", "07": "Fukushima", "09": "Tochigi",
         "10": "Gunma", "15": "Niigata", "16": "Toyama", "17": "Ishikawa", "18": "Fukui", "19": "Yamanashi",
         "20": "Nagano", "22": "Shizuoka", "25": "Shiga", "30": "Wakayama", "31": "Tottori", "32": "Shimane",
         "33": "Okayama", "35": "Yamaguchi", "36": "Tokushima", "37": "Kagawa", "38": "Ehime", "39": "Kochi",
         "41": "Saga", "42": "Nagasaki", "43": "Kumamoto", "44": "Oita", "45": "Miyazaki", "46": "Kagoshima",
         "47": "Okinawa"}
IDS = {"02": "aom", "03": "iwt", "05": "akt", "06": "ygt", "07": "fks", "09": "tcg", "10": "gnm", "15": "nig",
       "16": "tym", "17": "isk", "18": "fki", "19": "yns", "20": "ngn", "22": "szo", "25": "shg", "30": "wky",
       "31": "ttr", "32": "smn", "33": "oky", "35": "ymg", "36": "tks", "37": "kgw", "38": "ehm", "39": "kch",
       "41": "sag", "42": "ngs", "43": "kmm", "44": "oit", "45": "mzk", "46": "kgs", "47": "oka",
       "01-kamikawa": "kmk", "01-sorachi": "sor", "01-iburi": "ibr", "01-rumoi": "rmi", "01-soya": "soy",
       "01-okhotsk": "okh", "01-tokachi": "tkc", "01-kushiro": "ksr", "01-nemuro": "nmr", "01-hidaka": "hdk",
       "01-oshima": "osh", "01-hiyama": "hym", "01-shiribeshi~01203": "shb"}
HOKKAIDO = {"01-kamikawa": "Kamikawa", "01-sorachi": "Sorachi", "01-iburi": "Iburi", "01-rumoi": "Rumoi",
            "01-soya": "Soya", "01-okhotsk": "Okhotsk", "01-tokachi": "Tokachi", "01-kushiro": "Kushiro",
            "01-nemuro": "Nemuro", "01-hidaka": "Hidaka", "01-oshima": "Oshima", "01-hiyama": "Hiyama",
            "01-shiribeshi~01203": "Shiribeshi"}
# the metro packs' units: neighbours only
METRO = ["08", "11", "12", "13", "14", "26", "27", "28", "29", "21", "23", "24", "40", "04", "34", "01-ishikari", "01-01203"]
REGIONS = ["hokkaido", "tohoku", "kanto", "chubu", "kansai", "chugoku", "shikoku", "kyushu"]
MIN_POP = 100_000
TELEWORK = b.TELEWORK_JP
GRID = [(pw, km) for pw in (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4) for km in (16.0, 32.0, 50.0, 80.0, 160.0)]


def _poly(region):
    """A Geofabrik region's .poly as a shapely geometry."""
    from shapely.geometry import Polygon
    from shapely import union_all
    path = japan._get(f"https://download.geofabrik.de/asia/japan/{region}.poly", os.path.join(b.RAW, "osm", f"asia_japan_{region}.poly"))
    rings, cur, hole = [], None, False
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        t = line.strip()
        if not t:
            continue
        if t == "END":
            if cur is not None:
                rings.append((hole, cur))
                cur = None
            continue
        if cur is None:
            cur, hole = [], t.startswith("!")
            continue
        x, y = t.split()[:2]
        cur.append((float(x), float(y)))
    outer = union_all([Polygon(r) for h, r in rings if not h])
    holes = [Polygon(r) for h, r in rings if h]
    return outer.difference(union_all(holes)) if holes else outer


def _unit_geom(u):
    from shapely import union_all
    return union_all([m[3] for m in japan.municipalities([u])]).buffer(0)


def plan():
    import shapely
    from shapely import union_all
    units = list(PREFS) + list(HOKKAIDO)
    geom = {u: _unit_geom(u) for u in units + METRO}
    simple = {u: g.simplify(0.002) for u, g in geom.items()}
    t = japan.mesh_table(japan.POP_MESH, sorted({c for g in geom.values() for c in japan.first_meshes(g.bounds)}))
    codes = t.index.to_numpy(np.int64)
    lat, lon = japan.mesh_sw(codes)
    clat, clon = lat + 1 / 480, lon + 1 / 320
    people = t[japan.POP_MESH + "001"].to_numpy(float)
    pop, top = {}, {}
    for u in units:
        g = geom[u]
        shapely.prepare(g)
        w, s, e, n = g.bounds
        sel = np.flatnonzero((clon >= w) & (clon <= e) & (clat >= s) & (clat <= n))
        sel = sel[shapely.contains_xy(g, clon[sel], clat[sel])]
        pop[u] = float(people[sel].sum())
        i = sel[np.argmax(people[sel])]
        top[u] = (float(people[i]), round(float(clon[i]), 4), round(float(clat[i]), 4))
        print(f"{u}: {pop[u]:,.0f} people (mesh centres); top mesh {top[u][0]:,.0f}", flush=True)

    def border(a, c):
        ga = union_all([simple[x] for x in a])
        gc = union_all([simple[x] for x in c])
        return ga.boundary.intersection(gc.buffer(0.01)).length

    zones = {u: [u] for u in units}
    while True:
        small = sorted((sum(pop[x] for x in us), z) for z, us in zones.items() if z in HOKKAIDO and sum(pop[x] for x in us) < MIN_POP)
        if not small:
            break
        _, z = small[0]
        cands = [(border(zones[z], zones[o]), o) for o in zones if o != z and o in HOKKAIDO]
        length, into = max(cands)
        assert length > 0, f"{z} touches no Hokkaido zone"
        print(f"merge {z} ({sum(pop[x] for x in zones[z]):,.0f}) into {into} (shared border {length:.3f} deg)", flush=True)
        zones[into] += zones.pop(z)
    polys = {r: _poly(r) for r in REGIONS}
    out = {}
    for z, us in zones.items():
        us = sorted(us, key=lambda x: -pop[x])
        g = union_all([geom[x] for x in us])
        gs = union_all([simple[x] for x in us])
        lead = us[0]
        zid = IDS[lead]
        name = PREFS.get(lead) or HOKKAIDO[lead]
        best = max(us, key=lambda x: top[x][0])
        nb = [x for x in units + METRO if x not in us and simple[x].distance(gs) < 0.02]
        inside = [r for r in REGIONS if polys[r].buffer(0.01).contains(gs)]
        extract = inside[0] if inside else None
        w, s, e, n = g.bounds
        box = shapely.box(w - 0.05, s - 0.05, e + 0.05, n + 0.05)
        near = [r for r in REGIONS if polys[r].intersects(box)]
        if extract is None:
            extract = [r for r in near if polys[r].intersects(gs)]
        k = math.cos(math.radians((s + n) / 2))
        px = ((e - w) * 111.32 * k + 4) * ((n - s) * 111.32 + 4) / 0.025 ** 2
        allw, home = japan.workers_at_work(us)
        h = home
        out[zid] = {"name": name, "units": us, "population": round(sum(pop[x] for x in us)),
                    "neighbours": nb, "origin": [top[best][1], top[best][2]],
                    "extract": f"asia/japan/{extract}" if isinstance(extract, str) else [f"asia/japan/{r}" for r in extract],
                    "water_extracts": [f"asia/japan/{r}" for r in near],
                    "water_cell_m": 50.0 if px > 1.5e8 else 25.0,
                    "census_at_home": round(h, 4), "wfh": round(h + (1 - h) * TELEWORK, 3)}
        print(f"{zid} {name}: {us}, {out[zid]['population']:,} people, extract {out[zid]['extract']}, water {near}, "
              f"{px / 1e6:.0f}M px at 25 m, wfh {out[zid]['wfh']}, neighbours {nb}", flush=True)
    doc = {"task": "T-420 stage 3", "rule": __doc__.split("**Zones.**")[1].split("**Per zone.**")[0].strip(),
           "unit_population": {u: round(p) for u, p in pop.items()}, "zones": out}
    if os.path.exists(OUT):  # keep fitted curves
        old = json.load(open(OUT, encoding="utf-8"))["zones"]
        for zid, z in out.items():
            for k in ("decay_pow", "decay_km", "curve_fit"):
                if k in old.get(zid, {}):
                    z[k] = old[zid][k]
    with open(OUT + ".part", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(OUT + ".part", OUT)
    print(f"{len(out)} zones -> {OUT}")


def prefetch_ghsl():
    """Every GHSL tile the zones read, before parallel builds (tile downloads share files)."""
    import tier0
    for zid in json.load(open(OUT, encoding="utf-8"))["zones"]:
        cfg = b.CITIES[zid]
        bounds = tier0.build_boundary(zid, cfg).bounds
        for kind in ("pop", "nres_v", "nres_s"):
            tier0.ghsl_window(kind, bounds, 2020, pad_m=60000.0 if kind == "pop" else 0.0)
        print(f"{zid}: GHSL tiles cached", flush=True)


def miss(a, c):
    a, c = np.asarray(a, float), np.asarray(c, float)
    return 0.5 * float(np.abs(a / a.sum() - c / c.sum()).sum())


def fit(zid, cfg, boundary, sig):
    """The zone's curve on the census's municipality flows with the pack's own margins (as
    notes/t420_curve.py, pack margins only, GRID): (pow, km, record)."""
    import england
    import tier0
    units, nb = cfg["prefectures"], cfg["neighbours"]
    lon0, lat0 = cfg["origin"]
    sx = b.R_EARTH * np.radians(sig["lon"] - lon0) * np.cos(np.radians(lat0))
    sy = b.R_EARTH * np.radians(sig["lat"] - lat0)
    ms = japan.municipalities(units)
    nl = len(ms)
    ci = {m[0]: i for i, m in enumerate(ms)}
    li, near = england.cell_units(sig["lat"], sig["lon"], [(m[0], m[3]) for m in ms])
    pop, _ = japan.mesh_population(zid, boundary, sig, units, nb)
    jobs, _ = japan.census_jobs(zid, boundary, sig, tier0.jobs_from(sig, 1.0), units, nb)
    h = cfg["at_home"](sig["lat"], sig["lon"])
    work_c = jobs * (1 - h) * (1 - cfg["telework"])
    home_c = pop * work_c.sum() / pop.sum()
    li = np.where(li < 0, 0, li)
    for u, p in near.items():
        li[p] = u
    fl = japan.flows(units)
    fl = fl[fl.h.isin(ci) & fl.w.isin(ci)]
    zone, gx, gy = tier0.zones_of(sx, sy)
    nz = len(gx)
    d = tier0.zone_distances(gx, gy)
    h_i, w_i, n = fl.h.map(ci).to_numpy(), fl.w.map(ci).to_numpy(), fl.n.to_numpy(float)
    Cl = np.zeros((nl, nl))
    np.add.at(Cl, (h_i, w_i), n)
    H = np.bincount(zone, weights=home_c, minlength=nz)
    J = np.bincount(zone, weights=work_c, minlength=nz)
    P = np.zeros((nz, nl)); Q = np.zeros((nz, nl))
    np.add.at(P, (zone, li), home_c); np.add.at(Q, (zone, li), work_c)
    P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    Q /= np.maximum(Q.sum(1, keepdims=True), 1e-12)
    own_census = float(np.trace(Cl) / Cl.sum())
    rows = []
    for pw, km in GRID:
        T = tier0.gravity(H, J, d, pw, km)
        C = P.T @ T.astype(np.float64) @ Q
        mean, _ = tier0.trip_stats(T, d)
        rows.append({"pow": pw, "km": km, "misplaced": miss(C, Cl), "own": float(np.trace(C) / C.sum()), "mean_km": mean})
    lo = min(r["misplaced"] for r in rows)
    best = min((r for r in rows if r["misplaced"] <= lo + 0.003), key=lambda r: abs(r["own"] - own_census))
    rec = {"commuters": float(n.sum()), "municipalities": nl, "zones": nz, "own_census": own_census,
           "best": {k: round(v, 4) for k, v in best.items()}, "lowest_misplaced": round(lo, 4)}
    print(f"curve: {n.sum():,.0f} census commuters with both ends inside, {nl} municipalities, {nz} zones; census own "
          f"{own_census:.1%}; chosen d^-{best['pow']} exp(-d/{best['km']:g}) misplaced {best['misplaced']:.1%} "
          f"(lowest {lo:.1%}), own {best['own']:.1%}, mean {best['mean_km']:.2f} km", flush=True)
    return best["pow"], best["km"], rec


def build(zid):
    """One zone, start to finish, in this process and its children; stops at the first failure."""
    import tier0
    t0 = time.time()
    cfg = b.CITIES[zid]
    japan.outline(zid)
    boundary = tier0.build_boundary(zid, cfg)
    memo = {}
    orig = tier0.build_water

    def water_once(c, bd):
        if "m" not in memo:
            memo["m"] = orig(c, bd)
        return memo["m"]
    tier0.build_water = water_once
    curve = os.path.join(b.WORK, f"{zid}.jp_curve.json")
    if "decay_pow" in json.load(open(OUT, encoding="utf-8"))["zones"][zid]:
        pass
    elif os.path.exists(curve):
        c = json.load(open(curve))
        cfg["decay_pow"], cfg["decay_km"] = c["pow"], c["km"]
    else:
        cache = os.path.join(b.WORK, f"{zid}.tier0_signals.npz")
        mask = None if os.path.exists(cache) else water_once(cfg, boundary)[0]
        sig = tier0.signals(zid, cfg, boundary, mask)
        pw, km, rec = fit(zid, cfg, boundary, sig)
        cfg["decay_pow"], cfg["decay_km"] = pw, km
        with open(curve, "w") as f:
            json.dump({"pow": pw, "km": km, **rec}, f, indent=1)
    print(f"[{zid}] curve d^-{cfg['decay_pow']} exp(-d/{cfg['decay_km']}) at {time.time() - t0:.0f} s", flush=True)
    tier0.main(zid)
    print(f"[{zid}] pack at {time.time() - t0:.0f} s", flush=True)
    ex = cfg["extract"] if isinstance(cfg["extract"], list) else [cfg["extract"]]
    steps = [["places.py", zid],
             ["routes.py", zid, "--extracts", ",".join(ex), "--workers", "1", "--keep-pbf"],
             ["merge_cells.py", zid],
             ["read_pack.py", zid],
             ["look_water.py", zid, os.path.join(ART, f"T-420s3-{zid}-water.png")]]
    for s in steps:
        subprocess.run([PY, "-u", os.path.join(HERE, s[0]), *s[1:]], check=True)
        print(f"[{zid}] {s[0]} at {time.time() - t0:.0f} s", flush=True)
    print(f"[{zid}] done in {time.time() - t0:.0f} s", flush=True)


def run(ids, jobs):
    """Each zone's build in its own process, `jobs` at a time; a log each, a status line each."""
    from concurrent.futures import ThreadPoolExecutor
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               RAYON_NUM_THREADS="1", CARGO_BUILD_JOBS="1", PYTHONIOENCODING="utf-8")
    status = os.path.join(ART, "T-420s3-status.txt")

    def one(zid):
        t = time.time()
        with open(os.path.join(ART, f"T-420s3-{zid}.log"), "w", encoding="utf-8") as log:
            r = subprocess.run([PY, "-u", __file__, "build", zid], stdout=log, stderr=subprocess.STDOUT, env=env)
        line = f"{zid} {'ok' if r.returncode == 0 else 'FAILED'} {time.time() - t:.0f} s"
        with open(status, "a") as f:
            f.write(line + "\n")
        return line
    with ThreadPoolExecutor(jobs) as pool:
        for line in pool.map(one, ids):
            print(line, flush=True)


def fold():
    """Each zone's fitted curve (data/work/<id>.jp_curve.json) into japan_zones.json."""
    doc = json.load(open(OUT, encoding="utf-8"))
    for zid, z in doc["zones"].items():
        p = os.path.join(b.WORK, f"{zid}.jp_curve.json")
        if os.path.exists(p):
            c = json.load(open(p))
            z["decay_pow"], z["decay_km"] = c["pow"], c["km"]
            z["curve_fit"] = {"misplaced": c["best"]["misplaced"], "own": c["best"]["own"], "own_census": round(c["own_census"], 4),
                              "commuters": round(c["commuters"]), "mean_km": c["best"]["mean_km"]}
    with open(OUT + ".part", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(OUT + ".part", OUT)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "plan":
        plan()
    elif cmd == "ghsl":
        prefetch_ghsl()
    elif cmd == "build":
        build(args[0])
    elif cmd == "run":
        jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 3
        ids = [a for a in args if not a.startswith("--") and not a.isdigit()] or list(json.load(open(OUT, encoding="utf-8"))["zones"])
        run(ids, jobs)
    elif cmd == "fold":
        fold()
