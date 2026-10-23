"""T-1004 dry run (read-only): the world plan's metro rule and zone rule applied to the US, in memory.
US units are the 2020 cartographic counties (data/raw/cb_2020_us_county_500k.zip), states the regions.
Writes only into this script's folder: dry_state.pkl, dry_rows.json, dry_zones_<floor>.json, logs."""
import collections
import json
import os
import pickle
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = r"C:\Users\anita\projects\maps\trainworld\pipeline"
sys.path.insert(0, PIPE)
import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402
from shapely import STRtree, contains_xy, union_all  # noqa: E402

import world_plan as wp  # noqa: E402

COUNTIES = r"C:\Users\anita\projects\maps\trainworld\data\raw\cb_2020_us_county_500k.zip"
TERR = {"60", "66", "69", "72", "78"}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def us_metros(pop, ne):
    fpath = wp.unzip(wp.FUA[0], wp.FUA[1], ".gpkg")
    fua = pyogrio.read_dataframe(fpath, columns=["eFUA_ID", "eFUA_name", "Cntry_ISO", "FUA_p_2015"])
    fgeom = [wp.polygonal(g) for g in fua.geometry]
    lay = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
    uc = pyogrio.read_dataframe(wp.UCDB, layer=lay, columns=["ID_UC_G0", "GC_UCN_MAI_2025", "GC_POP_TOT_2025"])
    cen = pyogrio.read_dataframe(wp.UCDB, layer="UC_centroids")
    cen = dict(zip(cen.ID_UC_G0, cen.geometry))
    uc_iso = wp.iso_at(ne, STRtree([g for _, g in ne]), [cen[i] for i in uc.ID_UC_G0])
    ftree = STRtree(fgeom)
    world = []
    for k in range(len(uc)):
        if uc_iso[k] != "USA" or uc.GC_POP_TOT_2025[k] < wp.MIN_POP:
            continue
        ug = wp.polygonal(uc.geometry[k])
        pt = cen[uc.ID_UC_G0[k]]
        joined = []
        for i in ftree.query(ug, predicate="intersects"):
            x, y, p = pop.inside(fgeom[i])
            share = p[contains_xy(ug, x, y)].sum() / max(p.sum(), 1)
            if fgeom[i].contains(pt) or share >= wp.FUA_SHARE:
                joined.append(int(i))
        world.append({"uc": int(uc.ID_UC_G0[k]), "name": uc.GC_UCN_MAI_2025[k], "iso": "USA",
                      "pop": float(uc.GC_POP_TOT_2025[k]), "geom": ug, "fua": joined})
    parent = list(range(len(world)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    owner = {}
    for k, w in enumerate(world):
        for i in w["fua"]:
            if i in owner:
                parent[root(k)] = root(owner[i])
            else:
                owner[i] = k
    groups = collections.defaultdict(list)
    for k in range(len(world)):
        groups[root(k)].append(world[k])
    out = []
    for ws in groups.values():
        ws.sort(key=lambda w: -w["pop"])
        fu = sorted({i for w in ws for i in w["fua"]})
        area = wp.polygonal(union_all([w["geom"] for w in ws] + [fgeom[i] for i in fu]))
        out.append({"key": f"uc{ws[0]['uc']}", "name": ws[0]["name"], "iso": "USA", "region": "world",
                    "source": {"ucdb": [w["uc"] for w in ws], "fua": [int(fua.eFUA_ID[i]) for i in fu]},
                    "alt_names": [w["name"] for w in ws[1:]] + [fua.eFUA_name[i] for i in fu], "area": area,
                    "ucpop": sum(w["pop"] for w in ws)})
    log(f"US: {len(world)} urban centres over 1M, {len(out)} metros after merging")
    # satellites: smaller US centres within SAT_GAP of a metro's own centres (world_plan.satellites, US only)
    ids = [int(i) for i in uc.ID_UC_G0]
    geom = [wp.polygonal(g) for g in uc.geometry]
    cpts = [cen[i] for i in uc.ID_UC_G0]
    atree = STRtree([m["area"] for m in out])
    pi, mi = atree.query(cpts, predicate="within")
    inside = set(pi.tolist())
    own = collections.defaultdict(list)
    for a, k in zip(pi, mi):
        if ids[a] in out[k]["source"]["ucdb"]:
            own[int(k)].append(geom[a])
    centres = [union_all(own[k]) if own[k] else shapely.Polygon() for k in range(len(out))]
    ctree = STRtree(centres)
    add = collections.defaultdict(list)
    for a in range(len(ids)):
        if a in inside or uc.GC_POP_TOT_2025[a] >= wp.MIN_POP or uc_iso[a] != "USA":
            continue
        near = ctree.query(geom[a], predicate="dwithin", distance=wp.SAT_GAP)
        if not len(near):
            continue
        k = int(min(near, key=lambda j: geom[a].distance(centres[j])))
        add[k].append(geom[a])
        out[k]["source"].setdefault("satellites", []).append(ids[a])
        out[k]["alt_names"].append(uc.GC_UCN_MAI_2025[a])
    for k, gs in add.items():
        out[k]["area"] = wp.polygonal(union_all([out[k]["area"]] + gs))
    log(f"satellites: {sum(len(m['source'].get('satellites', [])) for m in out)} join {len(add)} metros")
    for m in out:
        x, y, p = pop.inside(m["area"])
        m["px"] = (x, y, p)
        m["pop_area"] = float(p.sum())
        i = int(np.argmax(p))
        m["peak"] = shapely.Point(x[i], y[i])
        shapely.prepare(m["area"])
    return out


def us_units(pop):
    df = pyogrio.read_dataframe(f"zip://{COUNTIES}")
    units = []
    for r in df.itertuples():
        g = wp.polygonal(wp.to_moll(r.geometry))
        u = wp.Unit(r.GEOID, f"{r.NAMELSAD}, {r.STUSPS}", "PRI" if r.STATEFP == "72" else ("TERR" if r.STATEFP in TERR else "USA"),
                    g, [(r.STATEFP, r.STATE_NAME)])
        xs, ys, ps = [], [], []
        for part in shapely.get_parts(g):
            x, y, p = pop.inside(part)
            xs.append(x), ys.append(y), ps.append(p)
        x, y, p = np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)
        u.px = (x, y, p)
        u.pop = float(p.sum())
        u.free = u.pop
        units.append(u)
    log(f"{len(units)} counties, GHSL people {sum(u.pop for u in units) / 1e6:.1f}M")
    return units


def main():
    cache = os.path.join(HERE, "dry_state.pkl")
    pop = wp.Pop()
    if os.path.exists(cache):
        st = pickle.load(open(cache, "rb"))
        log("state from cache")
    else:
        ne = wp.countries()
        ms = us_metros(pop, ne)
        units = us_units(pop)
        pk = wp.packs()
        log(f"{len(pk)} packs")
        wp.shares(units, ms, list(pk.values()))
        near = collections.defaultdict(list)
        for u in units:
            for key in u.share:
                near[key].append(u)
        for m in ms:
            m["units_near"] = near[m["key"]]
        st = {"ms": ms, "units": units, "packs": pk}
        pickle.dump(st, open(cache, "wb"))
        log("state cached")
    ms, units, pk = st["ms"], st["units"], st["packs"]
    rows, dropped = wp.plan({"ms": ms, "units": units}, pk)
    by = {u.id: u for u in units}
    out = []
    for r in sorted(rows, key=lambda r: -r["people"]):
        states = collections.Counter()
        for u in r["units"]:
            states[u.parents[0][0]] += u.pop
        out.append({"name": r["name"], "people": round(r["people"]), "units": [u.id for u in r["units"]],
                    "states": [s for s, _ in states.most_common()], "lead_state": states.most_common(1)[0][0],
                    "split": r["split"], "ucpop": round(r["metro"]["ucpop"])})
    json.dump({"rows": out, "dropped": [(m["name"], why, round(m.get("ucpop", 0))) for m, why in dropped]},
              open(os.path.join(HERE, "dry_rows.json"), "w"), indent=1)
    log(f"{len(out)} metro rows, {sum(r['people'] for r in out) / 1e6:.1f}M; dropped {len(dropped)}")
    for m, why in dropped:
        log(f"  dropped {m['name']}: {why}")
    held = {uid for r in out for uid in r["units"]}
    for floor, cap in ((500_000, 1_500_000), (1_000_000, 2_500_000)):
        free, region = [], {}
        for u in units:
            region[u.id] = u.parents[0][0]
            if u.id in held or u.iso == "TERR":
                continue
            x, y, p = u.px
            inpk = np.zeros(len(x), bool)
            for pid in u.pshare:
                inpk |= contains_xy(pk[pid]["geom"], x, y)
            u.free = float(p[~inpk].sum())
            if (u.pop > 0 and u.free < wp.UNIT_SHARE * u.pop) or u.free <= 0:
                continue
            free.append(u.id)
        log(f"floor {floor:,}: {len(free)} free units, {sum(by[i].free for i in free) / 1e6:.1f}M")
        groups, lines = wp.zone_groups(units, free, region, floor, cap, contiguous=True)
        zs = []
        for lead, ids in groups.items():
            n = sum(by[i].free for i in ids)
            zs.append({"lead": lead, "name": by[max(ids, key=lambda i: by[i].free)].name, "people": round(n),
                       "units": ids, "state": region[lead], "km2": round(sum(by[i].geom.area for i in ids) / 1e6)})
        json.dump({"zones": zs, "lines": lines}, open(os.path.join(HERE, f"dry_zones_{floor}.json"), "w"), indent=1)
        log(f"floor {floor:,}: {len(zs)} zones")


if __name__ == "__main__":
    main()
