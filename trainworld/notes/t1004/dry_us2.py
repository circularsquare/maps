"""T-1004 dry run, variant B: US metros as Europe's are planned, GHS-FUAs over 1M (world_plan.metros'
European branch), satellites as Europe's. Units from dry_us.py's cache (dry_state.pkl). Read-only."""
import collections
import json
import os
import pickle
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402
from shapely import STRtree, contains_xy, union_all  # noqa: E402

import world_plan as wp  # noqa: E402
import dry_us  # noqa: E402

t0 = time.time()
FLOORS = [(500_000, 1_500_000), (1_000_000, 2_500_000)]


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def fua_metros(pop, ne):
    fpath = wp.unzip(wp.FUA[0], wp.FUA[1], ".gpkg")
    fua = pyogrio.read_dataframe(fpath, columns=["eFUA_ID", "eFUA_name", "Cntry_ISO", "FUA_p_2015"])
    out = []
    for i in range(len(fua)):
        if fua.Cntry_ISO[i] == "USA" and fua.FUA_p_2015[i] >= 0.5 * wp.MIN_POP:
            g = wp.polygonal(fua.geometry[i])
            _, _, p = pop.inside(g)
            if p.sum() >= wp.MIN_POP:
                out.append({"key": f"fua{int(fua.eFUA_ID[i])}", "name": fua.eFUA_name[i], "iso": "USA", "region": "europe",
                            "source": {"fua": [int(fua.eFUA_ID[i])]}, "alt_names": [], "area": g, "ucpop": float(p.sum())})
    log(f"US: {len(out)} FUAs over 1M")
    lay = "GHS_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
    uc = pyogrio.read_dataframe(wp.UCDB, layer=lay, columns=["ID_UC_G0", "GC_UCN_MAI_2025", "GC_POP_TOT_2025"])
    cen = pyogrio.read_dataframe(wp.UCDB, layer="UC_centroids")
    cen = dict(zip(cen.ID_UC_G0, cen.geometry))
    uc_iso = wp.iso_at(ne, STRtree([g for _, g in ne]), [cen[i] for i in uc.ID_UC_G0])
    ids = [int(i) for i in uc.ID_UC_G0]
    geom = [wp.polygonal(g) for g in uc.geometry]
    cpts = [cen[i] for i in uc.ID_UC_G0]
    atree = STRtree([m["area"] for m in out])
    pi, mi = atree.query(cpts, predicate="within")
    inside = set(pi.tolist())
    own = collections.defaultdict(list)
    for a, k in zip(pi, mi):
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
        out[k]["alt_names"].append(uc.GC_UCN_MAI_2025[a])
    for k, gs in add.items():
        out[k]["area"] = wp.polygonal(union_all([out[k]["area"]] + gs))
    log(f"satellites join {len(add)} metros")
    for m in out:
        x, y, p = pop.inside(m["area"])
        m["px"] = (x, y, p)
        m["pop_area"] = float(p.sum())
        i = int(np.argmax(p))
        m["peak"] = shapely.Point(x[i], y[i])
        shapely.prepare(m["area"])
    return out


def main():
    st = pickle.load(open(os.path.join(HERE, "dry_state.pkl"), "rb"))
    units, pk = st["units"], st["packs"]
    pop = wp.Pop()
    ms = fua_metros(pop, wp.countries())
    for u in units:
        u.share = {}
        u.pshare = {}
    wp.shares(units, ms, list(pk.values()))
    near = collections.defaultdict(list)
    for u in units:
        for key in u.share:
            near[key].append(u)
    for m in ms:
        m["units_near"] = near[m["key"]]
    rows, dropped = wp.plan({"ms": ms, "units": units}, pk)
    by = {u.id: u for u in units}
    out = []
    for r in sorted(rows, key=lambda r: -r["people"]):
        states = collections.Counter()
        for u in r["units"]:
            states[u.parents[0][0]] += u.pop
        out.append({"name": r["name"], "people": round(r["people"]), "units": [u.id for u in r["units"]],
                    "unit_names": [u.name for u in r["units"]],
                    "states": [s for s, _ in states.most_common()], "lead_state": states.most_common(1)[0][0],
                    "split": r["split"], "fua_pop": round(r["metro"]["ucpop"])})
    json.dump({"rows": out, "dropped": [(m["name"], why, round(m.get("ucpop", 0))) for m, why in dropped]},
              open(os.path.join(HERE, "dry2_rows.json"), "w"), indent=1)
    log(f"{len(out)} metro rows, {sum(r['people'] for r in out) / 1e6:.1f}M; dropped {len(dropped)}")
    held = {uid for r in out for uid in r["units"]}
    for floor, cap in FLOORS:
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
        json.dump({"zones": zs, "lines": lines}, open(os.path.join(HERE, f"dry2_zones_{floor}.json"), "w"), indent=1)
        log(f"floor {floor:,}: {len(zs)} zones")


if __name__ == "__main__":
    main()
