"""T-460: the land-use cap as tier 0's default, measured on London and Toronto. Read-only on the
packs and caches; writes land-area pickles and a result JSON in the artifacts folder.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t460_measure.py [extract] [grid]

"Misplaced" is half the L1 distance between two shares of jobs by 2 km zone (as notes/t360_jobs.py).
London: the tier 0 proxy (jobs_from on the signals) against BRES employment by LSOA, the BRES
spread inside each LSOA by the proxy of the signals before the change (a fixed reference), and also
re-spread by the changed proxy. Toronto: the proxy on t0tor's signals (data/work/t287) against the
real pack's jobs (StatCan place of work). The changed signals are the old ones with their OSM
columns rebuilt from land rows capped and spread, new cells added (checked: the old rows give the
old columns exactly).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import pickle
import sys

import h3
import numpy as np
import pandas as pd
from shapely import make_valid, union_all
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, HERE)
import build_city as b  # noqa: E402
import canada  # noqa: E402
import england  # noqa: E402
import overpass  # noqa: E402
import tier0  # noqa: E402
from read_pack import load_pack  # noqa: E402
import t360_jobs as t3  # noqa: E402

ART = os.path.join(ROOT, "..", "artifacts", "trainworld")
SCRATCH = os.environ.get("T460_SCRATCH", ART)
OUT = os.path.join(ART, "T-460-measure.json")
CITIES = {
    "lon": dict(extract="europe/united-kingdom/england", csv=tier0.raw("osm_jobs_lon.csv.gz"),
                signals=os.path.join(b.WORK, "lon.tier0_signals.npz"),
                boundary=lambda: union_all([make_valid(shape(f["geometry"])) for f in
                                            json.load(open(england.raw("lon_itl3.geojson"), encoding="utf-8"))["features"]])),
    "t0tor": dict(extract="north-america/canada/ontario", csv=os.path.join(canada.RAW_CA, "osm_jobs_tor.csv.gz"),
                  signals=os.path.join(ROOT, "data", "work", "t287", "work", "t0tor.tier0_signals.npz"),
                  boundary=lambda: make_valid(shape(json.load(open(os.path.join(ROOT, "data", "packs", "tor.boundary.geojson"),
                                                                  encoding="utf-8"))["features"][0]["geometry"]))),
}


def land_areas(city):
    """[(geometry, m2, ref, name)] of the land areas osm_places counts, from the cached features."""
    pk = os.path.join(SCRATCH, f"T-460-land-{city}.pkl")
    if os.path.exists(pk):
        return pickle.load(open(pk, "rb"))
    import osm_features
    c = CITIES[city]
    boundary = c["boundary"]()
    w, s, e, n = boundary.bounds
    cols = ["shop", "amenity", "office", "craft", "tourism", "leisure", "landuse"]
    mp = osm_features.read(c["extract"], "multipolygons", columns=cols + ["osm_id", "osm_way_id", "name"], bbox=(w, s, e, n),
                           where=" OR ".join(f"{k} IS NOT NULL" for k in cols))
    out = []
    for r in mp.itertuples():
        if r.landuse in ("industrial", "commercial", "retail") and not (r.shop or r.amenity or r.office):
            out.append((r.geometry, overpass.area_m2(r.geometry), canada._osm_ref(r), r.name if isinstance(r.name, str) else ""))
    pickle.dump(out, open(pk, "wb"))
    print(f"{city}: {len(out):,} land areas, total {sum(a[1] for a in out) / 1e6:,.0f} km2, "
          f"largest {max(a[1] for a in out) / 1e6:.1f} km2", flush=True)
    return out


def places_for(city, cap_km2, spread):
    """The city's place rows: the cached non-land rows plus land rows under the settings."""
    old = pd.read_csv(CITIES[city]["csv"])
    keep = old[old.kind != "land"][["kind", "lat", "lon", "w"]]
    rows = []
    for geom, m2, ref, name in land_areas(city):
        if cap_km2:
            m2 = min(m2, cap_km2 * 1e6)
        rows += canada._land_rows(geom, m2 / canada.LANDUSE_M2_PER_PLACE, spread, ref, name)
    land = pd.DataFrame(rows, columns=["kind", "lat", "lon", "w", "ref", "name"])
    return old, pd.concat([keep, land[["kind", "lat", "lon", "w"]]], ignore_index=True), land


def signals_with(city, base, boundary, places):
    """base (the cached signals) with its OSM columns rebuilt from `places`, new cells added."""
    pids = np.fromiter((h3.str_to_int(h3.latlng_to_cell(la, lo, 9)) for la, lo in zip(places.lat, places.lon)), np.uint64, len(places))
    new = np.setdiff1d(np.unique(pids), base["h3"])
    if len(new):
        ok, ll = tier0.inside(new, boundary)
        new, ll = new[ok], ll[ok]
    else:
        ll = np.zeros((0, 2))
    cells = np.concatenate([base["h3"], new])
    sig = {"h3": cells, "lat": np.concatenate([base["lat"], ll[:, 0]]), "lon": np.concatenate([base["lon"], ll[:, 1]])}
    for k in ("pop", "nres_v", "nres_s", "pop_ghsl"):
        sig[k] = np.concatenate([base[k], np.zeros(len(new))])
    index = {int(c): i for i, c in enumerate(cells)}
    pk = places.kind.to_numpy()
    w_all = places.w.to_numpy()
    for kind in ("errand", "leisure", "job", "land"):
        sig["osm_" + kind] = np.zeros(len(cells))
        sel = pk == kind
        rows = np.array([index.get(int(c), -1) for c in pids[sel]], np.int64)
        np.add.at(sig["osm_" + kind], rows[rows >= 0], w_all[sel][rows >= 0])
    return sig


def zone_totals(sig, v, origin, keys):
    zk = t3.zone_keys(sig["lat"], sig["lon"], origin)
    return np.bincount(np.searchsorted(keys, zk), weights=v, minlength=len(keys))


def top_cell(sig, v):
    i = int(np.argmax(v))
    return float(v[i] / v.sum()), (round(float(sig["lon"][i]), 4), round(float(sig["lat"][i]), 4))


def run_city(city, grid, full):
    c = CITIES[city]
    boundary = c["boundary"]()
    base = dict(np.load(c["signals"]))
    origin = b.CITIES["lon" if city == "lon" else "tor"]["origin"]
    # the old rows must reproduce the cached columns
    old, again, _ = places_for(city, None, None)
    chk = signals_with(city, base, boundary, again)
    same = all(np.array_equal(chk["osm_" + k][:len(base["h3"])], base["osm_" + k]) for k in ("errand", "leisure", "job", "land")) and len(chk["h3"]) >= len(base["h3"])
    print(f"{city}: old rows reproduce the cached OSM columns exactly: {same} ({len(chk['h3']) - len(base['h3'])} extra cells); "
          f"cached land {base['osm_land'].sum():,.0f} units, rebuilt {chk['osm_land'].sum():,.0f}", flush=True)
    assert same or city == "t0tor"  # Toronto's cached rows are from an older Ontario extract than today's

    if city == "lon":
        lad_sig = base
        ref0 = england.bres_jobs("lon", boundary, base, tier0.jobs_from(base, 1.0))[0]
        ref_fixed = lambda keys: zone_totals(base, ref0, origin, keys)
    else:
        h, a = load_pack("tor", os.path.join(ROOT, "data", "packs"))
        ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(x))) for x in a["h3"]])
        pk_sig = {"lat": ll[:, 0], "lon": ll[:, 1]}
        ref_fixed = lambda keys: zone_totals(pk_sig, a["jobs"].astype(float), origin, keys)

    res = []
    for cap, spread in grid:
        _, places, land = places_for(city, cap, spread)
        sig = signals_with(city, base, boundary, places)
        v = tier0.jobs_from(sig, 1.0)
        zk = np.union1d(t3.zone_keys(sig["lat"], sig["lon"], origin), t3.zone_keys(base["lat"], base["lon"], origin))
        if city != "lon":
            zk = np.union1d(zk, t3.zone_keys(pk_sig["lat"], pk_sig["lon"], origin))
        row = dict(cap_km2=cap, spread_deg=spread, zone_misplaced=t3.miss(zone_totals(sig, v, origin, zk), ref_fixed(zk)))
        if city == "lon" and full:
            r1 = england.bres_jobs("lon", boundary, sig, v)[0]
            row["zone_misplaced_respread"] = t3.miss(zone_totals(sig, v, origin, zk), zone_totals(sig, r1, origin, zk))
        share, where = top_cell(sig, v)
        row.update(top_cell_share=share, top_cell=where, land_units=float(sig["osm_land"].sum()), osm_units=float(sum(sig["osm_" + k].sum() for k in ("errand", "leisure", "job", "land"))))
        g = land.groupby("ref").w.sum()
        row.update(top_place=str(g.idxmax()), top_place_osm_share=float(g.max() / row["osm_units"]))
        res.append(row)
        print(f"{city} cap {cap} spread {spread}: " + ", ".join(f"{k} {v_:.5f}" if isinstance(v_, float) else f"{k} {v_}" for k, v_ in row.items() if k not in ("cap_km2", "spread_deg")), flush=True)
    return res


def main(args):
    default = [(None, None), (1.0, 0.003)]
    grid = default
    if "grid" in args:
        grid = default + [(cap, sp) for cap in (0.25, 0.5, 1.0, 2.0, 4.0, None) for sp in (None, 0.002, 0.003, 0.005) if (cap, sp) not in default]
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for city in [a for a in args if a in CITIES] or ("lon", "t0tor"):
        out[city] = run_city(city, grid, full=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
