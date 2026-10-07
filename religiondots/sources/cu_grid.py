"""Cuba: the placement layer, Kontur 400 m population hexagons keyed to province.

Writes data/geo/cu/cu_hexes.gpkg. Copied in shape from `sources/so_grid.py`; `sources/cu.md` §5 is
the record.

THE JOIN IS ON HEX CENTROIDS. A hex whose centroid is outside every province is snapped to the
nearest one within `SNAP_KM` when it holds people (coast, cays), and dropped beyond it.

THE GUANTÁNAMO BAY NAVAL BASE IS LEFT OUT. The United States administers it (spec §14.18: disputed
land goes to its de facto administrator), ONEI does not count the people on it, and Natural Earth
draws it as its own feature (`USG`), outside `CUB`. COD-AB, from GADM, draws it inside Guantánamo
province, so every hex whose centroid is inside Natural Earth's `USG` is dropped before the join.
The people dropped are printed.

THREE CHECKS AGAINST ONEI's 2024 COUNT:

  * **the national ratio**, Kontur over ONEI, inside `EXPECTED_RATIO` +/- `TOLERANCE`; Kontur 2023
    is built on population figures from before the emigration of 2021-2024, so it reads high;
  * **the rank witness for the join**: Kontur people per province against ONEI, Spearman, against
    `N_PERM` shuffles (at most `PERM_P` of them may reach it);
  * **no town lost**: every GeoNames seat of `SEAT_MIN_POP` or more, Kontur within 5 and 10 km, a
    hole counted only in a province under `LOW_RATIO` of its share (Morocco's gate).

Usage:
    python sources/cu_grid.py --fetch    Kontur CU and GeoNames CU.zip
    python sources/cu_grid.py            rebuild from data/raw/cu/
"""

import gzip
import json
import os
import shutil
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # kontur_cap
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cu")
GEO = os.path.join(ROOT, "data", "geo", "cu")
UNITS = os.path.join(GEO, "cu_provinces.gpkg")
OUT = os.path.join(GEO, "cu_hexes.gpkg")
NE = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_CU_20231101.gpkg.gz")
GZ_NAME = "kontur_population_CU_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_CU_20231101.gpkg"
GEONAMES_URL = "https://download.geonames.org/export/dump/CU.zip"
GEONAMES = os.path.join(RAW, "geonames_CU.zip")

EXPECTED_UNITS = 16
EXPECTED_MUNIS = 168
MUNI_GPKG = os.path.join(GEO, "cu_municipalities.gpkg")
MUNI_CSV = os.path.join(GEO, "cu_municipalities.csv")
HOLE_FACTOR = 10.0          # sources/cd_geo.py's bar: a municipality under 1/10 of the national ratio
MUNI_HOLES = set()          # is one Kontur has lost; none in Cuba (Granma's are 0.16-0.19, uniform)
# Raw Kontur blocks reaching the cap, (lon, lat) of the peak -> (action, name). Filled after review.
# Both 2026-10-03 blocks are off Havana's real core: the densest real place is Centro Habana
# (GeoNames 158,151 at -82.3666, 23.1330; about 46,000/km2 on ONEI's figures), where Kontur does not
# reach the cap. One is east of the harbour in COD's La Habana Vieja and San Miguel del Padrón,
# 106,329 people in 5 hexes, nearest named places Nalón and La Narcisa; the other is between
# Marianao and La Lisa, 162,318 people in 9 hexes, where ONEI puts Marianao's 115,623 people on
# 21.7 km2. Both capped to the 3 km ring's median.
BLOCKS = {
    (-82.2939, 23.1155): ("capped", "east of Havana harbour (COD La Habana Vieja / San Miguel del Padrón)"),
    (-82.4261, 23.0702): ("capped", "Marianao / La Lisa (Havana)"),
}
EXPECTED_RATIO = 1.15
TOLERANCE = 0.20
N_PERM = 20_000
PERM_P = 0.001
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)

SEAT_MIN_POP = 50_000
HOLE_KM = 5.0
WIDE_KM = 10.0
HOLE_RATIO = 0.10
LOW_RATIO = 0.5
KONTUR_HOLES = set()

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
GEONAMES_COLS = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass",
                 "fcode", "cc", "cc2", "admin1", "admin2", "admin3", "admin4", "population",
                 "elevation", "dem", "timezone", "modified"]
METRIC = "EPSG:32617"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GEONAMES):
        req = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if data[:2] != b"PK":
            raise SystemExit(f"{GEONAMES_URL} is not a zip")
        with open(GEONAMES, "wb") as fh:
            fh.write(data)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 1_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def ne_feature(a3, crs):
    import geopandas as gpd
    from shapely.geometry import shape

    d = json.load(open(NE, encoding="utf-8"))
    f = [x for x in d["features"] if x["properties"]["ADM0_A3"] == a3]
    if len(f) != 1:
        raise SystemExit(f"Natural Earth has {len(f)} features {a3}")
    return gpd.GeoSeries([shape(f[0]["geometry"])], crs=4326).to_crs(crs).iloc[0]


def seat_check(out, units, names, rel):
    import geopandas as gpd

    with zipfile.ZipFile(GEONAMES) as zf:
        t = pd.read_csv(zf.open("CU.txt"), sep="\t", header=None, names=GEONAMES_COLS,
                        quoting=3, dtype=str, keep_default_na=False)
    s = t[t["fcode"].isin(["PPLA", "PPLC"])].copy()
    s["lat"], s["lon"] = s["lat"].astype(float), s["lon"].astype(float)
    s["population"] = pd.to_numeric(s["population"], errors="coerce").fillna(0).astype(int)
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["lon"], s["lat"]), crs=4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="inner", predicate="within")
    print(f"\n  GeoNames: {len(s)} PPLA/PPLC places, {j['unit'].nunique()} of {len(units)} "
          "provinces hold one")
    rows = []
    for _i, r in j[j["population"] >= SEAT_MIN_POP].iterrows():
        h = out[out["unit"] == r["unit"]]
        d = km(r["lat"], r["lon"], h["lat"].to_numpy(), h["lon"].to_numpy())
        rows.append((r["unit"], r["name"], int(r["population"]),
                     float(h.loc[d <= HOLE_KM, "pop"].sum()), float(h.loc[d <= WIDE_KM, "pop"].sum())))
    c = pd.DataFrame(rows, columns=["unit", "seat", "geonames", "kontur", "kontur_wide"])
    c["ratio"] = c["kontur"] / c["geonames"]
    c["ratio_wide"] = c["kontur_wide"] / c["geonames"]
    c["region"] = c["unit"].map(rel)
    c = c.sort_values("ratio")
    print(f"  Kontur within {HOLE_KM:g} and {WIDE_KM:g} km of each seat of {SEAT_MIN_POP:,}+, "
          f"all {len(c)}, with the province's own Kontur/ONEI ratio over the national one:")
    for _i, r in c.iterrows():
        print(f"      {names[r['unit']]:<20} {r['seat']:<22} GeoNames {r['geonames']:>9,}   "
              f"Kontur {r['kontur']:>9,.0f} ({r['ratio']:.2f})  {r['kontur_wide']:>9,.0f} "
              f"({r['ratio_wide']:.2f})   province {r['region']:.2f}")
    holes = set(c.loc[(c["ratio"] < HOLE_RATIO) & (c["region"] < LOW_RATIO), "unit"])
    if holes != KONTUR_HOLES:
        raise SystemExit(f"seats with under {HOLE_RATIO:.0%} of their people in Kontur, in a "
                         f"province under {LOW_RATIO} of its share: {sorted(holes)}, not "
                         f"{sorted(KONTUR_HOLES)}")


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    gpkg = os.path.join(RAW, GPKG_NAME)
    if "--fetch" in sys.argv or not os.path.exists(gpkg) or not os.path.exists(GEONAMES):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/cu_geo.py first")

    hexes = read_layer(gpkg, "Kontur CU")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} provinces, expected {EXPECTED_UNITS}")
    munis = gpd.read_file(MUNI_GPKG)
    mpop = pd.read_csv(MUNI_CSV, dtype={"adm2_pcode": str, "adm1_pcode": str})
    if len(munis) != EXPECTED_MUNIS or set(munis["adm2_pcode"]) != set(mpop["adm2_pcode"]):
        raise SystemExit(f"{MUNI_GPKG} and {MUNI_CSV} are not the same {EXPECTED_MUNIS} municipalities")

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=cent,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    base = pts.within(ne_feature("USG", units.crs)).to_numpy()
    print(f"  Guantánamo Bay Naval Base (Natural Earth USG): {int(base.sum())} hexes, "
          f"{pts.loc[base, popcol].sum():,.0f} Kontur people, dropped")
    if not 0 < base.sum() < 200:
        raise SystemExit("the naval base drop caught no hexes, or far too many")

    mu = munis[["adm2_pcode", "geometry"]]
    joined = gpd.sjoin(pts, mu, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    joined.loc[base, "adm2_pcode"] = np.nan

    outside = joined["adm2_pcode"].isna().to_numpy() & ~base
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC), mu.to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")].reindex(pts.index[outside])
        print(f"\n  hexes whose centroid is outside every municipality: {int(outside.sum()):,} "
              f"({pts.loc[outside, popcol].sum():,.0f} people); people by distance to one:")
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km: {int(m.sum()):>6,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>10,.0f} people")
        snap = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snap], "adm2_pcode"] = near.loc[snap, "adm2_pcode"]
    dropped = joined["adm2_pcode"].isna()
    print(f"  dropped (base included): {int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} "
          f"people ({100.0 * pts.loc[dropped, popcol].sum() / pts[popcol].sum():.3f}%)")

    keep = ~dropped
    parent = dict(zip(mpop["adm2_pcode"], mpop["adm1_pcode"]))
    out = gpd.GeoDataFrame({"muni": joined.loc[keep, "adm2_pcode"].to_numpy(),
                            "kontur": pts.loc[keep, popcol].to_numpy(dtype=float),
                            "lat": pts.loc[keep].geometry.y.to_numpy(),
                            "lon": pts.loc[keep].geometry.x.to_numpy()},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    out["unit"] = out["muni"].map(parent)
    out = out.reset_index(drop=True)

    # ---- raw Kontur against ONEI, per municipality (2023) and per province (2024) ----
    m23 = dict(zip(mpop["adm2_pcode"], mpop["pop2023"]))
    mname = dict(zip(munis["adm2_pcode"], munis["adm2_name"]))
    per_m = out.groupby("muni")["kontur"].agg(["size", "sum"])
    missing = sorted(set(m23) - set(per_m.index))
    if missing or (per_m["sum"] <= 0).any():
        raise SystemExit(f"municipalities with no populated hex: {missing}")
    est = dict(zip(units["unit"], units["pop"]))
    names = dict(zip(units["unit"], units["name"]))
    tot = float(out["kontur"].sum())
    ratio = tot / sum(est.values())
    ratio23 = tot / sum(m23.values())
    print(f"\n  Kontur {tot:,.0f} vs ONEI 2024 {sum(est.values()):,}: ratio {ratio:.3f} "
          f"(expected about {EXPECTED_RATIO}); vs ONEI 2023 {sum(m23.values()):,}: {ratio23:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and ONEI disagree beyond the band")
    per_p = out.groupby("unit")["kontur"].sum()
    rel_p = {x: (per_p[x] / est[x]) / ratio for x in sorted(est)}
    print("  per-province raw Kontur 2023 / ONEI 2024, over the national ratio:")
    for x in sorted(rel_p, key=rel_p.get):
        print(f"      {names[x]:<20} {per_p[x]:>11,.0f}  {est[x]:>10,}  {rel_p[x]:5.2f}")

    u = sorted(m23)
    a = np.array([per_m.loc[x, "sum"] for x in u])
    b = np.array([m23[x] for x in u], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, ONEI 2023) over {len(u)} municipalities = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it (best {perm.max():+.3f})")
    if beaten > PERM_P * N_PERM:
        raise SystemExit("the rank witness fails; the municipality join may be permuted")
    rel_m = {x: (per_m.loc[x, "sum"] / m23[x]) / ratio23 for x in u}
    spread = pd.DataFrame({"prov": [parent[x] for x in u], "rel": [rel_m[x] for x in u]})
    print("  per-municipality raw Kontur / ONEI 2023 over the national ratio, by province (min, median, max):")
    for p, s in spread.groupby("prov")["rel"]:
        print(f"      {names[p]:<20} {s.min():5.2f} {s.median():5.2f} {s.max():5.2f}")
    holes = {x for x in u if rel_m[x] < 1.0 / HOLE_FACTOR}
    if holes != MUNI_HOLES:
        raise SystemExit(f"municipalities Kontur has lost (under 1/{HOLE_FACTOR:g} of the national ratio): "
                         f"{sorted(holes)}, pinned {sorted(MUNI_HOLES)}")

    out["pop"] = out["kontur"]
    seat_check(out, units, names, rel_p)

    # ---- Kontur's density cap, read on the RAW layer (sources/mr_grid.py's method) ----
    import kontur_cap

    raw = out[["unit", "geometry"]].copy()
    raw["pop"] = out["kontur"].to_numpy()
    raw = raw.to_crs(4326)
    blk = kontur_cap.find_blocks(raw)
    at_cap = [idx for idx in blk["groups"] if (blk["dens"][idx] >= kontur_cap.AT_CAP).any()]
    print(f"\n  raw Kontur: densest hex {blk['dens'].max():,.0f}/km2; {len(blk['groups'])} blocks "
          f"over {kontur_cap.HI:,.0f}/km2, {len(at_cap)} of them reaching the cap of "
          f"{kontur_cap.CAP:,.0f}")
    weight = out["kontur"].to_numpy(dtype=float).copy()
    found, unlisted = set(), []
    for idx in at_cap:
        pk = idx[np.argmax(blk["dens"][idx])]
        lon, lat = float(blk["lon"][pk]), float(blk["lat"][pk])
        ms = sorted(set(out.loc[idx, "muni"]))
        print(f"      block of {len(idx)} hexes at ({lon:.4f}, {lat:.4f}), {blk['pop'][idx].sum():,.0f} "
              f"people, in {[mname[x] for x in ms]}")
        key = next((k for k in BLOCKS if km(k[1], k[0], lat, lon) <= 1.0), None)
        if key is None:
            unlisted.append((lon, lat))
            continue
        found.add(key)
        action, name = BLOCKS[key]
        if action == "left":
            print(f"        {name}: left as Kontur has it (BLOCKS)")
            continue
        if action != "capped":
            raise SystemExit(f"BLOCKS action {action!r} is neither `capped` nor `left`")
        inblock = np.zeros(len(weight), dtype=bool)
        inblock[idx] = True
        near = blk["tree"].query_ball_point(blk["xy"][idx], kontur_cap.RING_KM * 1000.0)
        ring = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near]))
        ring = ring[~inblock[ring] & (weight[ring] > 0)]
        if len(ring) == 0:
            raise SystemExit(f"{name} has no populated hex in its ring")
        ceiling = float(np.median(blk["dens"][ring]))
        weight[idx] = np.minimum(weight[idx], ceiling * blk["area"][idx])
        print(f"        {name}: lowered to the {kontur_cap.RING_KM:g} km ring's median of "
              f"{ceiling:,.0f}/km2, now {weight[idx].sum():,.0f} people")
    if unlisted:
        raise SystemExit(f"raw Kontur blocks at the cap not in BLOCKS: {unlisted}; review them and name them there")
    if found != set(BLOCKS):
        raise SystemExit(f"BLOCKS rows matching no block at the cap: {sorted(set(BLOCKS) - found)}")

    # ---- calibrate every hex to its province's ONEI 2024 count ----
    # Province, not municipality: inside each province Kontur's error is near uniform (Granma's
    # municipalities all read 0.16-0.19 of their share, Santiago's 0.37-0.44), so a province
    # factor removes it, while COD-AB's municipal lines in Havana are drawn about 2 km east of
    # the real ones (its La Habana Vieja runs to -82.286, over the harbour and Regla), and a
    # municipal calibration there would move people onto the wrong polygons. sources/cu.md §5.
    out["capped"] = weight
    per_c = out.groupby("unit")["capped"].sum()
    out["pop"] = weight * out["unit"].map(lambda x: est[x] / per_c[x]).to_numpy()
    dens = out["pop"].to_numpy() / (out.to_crs(6933).geometry.area.to_numpy() / 1e6)
    top = int(np.argmax(dens))
    print(f"  calibrated densest hex {dens.max():,.0f}/km2 in {mname[out.loc[top, 'muni']]} "
          f"({'above' if dens.max() > kontur_cap.OVER_CAP else 'not above'} Kontur's limit, so "
          f"kontur_cap.py {'skips' if dens.max() > kontur_cap.OVER_CAP else 'checks'} this layer)")
    chk = out.groupby("unit")["pop"].sum()
    worst = max(abs(chk[x] - est[x]) for x in est)
    if worst > 0.5:
        raise SystemExit(f"calibration leaves a province {worst:.2f} people off ONEI's 2024 count")
    moved = 0.5 * sum(abs(per_p[x] / tot - est[x] / sum(est.values())) for x in est)
    print(f"  calibrated: every province sums to ONEI 2024 (worst {worst:.3f}); raw Kontur had "
          f"{moved:.1%} of its people in a different province from ONEI")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
