"""Morocco: the placement layer, Kontur 400 m population hexagons keyed to the 73 units.

Writes data/geo/ma/ma_hexes.gpkg. `sources/ma.md` §6 is the record in prose.

Two Kontur extracts, `MA` and `EH` (20231101), de-duplicated on `h3`. A hex belongs to the unit
holding its centroid (`sources/ma_geo.py`'s polygons, Western Sahara already cut to the part west
of the berm, Ceuta and Melilla already cut out). A hex whose centroid is in no unit is:

  * **in the Tarfaya strip**, between COD's Tan-Tan and 27°40'N, which neither COD file draws:
    kept for the Laâyoune and Tarfaya unit, and asserted to reach GeoNames' Tarfaya and Akhfennir;
  * **east of the berm** (Natural Earth B28), **in Ceuta or Melilla** (B60, B61), or **a hex
    Algeria's own placement layer draws** (`data/geo/dz/dz_hexes.gpkg`): dropped;
  * otherwise, within `SNAP_M` of a unit (the coast, slivers between the two COD files): snapped
    to the nearest unit, as Vanuatu's are (`playbooks/geography.md`); the rest dropped.

## TOWNS KONTUR HAS LOST

Kontur is 2023 and the census 2024, so the national ratio reads near 1 and each unit's ratio says
whether Kontur has its people. Where a unit reads under `LOW_RATIO`, its main municipality is
checked against HCP's own commune count: under `HOLE_SHARE` of it within 5 km of the GeoNames point,
a 3 km disc on the point takes the shortfall, as Algeria's Béchar (`sources/dz_grid.py`).

Usage:
    python sources/ma_grid.py            rebuild from data/raw/ma/ (unpacks the .gz once)
"""

import gzip
import json
import os
import shutil
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ma")
GEO = os.path.join(ROOT, "data", "geo", "ma")
UNITS = os.path.join(GEO, "ma_units.gpkg")
LOOKUP = os.path.join(GEO, "ma_lookup.csv")
OUT = os.path.join(GEO, "ma_hexes.gpkg")
DZ_HEXES = os.path.join(ROOT, "data", "geo", "dz", "dz_hexes.gpkg")
DISPUTED = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_disputed_areas.geojson")
KONTUR = {cc: f"kontur_population_{cc}_20231101.gpkg" for cc in ("MA", "EH")}
KONTUR_URL = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"

STRIP_UNIT = "EH03"                         # Laâyoune and Tarfaya
STRIP = dict(lon=(-13.6, -11.0), lat=(27.66, 28.45))
STRIP_TOWNS = ("Tarfaya", "Akhfennir")      # each must have strip hexes within 5 km
SNAP_M = 1000
EXPECTED_RATIO, RATIO_TOL = 1.0, 0.25       # Kontur 2023 over RGPH 2024, written before the run
HOLE_KM, DISC_KM = 5.0, 3.0
LOW_RATIO, HOLE_SHARE = 0.5, 0.5
# Every unit that may read under LOW_RATIO names its main municipality: (GeoNames names, HCP code).
TOWNS = {"MA005003": (("Tan-Tan", "Tantan"), 105210101),
         "EH02": (("Smara", "Es Semara", "Es-Semara", "Semara"), 112210101),
         "MA005001": (("Assa",), 100710101)}
# Towns Kontur has lost, asserted. Measured 2026-09-15: Smara 0.07 of its commune count within
# 5 km, Tan-Tan 0.12, Assa 0.32 (the first expectation missed Assa; the rule found it).
FILLED = {"MA005003", "EH02", "MA005001"}
SEAT_MIN_POP, SEAT_HOLE = 50_000, 0.10

GEONAMES_COLS = ["geonameid", "name", "asciiname", "alt", "lat", "lon", "fclass", "fcode", "cc",
                 "cc2", "a1", "a2", "a3", "a4", "population", "elev", "dem", "tz", "mod"]


def km(lat0, lon0, lat, lon):
    p = np.radians
    a = (np.sin(p(lat - lat0) / 2) ** 2
         + np.cos(p(lat0)) * np.cos(p(lat)) * np.sin(p(lon - lon0) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def unpack():
    import requests

    for gpkg in KONTUR.values():
        path = os.path.join(RAW, gpkg)
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            continue
        gz = path + ".gz"
        if not os.path.exists(gz):
            r = requests.get(KONTUR_URL + gpkg + ".gz", timeout=1800, stream=True)
            r.raise_for_status()
            with open(gz + ".part", "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            os.replace(gz + ".part", gz)
        with gzip.open(gz, "rb") as src, open(path + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(path + ".part", path)
        with open(path, "rb") as fh:
            if fh.read(4) != b"SQLi":
                raise SystemExit(f"{path} is not a GeoPackage")


def geonames():
    frames = []
    for cc in ("MA", "EH"):
        with zipfile.ZipFile(os.path.join(RAW, f"geonames_{cc}.zip")) as zf:
            frames.append(pd.read_csv(zf.open(f"{cc}.txt"), sep="\t", header=None,
                                      names=GEONAMES_COLS, quoting=3, dtype=str,
                                      keep_default_na=False))
    t = pd.concat(frames, ignore_index=True)
    t = t[t["fclass"] == "P"].copy()
    t["lat"], t["lon"] = t["lat"].astype(float), t["lon"].astype(float)
    t["population"] = pd.to_numeric(t["population"], errors="coerce").fillna(0).astype(int)
    return t


def centroid_keys(gdf):
    c = gdf.to_crs(3857).geometry.centroid
    return pd.Series(list(zip((c.x / 10).round().astype(int), (c.y / 10).round().astype(int))),
                     index=gdf.index)


def main():
    import geopandas as gpd
    from shapely.geometry import shape
    from geo_checks import read_layer
    from ma_geo import read_workbook

    unpack()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/ma_geo.py first")
    units = gpd.read_file(UNITS)
    lut = pd.read_csv(LOOKUP)
    if len(units) != 73:
        raise SystemExit(f"{len(units)} units, expected 73")

    parts = []
    for cc, gpkg in KONTUR.items():
        h = read_layer(os.path.join(RAW, gpkg), f"Kontur {cc}")
        popcol = next(c for c in h.columns if c.lower() == "population")
        h = h.rename(columns={popcol: "pop"})[["h3", "pop", "geometry"]]
        print(f"  Kontur {cc}: {len(h):,} hexes, {h['pop'].sum():,.0f} people")
        parts.append(h)
    hexes = pd.concat(parts, ignore_index=True)
    dup = hexes["h3"].duplicated()
    hexes = gpd.GeoDataFrame(hexes[~dup].reset_index(drop=True), crs=parts[0].crs).to_crs(4326)
    print(f"  {int(dup.sum()):,} hexes in both extracts, kept once")

    cent = gpd.GeoDataFrame({"i": hexes.index},
                            geometry=hexes.to_crs(3857).geometry.centroid.to_crs(4326), crs=4326)
    j = gpd.sjoin(cent, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(cent.index)
    hexes["unit"] = j["unit"].to_numpy()
    hexes["lon"], hexes["lat"] = cent.geometry.x.to_numpy(), cent.geometry.y.to_numpy()

    # ---- hexes with no unit ----
    with open(DISPUTED, encoding="utf-8") as fh:
        brk = {f["properties"]["BRK_A3"]: shape(f["geometry"]) for f in json.load(fh)["features"]
               if f["properties"]["BRK_A3"] in ("B28", "B60", "B61")}
    orphan = hexes["unit"].isna()
    strip = orphan & hexes["lon"].between(*STRIP["lon"]) & hexes["lat"].between(*STRIP["lat"])
    east = orphan & ~strip & cent.geometry.within(brk["B28"]).to_numpy()
    spain = orphan & (cent.geometry.within(brk["B60"]) | cent.geometry.within(brk["B61"])).to_numpy()
    dz_keys = set(centroid_keys(gpd.read_file(DZ_HEXES)))
    algeria = orphan & ~strip & ~east & ~spain & centroid_keys(hexes).isin(dz_keys)
    rest = orphan & ~strip & ~east & ~spain & ~algeria
    near = gpd.sjoin_nearest(cent[rest].to_crs(32629), units[["unit", "geometry"]].to_crs(32629),
                             how="left", max_distance=SNAP_M)
    near = near[~near.index.duplicated(keep="first")]
    snapped = near.index[near["unit"].notna()]
    hexes.loc[snapped, "unit"] = near.loc[snapped, "unit"]
    hexes.loc[strip, "unit"] = STRIP_UNIT
    dropped = orphan & ~strip & ~hexes.index.isin(snapped)

    def p(mask):
        return f"{int(mask.sum()):,} hexes, {hexes.loc[mask, 'pop'].sum():,.0f} people"
    print(f"  hexes with no unit: Tarfaya strip {p(strip)} (kept); east of the berm {p(east)}; "
          f"Ceuta and Melilla {p(spain)}; drawn by Algeria {p(algeria)}; within {SNAP_M} m of a "
          f"unit {p(hexes.index.isin(snapped))} (snapped); dropped otherwise "
          f"{p(dropped & ~east & ~spain & ~algeria)}")
    gn = geonames()
    for town in STRIP_TOWNS:
        t = gn[gn["name"] == town].sort_values("population").iloc[-1]
        d = km(t["lat"], t["lon"], hexes.loc[strip, "lat"].to_numpy(), hexes.loc[strip, "lon"].to_numpy())
        if hexes.loc[strip, "pop"].to_numpy()[d <= HOLE_KM].sum() <= 0:
            raise SystemExit(f"the strip rule does not reach {town}; read STRIP")
    out = hexes[hexes["unit"].notna()].reset_index(drop=True)

    # ---- the national ratio and the rank join witness ----
    census = lut.set_index("unit")["pop"]
    names = lut.set_index("unit")["name"]
    per = out.groupby("unit")["pop"].sum().reindex(census.index, fill_value=0.0)
    if (per <= 0).any():
        raise SystemExit(f"units with no populated hex: {sorted(per.index[per <= 0])}")
    ratio = per.sum() / census.sum()
    print(f"\n  Kontur {per.sum():,.0f} against RGPH 2024 {census.sum():,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > RATIO_TOL:
        raise SystemExit("Kontur and the census disagree beyond the band; check the extracts")
    from scipy.stats import spearmanr
    rho = spearmanr(per, census).statistic
    rng = np.random.default_rng(0)
    best = max(spearmanr(per, rng.permutation(census.to_numpy())).statistic for _ in range(2000))
    print(f"  Kontur against census per unit: rho {rho:+.3f}; best of 2,000 shuffles {best:+.3f}")
    if rho <= best:
        raise SystemExit("the unit join does not beat a shuffle")
    rel = (per / census) / ratio

    # ---- towns Kontur has lost ----
    _admin, communes, _p, _r = read_workbook()
    low = set(rel.index[rel < LOW_RATIO])
    if low - set(TOWNS):
        raise SystemExit(f"units under {LOW_RATIO} with no municipality in TOWNS: {sorted(low - set(TOWNS))}")
    pts = gpd.GeoDataFrame(gn, geometry=gpd.points_from_xy(gn["lon"], gn["lat"]), crs=4326)
    add, filled = [], set()
    for u in sorted(low):
        aliases, code = TOWNS[u]
        cname, _mor, _fo, cpop = communes[code]
        cand = pts[pts["name"].isin(aliases) | pts["asciiname"].isin(aliases)]
        cand = cand[cand.within(units.loc[units["unit"] == u].geometry.iloc[0])]
        if not len(cand):
            raise SystemExit(f"no GeoNames point for {aliases} inside {names[u]}")
        t = cand.sort_values("population").iloc[-1]
        h = out[out["unit"] == u]
        have = float(h["pop"].to_numpy()[km(t["lat"], t["lon"], h["lat"].to_numpy(),
                                            h["lon"].to_numpy()) <= HOLE_KM].sum())
        share = have / (cpop * ratio)
        print(f"  {names[u]}: Kontur/census {rel[u]:.2f}; {cname} counts {cpop:,}, Kontur holds "
              f"{have:,.0f} within {HOLE_KM:g} km of GeoNames' {t['name']} ({share:.2f})")
        if share < HOLE_SHARE:
            short = cpop * ratio - have
            local = gpd.GeoSeries(gpd.points_from_xy([t["lon"]], [t["lat"]]), crs=4326).to_crs(
                f"+proj=aeqd +lat_0={t['lat']} +lon_0={t['lon']} +units=m")
            disc = local.buffer(DISC_KM * 1000, 32).to_crs(4326).iloc[0]
            add.append({"unit": u, "pop": short, "geometry": disc, "lat": t["lat"], "lon": t["lon"]})
            filled.add(u)
            print(f"    a {DISC_KM:g} km disc on the town takes {short:,.0f}")
    if filled != FILLED:
        raise SystemExit(f"towns filled: {sorted(filled)}, not {sorted(FILLED)}")
    if add:
        out = pd.concat([out, gpd.GeoDataFrame(add, crs=4326)], ignore_index=True)
    per = out.groupby("unit")["pop"].sum()
    rel = (per / census.reindex(per.index)) / (per.sum() / census.sum())
    print("  after: lowest and highest Kontur/census over the national ratio: "
          + ", ".join(f"{names[u]} {rel[u]:.2f}" for u in list(rel.sort_values().index[:4])
                      + list(rel.sort_values().index[-3:])))

    # ---- no other seat of 50,000+ lost ----
    seats = pts[pts["fcode"].isin(["PPLA", "PPLA2", "PPLC"]) & (pts["population"] >= SEAT_MIN_POP)]
    sj = gpd.sjoin(seats, units[["unit", "geometry"]], how="inner", predicate="within")
    bad = []
    for _i, r in sj.iterrows():
        h = out[out["unit"] == r["unit"]]
        q = h["pop"].to_numpy()[km(r["lat"], r["lon"], h["lat"].to_numpy(),
                                   h["lon"].to_numpy()) <= HOLE_KM].sum() / r["population"]
        if q < SEAT_HOLE:
            bad.append(f"{r['name']} {q:.2f}")
    if bad:
        raise SystemExit(f"GeoNames seats Kontur has lost: {bad}")
    print(f"  every one of {len(sj)} GeoNames seats of {SEAT_MIN_POP:,}+ holds {SEAT_HOLE:.0%} or more "
          f"of its people within {HOLE_KM:g} km")

    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
