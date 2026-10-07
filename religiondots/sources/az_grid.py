"""Azerbaijan: the placement layer, Kontur 400 m population hexagons keyed to the 74 units.

Writes data/geo/az/az_hexes.gpkg. Copied in shape from `sources/so_grid.py`; `sources/az.md` §5 is
the record.

THE JOIN IS ON HEX CENTROIDS. A hex whose centroid is outside every unit is snapped to the nearest
unit within `SNAP_KM` when it is not inside a neighbouring country on Natural Earth (Armenia,
Georgia, Iran, Russia, Türkiye), and dropped otherwise.

KARABAKH. The counts are the 2019 census's EXISTING population, and the census enumerated nobody in
the territory Armenian forces held in 2019 (`sources/az_geo.py`). Kontur's extract is dated
2023-11-01 and is modelled from built-up area, which still shows the ruins of Aghdam and Fuzuli and
the towns the Karabakh Armenians lived in until September 2023, so Kontur puts 2.7 times Aghdam's
census people inside Aghdam and 79,000 people in Jabrayil, where the census found 420. Every hex
inside the area held in 2019 is therefore masked: Natural Earth 4.1.0's `Nagorno-Karabakh` feature
(2018, public domain, 11,923 km2, the 1994-2020 line; `NE_KARABAKH_URL`), which matches the
census's own year. Today's Natural Earth has only the 2020-2023 remainder (`Artsakh`, inside it).
The eight units with no existing population draw nothing whatever Kontur holds there.

THREE CHECKS AGAINST THE CENSUS: the national ratio (Kontur over the existing population, inside
`EXPECTED_RATIO` +/- `TOLERANCE`); the rank witness for the join (Spearman over the populated
units against `N_PERM` shuffles); every populated unit holds a populated hex.

Usage:
    python sources/az_grid.py --fetch    Kontur AZ (2.6 MB gzipped) into data/raw/az/
    python sources/az_grid.py            rebuild from data/raw/az/
"""

import gzip
import json
import os
import shutil
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "az")
GEO = os.path.join(ROOT, "data", "geo", "az")
UNITS = os.path.join(GEO, "az_units.gpkg")
OUT = os.path.join(GEO, "az_hexes.gpkg")
NE = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")
NE_KARABAKH_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v4.1.0/geojson/"
                   "ne_10m_admin_0_disputed_areas.geojson")
NE_KARABAKH = os.path.join(RAW, "ne_v410_disputed_areas.geojson")
NE_KARABAKH_KM2 = (11_800, 12_050)       # measured 11,923 km2 in EPSG:32638

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_AZ_20231101.gpkg.gz")
GZ = os.path.join(RAW, "kontur_population_AZ_20231101.gpkg.gz")
GPKG = os.path.join(RAW, "kontur_population_AZ_20231101.gpkg")

NEIGHBOURS = ("ARM", "GEO", "IRN", "RUS", "TUR")
SNAP_KM = 2.0
EXPECTED_RATIO = 1.0
TOLERANCE = 0.25
N_PERM = 20_000
PERM_P = 0.001
METRIC = "EPSG:32639"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"


def fetch():
    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GZ):
        req = urllib.request.Request(GZ_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=900) as r:
            data = r.read()
        with open(GZ + ".part", "wb") as fh:
            fh.write(data)
        os.replace(GZ + ".part", GZ)
    if not os.path.exists(NE_KARABAKH):
        req = urllib.request.Request(NE_KARABAKH_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        with open(NE_KARABAKH + ".part", "wb") as fh:
            fh.write(data)
        os.replace(NE_KARABAKH + ".part", NE_KARABAKH)
    if not os.path.exists(GPKG):
        with gzip.open(GZ, "rb") as src, open(GPKG + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(GPKG + ".part", GPKG)
    with open(GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{GPKG} is not a GeoPackage")


def ne_layer(path, keep):
    import geopandas as gpd
    from shapely.geometry import shape

    d = json.load(open(path, encoding="utf-8"))
    rows = [f for f in d["features"] if keep(f["properties"])]
    return gpd.GeoDataFrame({"a3": [f["properties"].get("ADM0_A3") for f in rows],
                             "brk": [f["properties"].get("BRK_NAME") for f in rows]},
                            geometry=[shape(f["geometry"]) for f in rows], crs=4326)


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    if "--fetch" in sys.argv or not os.path.exists(GPKG) or not os.path.exists(NE_KARABAKH):
        fetch()
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS}; run sources/az_geo.py first")

    hexes = read_layer(GPKG, "Kontur AZ")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    print(f"Kontur hexes: {len(hexes):,}, population {hexes[popcol].sum():,.0f}")
    units = gpd.read_file(UNITS)
    if len(units) != 74:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected 74")

    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)

    # Natural Earth 4.1.0's Nagorno-Karabakh: the area held by the Armenian side from 1994 to
    # 2020, when the census was taken. Masked whatever unit it is in.
    art = ne_layer(NE_KARABAKH, lambda p: p.get("BRK_NAME") == "Nagorno-Karabakh").to_crs(units.crs)
    km2 = float(art.to_crs("EPSG:32638").area.sum() / 1e6) if len(art) else 0.0
    if len(art) != 1 or not NE_KARABAKH_KM2[0] <= km2 <= NE_KARABAKH_KM2[1]:
        raise SystemExit(f"Natural Earth 4.1.0 has {len(art)} Nagorno-Karabakh features, {km2:,.0f} km2")
    in_art = gpd.sjoin(pts, art[["geometry"]], how="left", predicate="within")
    in_art = in_art[~in_art.index.duplicated(keep="first")].reindex(pts.index)["index_right"].notna()
    t = pd.DataFrame({"unit": j["unit"].fillna("(outside)"), "pop": pts["pop"]})[in_art]
    print(f"\n  Kontur people inside the area held in 2019 (Natural Earth 4.1.0), masked: {t['pop'].sum():,.0f}, by unit:")
    for u, v in t.groupby("unit")["pop"].sum().sort_values(ascending=False).items():
        print(f"      {u:<12} {v:>9,.0f}")
    j.loc[in_art, "unit"] = np.nan
    j["masked"] = in_art

    outside = j["unit"].isna() & ~in_art
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")].reindex(pts.index[outside])
        ne = ne_layer(NE, lambda p: p.get("ADM0_A3") in NEIGHBOURS).to_crs(units.crs)
        w = gpd.sjoin(pts.loc[outside], ne[["a3", "geometry"]], how="left", predicate="within")
        w = w[~w.index.duplicated(keep="first")].reindex(pts.index[outside])
        a3 = w["a3"].fillna("none")
        snap = (a3 == "none") & (near["d"] <= SNAP_KM * 1000)
        tab = pd.DataFrame({"rule": np.where(snap, "snap", "drop"), "ne": a3,
                            "pop": pts.loc[outside, "pop"]})
        print(f"\n  hexes outside every unit and outside the 2019 Karabakh mask: {int(outside.sum()):,}, "
              f"{pts.loc[outside, 'pop'].sum():,.0f} people; by rule and Natural Earth country:")
        for (r_, n_), r in tab.groupby(["rule", "ne"])["pop"].agg(["size", "sum"]).iterrows():
            print(f"      {r_:<5} {n_:<5} {int(r['size']):>6,} hexes  {r['sum']:>10,.0f} people")
        j.loc[near.index[snap], "unit"] = near.loc[snap, "unit"]
    keep = j["unit"].notna()
    print(f"  dropped or masked: {int((~keep).sum()):,} hexes, {pts.loc[~keep, 'pop'].sum():,.0f} people")

    out = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, "pop"].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    est = dict(zip(units["unit"], units["pop"]))
    names = dict(zip(units["unit"], units["name"]))
    populated = sorted(u for u in est if est[u] > 0)
    missing = [names[u] for u in populated if u not in per.index or per.loc[u, "sum"] <= 0]
    if missing:
        raise SystemExit(f"populated units with no populated hex: {missing}")

    tot = float(per.loc[populated, "sum"].sum())
    ratio = tot / sum(est.values())
    print(f"\n  Kontur in the populated units {tot:,.0f} vs census existing {sum(est.values()):,}: "
          f"ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census disagree beyond the band")
    a = np.array([per.loc[x, "sum"] for x in populated])
    b = np.array([est[x] for x in populated], dtype=float)
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    print(f"  join witness: Spearman(Kontur, census) over {len(populated)} units = {rho:+.3f}; "
          f"{beaten} of {N_PERM:,} shuffles reach it")
    if beaten > PERM_P * N_PERM:
        raise SystemExit("the rank witness fails; the unit join may be permuted")
    rel = {x: (per.loc[x, "sum"] / est[x]) / ratio for x in populated}
    print("\n  per-unit Kontur / census existing, over the national ratio (lowest and highest ten):")
    srt = sorted(rel, key=rel.get)
    for x in srt[:10] + ["..."] + srt[-10:]:
        if x == "...":
            print("      ...")
            continue
        print(f"      {names[x]:<14} census {est[x]:>9,}  Kontur {per.loc[x, 'sum']:>10,.0f}  {rel[x]:5.2f}")
    for x in sorted(est):
        if est[x] == 0 and x in per.index:
            print(f"  empty at the census, Kontur holds {per.loc[x, 'sum']:>8,.0f} in {names[x]} (never drawn)")

    os.makedirs(GEO, exist_ok=True)
    out[["unit", "pop", "geometry"]].to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
