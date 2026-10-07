"""New Caledonia: the 33 communes and the Kontur placement layer inside them.

Writes data/geo/nc/nc_units.gpkg (`unit`, `pop`, geometry), data/geo/nc/nc_lookup.csv
(`geo_id` -> `unit`) and data/geo/nc/nc_hexes.gpkg (`unit`, `pop` as Kontur has it, geometry),
from data/raw/nc/:

  * Gouvernement de la Nouvelle-Calédonie (DTSI, Georep), *Communes de la Nouvelle-Calédonie
    (limites communales terrestres simplifiées)*, data.gouv.nc dataset
    `communes-nc-limites-terrestres-simplifiees` (modified 2024-12-06; Licence Ouverte v2.0),
    GeoJSON export: 33 communes with the commune code `code_com`. HDX has no COD-AB for New
    Caledonia and geoBoundaries has only ADM0;
  * Kontur Population NC (2023-11-01), 400 m hexagons.

THE JOIN IS ON THE COMMUNE CODE (`sources/nc.py::COMMUNES`), witnessed by the name (the file's
upper-case `nom` must equal the census name folded) and by Kontur's people per commune against the
2019 census (`UNIT_BAND`). Hex centroids are joined to the communes; an offshore centroid is
snapped to the nearest commune within `SNAP_KM`, and anything further (the uninhabited outer
islands, reefs) is dropped and counted.

Usage:
    python sources/nc_geo.py --fetch    the commune GeoJSON (about 17 MB) and Kontur NC (gz)
    python sources/nc_geo.py            rebuild from data/raw/nc/
"""

import gzip
import os
import shutil
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

from nc import COMMUNES

RAW = os.path.join(ROOT, "data", "raw", "nc")
GEO = os.path.join(ROOT, "data", "geo", "nc")
COMMUNES_GJ = os.path.join(RAW, "communes_simplifiees.geojson")
COMMUNES_URL = ("https://data.gouv.nc/api/explore/v2.1/catalog/datasets/"
                "communes-nc-limites-terrestres-simplifiees/exports/geojson")
KONTUR_GZ = os.path.join(RAW, "kontur_population_NC_20231101.gpkg.gz")
KONTUR = os.path.join(RAW, "kontur_population_NC_20231101.gpkg")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
              "kontur_population_NC_20231101.gpkg.gz")
NORMALIZED = os.path.join(ROOT, "data", "normalized", "nc.csv")

EXPECTED_RATIO = 1.08           # Kontur 2023 over the 2019 count; measured 2026-10-03
TOLERANCE = 0.15
# Kontur reads the Loyalty Islands low (Mare 0.47, Ouvea 0.55, Lifou 0.58 of the national ratio):
# GHSL-built grids miss dispersed tribal hamlets. Dots are placed inside each commune only, so the
# per-commune scale does not move anyone between communes; the band catches a wrong join.
UNIT_BAND = (0.4, 2.0)
SNAP_KM = 2.0
SNAP_BANDS_KM = (0.5, 1.0, 2.0, 5.0, 50.0)
DROP_MAX = 0.01
AREA_KM2 = 18_576               # ISEE's land area of New Caledonia
AREA_TOL = 0.03
METRIC = "EPSG:3163"            # RGNC91-93 / Lambert New Caledonia
KONTUR_CAP = 46_200
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, dst, minsize in ((COMMUNES_URL, COMMUNES_GJ, 1_000_000),
                              (KONTUR_URL, KONTUR_GZ, 50_000)):
        if os.path.exists(dst) and os.path.getsize(dst) > minsize:
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
    if not os.path.exists(KONTUR):
        with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR + ".part", KONTUR)


def fold(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    for a in ("(L')", "(LE)"):
        s = s.replace(a, "")
    return s.replace("-", " ").replace("'", " ").strip()


def read_communes():
    import geopandas as gpd

    g = gpd.read_file(COMMUNES_GJ)
    code_of = {v[0]: k for k, v in COMMUNES.items()}
    if len(g) != 33 or set(g["code_com"].astype(str)) != set(code_of):
        raise SystemExit(f"commune file: {len(g)} features, codes "
                         f"{sorted(set(g['code_com'].astype(str)) ^ set(code_of))} differ")
    for _, r in g.iterrows():
        want = fold(code_of[str(r["code_com"])])
        if fold(r["nom"]) != want:
            raise SystemExit(f"{r['code_com']} is `{r['nom']}` in the file, `{want}` in the census")
    if g.geometry.is_empty.any() or g.geometry.isna().any():
        raise SystemExit("a commune has no geometry")
    g["unit"] = g["code_com"].astype(str)
    g = g[["unit", "geometry"]].to_crs("EPSG:4326")
    area = g.to_crs(METRIC).area.sum() / 1e6
    print(f"  communes: 33, joined on code and name; land area {area:,.0f} km2 against ISEE's "
          f"{AREA_KM2:,}")
    if abs(area / AREA_KM2 - 1) > AREA_TOL:
        raise SystemExit("the commune polygons' area is off ISEE's")
    return g


def main():
    import geopandas as gpd
    from geo_checks import read_layer

    if "--fetch" in sys.argv or not (os.path.exists(COMMUNES_GJ) and os.path.exists(KONTUR)):
        fetch()
    if not os.path.exists(NORMALIZED):
        raise SystemExit(f"missing {NORMALIZED}; run sources/nc.py first")
    units = read_communes()
    rows = pd.read_csv(NORMALIZED, usecols=["geo_id", "geo_name", "count"], dtype={"geo_id": str})
    pop = rows.groupby("geo_id")["count"].sum()
    names = rows.drop_duplicates("geo_id").set_index("geo_id")["geo_name"]
    if set(pop.index) != set(units["unit"]):
        raise SystemExit(f"nc.csv communes {sorted(set(pop.index) ^ set(units['unit']))} differ")
    units["pop"] = units["unit"].map(pop).astype(int)

    hexes = read_layer(KONTUR, "Kontur NC")
    popcol = "population"
    print(f"\nKontur hexes: {len(hexes):,}, population {hexes[popcol].sum():,.0f}")
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()}, geometry=hexes.geometry.centroid,
                           crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    outside = joined["unit"].isna()
    print(f"  hexes whose centroid is outside every commune: {int(outside.sum()):,} "
          f"({pts.loc[outside, popcol].sum():,.0f} people)")
    if outside.any():
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(METRIC),
                                 units[["unit", "geometry"]].to_crs(METRIC),
                                 how="left", distance_col="d")
        near = near[~near.index.duplicated(keep="first")]
        for b in SNAP_BANDS_KM:
            m = near["d"] <= b * 1000
            print(f"      within {b:>4g} km of a commune: {int(m.sum()):>5,} hexes, "
                  f"{pts.loc[near.index[m], popcol].sum():>9,.0f} people")
        snapped = near["d"] <= SNAP_KM * 1000
        joined.loc[near.index[snapped], "unit"] = near.loc[snapped, "unit"]
        print(f"  snapped within {SNAP_KM:g} km: {int(snapped.sum()):,} hexes")
    dropped = joined["unit"].isna()
    share = pts.loc[dropped, popcol].sum() / pts[popcol].sum()
    print(f"  dropped: {int(dropped.sum()):,} hexes, {pts.loc[dropped, popcol].sum():,.0f} people "
          f"({share:.3%})")
    if share > DROP_MAX:
        raise SystemExit(f"more than {DROP_MAX:.0%} of Kontur's people dropped")

    keep = ~dropped
    out = gpd.GeoDataFrame({"unit": joined.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)
    per = out.groupby("unit")["pop"].sum()
    drawn = dict(zip(units["unit"], units["pop"].astype(int)))
    tot = float(per.sum())
    ratio = tot / sum(drawn.values())
    print(f"\n  Kontur {tot:,.0f} vs the 2019 census {sum(drawn.values()):,}: ratio {ratio:.3f}")
    if abs(ratio - EXPECTED_RATIO) > TOLERANCE:
        raise SystemExit("Kontur and the census total disagree beyond the band")
    print("  per commune: Kontur / census over the national ratio")
    bad = []
    for u in sorted(drawn, key=lambda x: -drawn[x]):
        rel = (per.get(u, 0.0) / drawn[u]) / ratio
        print(f"      {names[u]:<20} census {drawn[u]:>7,}  Kontur {per.get(u, 0.0):>9,.0f}  "
              f"{rel:5.2f}")
        if not UNIT_BAND[0] <= rel <= UNIT_BAND[1]:
            bad.append(names[u])
    if bad:
        raise SystemExit(f"communes outside UNIT_BAND {UNIT_BAND}: {bad}")
    dens = out["pop"] / (out.to_crs("EPSG:6933").area / 1e6)
    print(f"  densest hex {dens.max():,.0f}/km2 (Kontur's cap {KONTUR_CAP:,})")

    os.makedirs(GEO, exist_ok=True)
    units[["unit", "pop", "geometry"]].to_file(os.path.join(GEO, "nc_units.gpkg"), layer="units",
                                               driver="GPKG")
    pd.DataFrame({"geo_id": sorted(drawn), "unit": sorted(drawn)}).to_csv(
        os.path.join(GEO, "nc_lookup.csv"), index=False, encoding="utf-8")
    out[["unit", "pop", "geometry"]].to_file(os.path.join(GEO, "nc_hexes.gpkg"), layer="hexes",
                                             driver="GPKG")
    print(f"\nwrote {GEO}/nc_units.gpkg, nc_lookup.csv and nc_hexes.gpkg ({len(out):,} cells)")


if __name__ == "__main__":
    main()
