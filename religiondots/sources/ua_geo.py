"""Ukraine — COD-AB's 27 first-level units, and a population grid to place dots inside them.

Writes:
    data/geo/ua/ua_oblasts.gpkg      27 polygons keyed by COD-AB pcode, as ua.py keys them (`units`)
    data/geo/ua/ua_grid_400m.gpkg    Kontur H3 r8 hexes with `unit` and `pop` (`place`)

Usage:
    python sources/ua_geo.py --fetch   # COD-AB shapefile zip + the Kontur extract
    python sources/ua_geo.py           # build both layers

THE BOUNDARIES ARE UKRAINE'S OWN. COD-AB Ukraine (HDX, CC BY-IGO, v05 valid from 2025-09-01, source
the State Scientific Production Enterprise "Kartographia") carries all 27 first-level units: 24
oblasts, Kyiv city, the Autonomous Republic of Crimea and Sevastopol. Nothing is dropped; how the
occupied territory is drawn is sources/ua.md §3.

THE JOIN IS BY PCODE, AND CHECKED BY NAME AND BY PEOPLE. The COD-AB name for each pcode must fold to
ua.UNIT_NAMES (names, not row order), and Kontur's population per unit is set against Ukrstat's figure:
a scrambled join between oblasts of similar size could pass a band, so the check is a Spearman over
all 27 as well.

WHY A POPULATION GRID. 27 units over 600,000 km2; Kyiv city is 823 km2 inside Kyiv oblast, and the
Carpathians and the southern steppe are empty in places. Kontur 2023-11-01 is pre-invasion in its
inputs, which is what the population base is too.
"""

import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
GEO = os.path.join(ROOT, "data", "geo", "ua")
RAW = os.path.join(ROOT, "data", "raw", "ua")

COD_URL = ("https://data.humdata.org/dataset/d23f529f-31e4-4021-a65b-13987e5cfb42/resource/"
           "2b38131d-5dd6-4b0f-b4c6-4aecc67f179d/download/ukr_admin_boundaries.shp.zip")
COD_ZIP = os.path.join(RAW, "ukr_admin_boundaries.shp.zip")

KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")
KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_UA_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_UA_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_UA_20231101.gpkg")

UNITS_OUT = os.path.join(GEO, "ua_oblasts.gpkg")
GRID_OUT = os.path.join(GEO, "ua_grid_400m.gpkg")


def fetch():
    import gzip
    import shutil

    import requests

    ua = {"User-Agent": "religiondots/1.0"}
    os.makedirs(RAW, exist_ok=True)
    if not (os.path.exists(COD_ZIP) and os.path.getsize(COD_ZIP) > 1_000_000):
        print("GET", COD_URL)
        r = requests.get(COD_URL, timeout=900, headers=ua)
        r.raise_for_status()
        if r.content[:2] != b"PK":
            raise SystemExit(f"not a zip: {r.content[:60]!r}")
        with open(COD_ZIP + ".tmp", "wb") as fh:
            fh.write(r.content)
        os.replace(COD_ZIP + ".tmp", COD_ZIP)
        print(f"  {os.path.getsize(COD_ZIP):,} bytes")
    else:
        print("already have", COD_ZIP)

    os.makedirs(KONTUR_DIR, exist_ok=True)
    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 500_000:
        print("GET", KONTUR_URL)
        with requests.get(KONTUR_URL, timeout=1800, headers=ua, stream=True) as r:
            r.raise_for_status()
            with open(KONTUR_GZ + ".tmp", "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        with open(KONTUR_GZ + ".tmp", "rb") as fh:
            if fh.read(2) != b"\x1f\x8b":
                raise SystemExit("Kontur download is not gzip")
        os.replace(KONTUR_GZ + ".tmp", KONTUR_GZ)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR + ".tmp", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(KONTUR + ".tmp", KONTUR)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s).strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z]+", "", s).lower()


def build_units():
    import zipfile

    import geopandas as gpd
    import ua

    if not os.path.exists(COD_ZIP):
        raise SystemExit(f"missing {COD_ZIP} -- run sources/ua_geo.py --fetch first")
    names = zipfile.ZipFile(COD_ZIP).namelist()
    shp = [n for n in names if re.search(r"adm(in)?1[^/]*\.shp$", n, re.I)]
    if len(shp) != 1:
        raise SystemExit(f"expected one ADM1 shapefile in the zip, found {shp} among {names[:20]}")
    g = gpd.read_file(f"/vsizip/{COD_ZIP}/{shp[0]}")
    g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
    print(f"  COD-AB {shp[0]}: {len(g)} polygons")
    if len(g) != 27 or set(g["adm1_pcode"]) != set(ua.UNITS):
        raise SystemExit(f"COD-AB pcodes {sorted(g['adm1_pcode'])} are not ua.UNITS")
    bad = [(p, n) for p, n in zip(g["adm1_pcode"], g["adm1_name"]) if norm(n) != norm(ua.UNIT_NAMES[p])]
    if bad:
        raise SystemExit(f"COD-AB names disagree with ua.UNIT_NAMES: {bad}")
    print("    all 27 pcodes present and every name agrees with ua.UNIT_NAMES")
    out = g.rename(columns={"adm1_pcode": "unit", "adm1_name": "geo_name"})[["unit", "geo_name", "geometry"]]
    os.makedirs(GEO, exist_ok=True)
    out.to_file(UNITS_OUT, layer="oblasts", driver="GPKG")
    print(f"  wrote {UNITS_OUT}")
    return out


def build_grid(units):
    import geopandas as gpd
    import numpy as np
    import pyogrio
    import shapely
    import ua
    from scipy.stats import spearmanr

    if not os.path.exists(KONTUR):
        raise SystemExit(f"missing {KONTUR} -- run sources/ua_geo.py --fetch first")
    layers = list(pyogrio.list_layers(KONTUR)[:, 0])
    layer = "population" if "population" in layers else layers[0]
    hexes = gpd.read_file(KONTUR, layer=layer).to_crs(4326)
    print(f"\n  Kontur r8: {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")

    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()
    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes outside the 27 units ({hexes.loc[outside, 'population'].sum():,.0f} people)")
    hexes = hexes[~outside].copy()

    kon = hexes.groupby("unit")["population"].sum()
    print("\n  Kontur 2023 against Ukrstat (2022; Crimea and Sevastopol 2014):")
    ratios = {u: kon.get(u, 0.0) / ua.POP[u] for u in ua.UNITS}
    for u in sorted(ratios, key=ratios.get):
        print(f"    {ua.UNIT_NAMES[u]:<32} Ukrstat {ua.POP[u]:>10,}  Kontur {kon.get(u, 0):>10,.0f}  {ratios[u]:.2f}x")
    rho = spearmanr([ua.POP[u] for u in ua.UNITS], [kon.get(u, 0.0) for u in ua.UNITS]).correlation
    print(f"    Spearman over 27: {rho:+.3f}")
    if rho < 0.9 or min(ratios.values()) < 0.4 or max(ratios.values()) > 2.5:
        raise SystemExit("the population check failed -- the join is suspect")

    poly = units.set_index("unit")["geometry"]
    geom = hexes.geometry.to_numpy()
    who = hexes["unit"].to_numpy()
    out = np.empty(len(hexes), dtype=object)
    for unit, parent in poly.items():
        idx = np.flatnonzero(who == unit)
        if not idx.size:
            continue
        shapely.prepare(parent)
        inside = shapely.contains_properly(parent, geom[idx])
        out[idx[inside]] = geom[idx[inside]]
        edge = idx[~inside]
        if edge.size:
            out[edge] = shapely.intersection(parent, geom[edge])
    hexes["geometry"] = gpd.GeoSeries(out, crs=4326, index=hexes.index)
    hexes = hexes[~(hexes.geometry.is_empty | hexes.geometry.isna())].copy()
    hexes = hexes.rename(columns={"population": "pop"})[["unit", "pop", "geometry"]]
    missing = sorted(set(units["unit"]) - set(hexes["unit"]))
    if missing:
        raise SystemExit(f"units with no hex: {missing}")
    hexes.to_file(GRID_OUT, layer="grid400m", driver="GPKG")
    print(f"  wrote {GRID_OUT}  {len(hexes):,} hexes")


def main():
    if "--fetch" in sys.argv:
        fetch()
    units = build_units()
    build_grid(units)


if __name__ == "__main__":
    main()
