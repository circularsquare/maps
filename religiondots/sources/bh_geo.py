"""Bahrain: the four governorates of the 2014 reform (Capital, Muharraq, Northern, Southern), from
geoBoundaries.

Writes data/geo/bh/bh_units.gpkg (`unit`, `pop`, geometry) and data/geo/bh/bh_lookup.csv
(`geo_id` -> `unit`), from data/raw/bh/:

  * geoBoundaries gbOpen BHR ADM1 (OpenStreetMap, Wambacher, 2017; ODbL): four governorates,
    land only. COD-AB `cod-ab-bhr` is FAO GAUL 2008, from before 2014 abolished the Central
    Governorate, so it is not used;
  * Kontur Boundaries BH (HDX `kontur-boundaries-bahrain`, 2023-06-28; OSM, with territorial sea),
    a witness only.

The census and this layer each have four governorates with distinct names, so the join is on the
name (`NAMES`). Witness that neither key decides: each geoBoundaries governorate must lie at least
`INSIDE_MIN` inside the 2023 OSM governorate of the same name and less than `OTHER_MAX` in any
other one, which catches a pre-2014 line or a swapped label; `sources/bh_grid.py` then counts
Kontur's people per governorate against the census.

Usage:
    python sources/bh_geo.py --fetch    both files (0.3 MB and 10 KB)
    python sources/bh_geo.py            rebuild from data/raw/bh/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "bh")
GEO = os.path.join(ROOT, "data", "geo", "bh")
GB = os.path.join(RAW, "geoBoundaries-BHR-ADM1.geojson")
GB_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/BHR/ADM1/"
          "geoBoundaries-BHR-ADM1.geojson")
KB_GZ = os.path.join(RAW, "kontur_boundaries_BH_20230628.gpkg.gz")
KB = os.path.join(RAW, "kontur_boundaries_BH_20230628.gpkg")
KB_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_boundaries_BH_20230628.gpkg.gz")
NORMALIZED = os.path.join(ROOT, "data", "normalized", "bh.csv")
NAMES = {"Capital Governorate": "Capital", "Muharraq Governorate": "Muharraq",
         "Northern Governorate": "Northern", "Southern Governorate": "Southern"}
INSIDE_MIN = 0.90
OTHER_MAX = 0.08
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, dst, minsize in ((GB_URL, GB, 100_000), (KB_URL, KB_GZ, 5_000)):
        if os.path.exists(dst) and os.path.getsize(dst) > minsize:
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
    if not os.path.exists(KB):
        with gzip.open(KB_GZ, "rb") as src, open(KB + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KB + ".part", KB)


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not (os.path.exists(GB) and os.path.exists(KB)):
        fetch()
    gb = gpd.read_file(GB)
    if len(gb) != 4 or set(gb["shapeName"]) != set(NAMES):
        raise SystemExit(f"geoBoundaries BHR ADM1: {len(gb)} units, {sorted(gb['shapeName'])}")
    gb["unit"] = gb["shapeName"].map(NAMES)

    kb = gpd.read_file(KB)
    kb = kb[kb["admin_level"] == 4].copy()
    if set(kb["name_en"]) != set(NAMES):
        raise SystemExit(f"Kontur boundaries' governorates: {sorted(kb['name_en'])}")
    kb["unit"] = kb["name_en"].map(NAMES)

    eq = "EPSG:6933"
    g_m = gb.set_index("unit").geometry.to_crs(eq)
    k_m = kb.set_index("unit").geometry.to_crs(eq)
    print("  governorate   gB km2   inside OSM 2023's same name   most in any other")
    bad = []
    for u in NAMES.values():
        a = g_m[u]
        frac = {v: a.intersection(k_m[v]).area / a.area for v in NAMES.values()}
        own = frac.pop(u)
        other = max(frac.values())
        print(f"  {u:<12} {a.area / 1e6:>7.1f}   {own:>8.3f}                      {other:.3f}")
        if own < INSIDE_MIN or other > OTHER_MAX:
            bad.append(u)
    if bad:
        raise SystemExit(f"geoBoundaries and OSM 2023 disagree for {bad}")

    if not os.path.exists(NORMALIZED):
        raise SystemExit(f"missing {NORMALIZED}; run sources/bh.py first")
    rows = pd.read_csv(NORMALIZED, usecols=["geo_id", "count"])
    pop = rows.groupby("geo_id")["count"].sum()
    if set(pop.index) != set(NAMES.values()):
        raise SystemExit(f"normalized rows cover {sorted(pop.index)}")

    units = gb[["unit", "geometry"]].copy()
    units["pop"] = units["unit"].map(pop).astype(int)
    os.makedirs(GEO, exist_ok=True)
    units.to_file(os.path.join(GEO, "bh_units.gpkg"), layer="units", driver="GPKG")
    pd.DataFrame({"geo_id": list(NAMES.values()), "unit": list(NAMES.values())}).to_csv(
        os.path.join(GEO, "bh_lookup.csv"), index=False, encoding="utf-8")
    print(f"\nwrote {GEO}/bh_units.gpkg (4 governorates, {units['pop'].sum():,} people) and "
          f"bh_lookup.csv")


if __name__ == "__main__":
    main()
