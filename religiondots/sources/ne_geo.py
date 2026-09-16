"""Niger — the eight régions and the placement grid.

Writes:
    data/geo/ne/ne_regions.gpkg    the 8 counted units (`units`)
    data/geo/ne/ne_hexes.gpkg      Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/ne/ne_lookup.csv      unit -> census population, areas, Kontur population

Usage:
    python sources/ne_geo.py --fetch    COD-AB shapefile zip (~1.4 MB), Kontur (~9.9 MB)
    python sources/ne_geo.py            rebuild from data/raw/ne/

## THE LAYER

OCHA COD-AB Niger (`cod-ab-ner`, v02, valid from 20 July 2023) has the eight régions as ADM1,
NE001-NE008, under the census's names. They are joined by pcode, and the names under each pcode
are asserted. geoBoundaries gbOpen NER ADM1 is no witness: it has six features, with Tahoua and
Agadez merged, Zinder and Diffa merged, and Dosso spelt `Dossa` (read 2026-09-15).

## THE WITNESSES THE PCODE DOES NOT DECIDE

1. Area. Tableau 7 of the census volume prints each région's 2012 density; population over
   density is the area the office used. COD's national polygon is smaller than the office's
   national area (Niger's official 1,267,000 km2 is not a GIS figure), so each région's ratio
   is divided by the national one before it is banded.
2. People. Kontur's 2023 grid per région, against the 2012 count and divided by the national
   ratio. Eleven years of uneven growth separate the two, so the band is wide; it is there to
   catch a shuffled or empty unit, not to judge Kontur.

## KONTUR IS A WEIGHT INSIDE A RÉGION

Hexes join on their centroid; one falling just outside the national line is snapped to the
nearest région within SNAP_M rather than dropped.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from ne import A11, DENSITY_2012, DENSITY_2012_NIGER, REGIONS, TOTAL   # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "ne")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "ne")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "ne.csv")

OUT_UNITS = os.path.join(GEO, "ne_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "ne_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "ne_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
COD_ZIP = os.path.join(RAW, "ner_admin_boundaries.shp.zip")
KONTUR_GZ = os.path.join(KONTUR, "kontur_population_NE_20231101.gpkg.gz")
KONTUR_GPKG = KONTUR_GZ[:-3]
DOWNLOADS = {
    COD_ZIP: ("https://data.humdata.org/dataset/c0e0998c-b45a-4aea-ac06-c1de1d94e596/resource/"
              "b2a4cf8d-da46-4f52-bed0-865160470dac/download/ner_admin_boundaries.shp.zip",
              b"PK", 500_000),
    KONTUR_GZ: ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
                "kontur_population_NE_20231101.gpkg.gz", b"\x1f\x8b", 1_000_000),
}

UNITS = 8
# COD ADM1 pcode -> (COD's name, the census unit).
COD_ADM1 = {
    "NE001": ("Agadez", "Agadez"),
    "NE002": ("Diffa", "Diffa"),
    "NE003": ("Dosso", "Dosso"),
    "NE004": ("Maradi", "Maradi"),
    "NE005": ("Tahoua", "Tahoua"),
    "NE006": ("Tillabéri", "Tillaberi"),
    "NE007": ("Zinder", "Zinder"),
    "NE008": ("Niamey", "Niamey"),
}
ADM2_PER_REGION = {"Agadez": 6, "Diffa": 6, "Dosso": 8, "Maradi": 9, "Tahoua": 13,
                   "Tillaberi": 13, "Zinder": 11, "Niamey": 1}

AREA_TOL = 0.002                  # COD's ADM1 against its own national polygon
# COD area over the census's (population / Tableau 7 density), over the national ratio.
AREA_BAND = (0.90, 1.10)
# unit -> (lo, hi) where a measured difference is recorded. NIAMEY: COD's Ville de Niamey is its
# five arrondissements communaux, 557 km2 (measured 2.339x over the national ratio). INS gives
# 255 km2 twice: Tableau 7's density, and *Niamey en chiffres 2015* (the région's INS office,
# stat-niger.org), which prints "Superficie : 255 Km²" beside the 2012 population 1,026,848 and the
# same five arrondissements. Which outline is right is not settled here, and it moves almost
# nobody: the densest 255 km2 of COD's polygon hold 96.4% of its Kontur people (NIAMEY_CORE_MIN),
# and the polygon touches only Kollo département of Tillabéri.
AREA_PINNED = {"Niamey": (2.20, 2.50)}
NIAMEY_CORE_KM2 = 255
NIAMEY_CORE_MIN = 0.95            # share of Kontur's Niamey people in its densest 255 km2
SNAP_M = 500
KONTUR_NATIONAL = (1.0, 2.0)      # Kontur Nov 2023 over the December 2012 count
KONTUR_UNIT = (0.60, 1.60)        # per région, over the national ratio


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    for dst, (url, magic, min_size) in DOWNLOADS.items():
        if os.path.exists(dst) and os.path.getsize(dst) > min_size:
            print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers=UA, timeout=900)
        r.raise_for_status()
        if not r.content.startswith(magic) or len(r.content) < min_size:   # a 200 is not a file
            raise SystemExit(f"{os.path.basename(dst)}: {len(r.content):,} bytes starting "
                             f"{r.content[:16]!r}, expected {magic!r} and > {min_size:,}")
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {os.path.basename(dst)} ({len(r.content):,} bytes)")


def unpack():
    if not os.path.exists(os.path.join(SHP_DIR, "ner_admin1.shp")):
        if not os.path.exists(COD_ZIP):
            raise SystemExit(f"missing {COD_ZIP}; run with --fetch first")
        with zipfile.ZipFile(COD_ZIP) as zz:
            zz.extractall(SHP_DIR)
    if not os.path.exists(KONTUR_GPKG):
        if not os.path.exists(KONTUR_GZ):
            raise SystemExit(f"missing {KONTUR_GZ}; run with --fetch first")
        with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR_GPKG + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(KONTUR_GPKG + ".part", KONTUR_GPKG)
    with open(KONTUR_GPKG, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{KONTUR_GPKG} is not a GeoPackage")


def hex_units(pts, layer, crs_m="EPSG:32632"):
    """Centroid-in-polygon, then snap within SNAP_M. Returns (unit per hex, snapped, dropped)."""
    import geopandas as gpd

    joined = gpd.sjoin(pts, layer[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit_of = joined["unit"].copy()
    out = unit_of.isna()
    snapped = 0.0
    if out.any():
        near = gpd.sjoin_nearest(pts.loc[out].to_crs(crs_m), layer[["unit", "geometry"]].to_crs(crs_m),
                                 how="left", max_distance=SNAP_M, distance_col="dist_m")
        near = near[~near.index.duplicated(keep="first")]
        got = near["unit"].dropna()
        unit_of.loc[got.index] = got
        snapped = float(pts.loc[got.index, "pop"].sum())
    dropped = float(pts.loc[unit_of.isna(), "pop"].sum())
    return unit_of, snapped, dropped


def main():
    import geopandas as gpd
    import pandas as pd

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    unpack()
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM}; run sources/ne.py first")
    census = pd.read_csv(NORM, keep_default_na=False, na_values=[""]).groupby("geo_id")["count"].sum()
    want = {u: A11[u][6] for u in REGIONS}
    if census.to_dict() != want:
        raise SystemExit(f"ne.csv régions {census.to_dict()} are not Tableau A 11's {want}")
    eq = "EPSG:6933"

    # ---- 1. COD-AB ADM1, named by pcode
    a0 = gpd.read_file(os.path.join(SHP_DIR, "ner_admin0.shp"), engine="fiona")
    a1 = gpd.read_file(os.path.join(SHP_DIR, "ner_admin1.shp"), engine="fiona")
    a2 = gpd.read_file(os.path.join(SHP_DIR, "ner_admin2.shp"), engine="fiona")
    if len(a1) != UNITS or a1["adm1_pcode"].nunique() != UNITS:
        raise SystemExit(f"COD-AB Niger ADM1 has {len(a1)} features, expected {UNITS}")
    got = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    bad = {pc: (got.get(pc), nm) for pc, (nm, _u) in COD_ADM1.items() if fold(got.get(pc)) != fold(nm)}
    if bad:
        raise SystemExit(f"COD ADM1 names moved under their pcodes: {bad}")
    print(f"COD-AB Niger: {len(a1)} ADM1 units, version {a1['version'].iloc[0]}, valid_on "
          f"{a1['valid_on'].iloc[0]}, crs={a1.crs}")
    a1["unit"] = a1["adm1_pcode"].map(lambda pc: COD_ADM1[pc][1])
    a2["unit"] = a2["adm1_pcode"].map(lambda pc: COD_ADM1[pc][1])
    n2 = a2.groupby("unit").size().to_dict()
    if n2 != ADM2_PER_REGION:
        raise SystemExit(f"COD ADM2 départements per région {n2}, expected {ADM2_PER_REGION}")
    units = a1[["unit", "adm1_pcode", "geometry"]].copy()
    area = units.to_crs(eq).area / 1e6
    nat = float(a0.to_crs(eq).area.sum() / 1e6)
    if abs(area.sum() / nat - 1) > AREA_TOL:
        raise SystemExit(f"COD's régions sum to {area.sum():,.0f} km2 against its national {nat:,.0f}")
    units["area_km2"] = area.to_numpy()

    # ---- 2. area against Tableau 7's density
    census_area = {u: A11[u][6] / DENSITY_2012[u] for u in REGIONS}
    nat_census = TOTAL / DENSITY_2012_NIGER
    nat_ratio = nat / nat_census
    print(f"\n  COD national {nat:,.0f} km2 against the census's {nat_census:,.0f} "
          f"(x{nat_ratio:.3f})")
    print(f"  {'région':<11}{'census km2':>12}{'COD km2':>11}{'x':>7}{'/nat':>7}   départements")
    fails = []
    by = units.set_index("unit")
    for u in REGIONS:
        r = by.loc[u, "area_km2"] / census_area[u]
        rn = r / nat_ratio
        print(f"  {u:<11}{census_area[u]:>12,.0f}{by.loc[u, 'area_km2']:>11,.0f}{r:>7.3f}{rn:>7.3f}"
              f"   {n2[u]}")
        lo, hi = AREA_PINNED.get(u, AREA_BAND)
        if not lo <= rn <= hi:
            fails.append(f"{u}: COD area {rn:.3f}x the census's, over the national ratio, "
                         f"outside ({lo}, {hi})")

    # ---- 3. Kontur per région against the 2012 count
    hexes = geo_checks.read_layer(KONTUR_GPKG, "Kontur NE")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(units.crs)
    ktot = float(pts["pop"].sum())
    kn = ktot / TOTAL
    print(f"\nKontur NE 2023-11: {len(hexes):,} hexes, {ktot:,.0f} people; 2012 census {TOTAL:,} "
          f"(x{kn:.3f})")
    if not KONTUR_NATIONAL[0] <= kn <= KONTUR_NATIONAL[1]:
        fails.append(f"Kontur over the 2012 count {kn:.3f} outside {KONTUR_NATIONAL}")
    unit_of, snapped, dropped = hex_units(pts, units)
    per = pts.groupby(unit_of)["pop"].agg(["size", "sum"])
    print(f"  snapped {snapped:,.0f} people within {SNAP_M} m, dropped {dropped:,.0f} "
          f"({100.0 * dropped / ktot:.2f}%)")
    print(f"  {'région':<11}{'2012':>11}{'Kontur':>12}{'x':>7}{'/nat':>7}{'hexes':>8}")
    for u in REGIONS:
        k = float(per["sum"].get(u, 0.0))
        print(f"  {u:<11}{A11[u][6]:>11,}{k:>12,.0f}{k / A11[u][6]:>7.3f}{k / A11[u][6] / kn:>7.3f}"
              f"{int(per['size'].get(u, 0)):>8,}")
    # Niamey: the people in COD's polygon beyond a city of the census's area
    import numpy as np

    hex_km2 = float(hexes.loc[unit_of.eq("Niamey").to_numpy()].to_crs(eq).area.median() / 1e6)
    npop = np.sort(pts.loc[unit_of.eq("Niamey"), "pop"].to_numpy())[::-1]
    core = npop[:int(round(NIAMEY_CORE_KM2 / hex_km2))].sum() / npop.sum()
    print(f"  Niamey: the densest {NIAMEY_CORE_KM2} km2 of COD's {by.loc['Niamey', 'area_km2']:,.0f} "
          f"hold {100 * core:.1f}% of its {npop.sum():,.0f} Kontur people "
          f"({npop.sum() * (1 - core):,.0f} outside)")
    if core < NIAMEY_CORE_MIN:
        fails.append(f"Niamey: only {core:.3f} of Kontur's people in the densest {NIAMEY_CORE_KM2} km2")
    touching = sorted(a2.loc[(a2["unit"] != "Niamey") & a2.intersects(
        units.loc[units["unit"] == "Niamey", "geometry"].iloc[0]), "adm2_name"])
    print(f"  COD départements touching Niamey: {touching}")
    if fails:
        raise SystemExit("région layer checks failed:\n  " + "\n  ".join(fails))
    missing = sorted(set(REGIONS) - set(per.index))
    if missing:
        raise SystemExit(f"régions with no populated hex: {missing}")
    geo_checks.ratio_band(want, {u: per.loc[u, "sum"] / kn for u in REGIONS}, *KONTUR_UNIT,
                          what="région (Kontur 2023 over the 2012 count, over the national ratio)")

    # ---- 4. write
    keep = unit_of.notna()
    out = gpd.GeoDataFrame({"unit": unit_of[keep].to_numpy(), "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    units = units.copy()
    units["census_pop"] = units["unit"].map(want).astype("int64")
    units["kontur_pop"] = units["unit"].map(per["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(per["size"]).astype("int64")
    print(f"\n  drawn: COD-AB v02 ADM1, {len(out):,} hexes, {out['pop'].sum():,.0f} people")

    os.makedirs(GEO, exist_ok=True)
    cols = ["unit", "adm1_pcode", "census_pop", "kontur_pop", "hexes", "area_km2", "geometry"]
    units[cols].to_file(OUT_UNITS, layer="regions", driver="GPKG")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm1_pcode", "census_2012", "area_census_km2", "area_cod_km2",
                    "kontur_2023", "hexes"])
        for u in REGIONS:
            w.writerow([u, by.loc[u, "adm1_pcode"], want[u], round(census_area[u]),
                        round(float(by.loc[u, "area_km2"]), 1), round(float(per["sum"][u])),
                        int(per["size"][u])])
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_HEXES}\nwrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
