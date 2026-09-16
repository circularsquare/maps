"""The Gambia — the eight Local Government Areas and the placement grid.

Writes:
    data/geo/gm/gm_lgas.gpkg     the 8 counted units (`units`)
    data/geo/gm/gm_hexes.gpkg    Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/gm/gm_lookup.csv    unit -> census populations, areas, Kontur population, both layers

Usage:
    python sources/gm_geo.py --fetch    COD-AB shapefile zip (~1 MB), geoBoundaries ADM1 (~0.4 MB),
                                        Kontur (~0.5 MB)
    python sources/gm_geo.py            rebuild from data/raw/gm/

## TWO FILES DRAW THE EIGHT LGAs, AND THEY DISAGREE ABOUT KANIFING

OCHA COD-AB Gambia (`cod-ab-gmb`, v01, valid from 1 September 2022, source NDMA) has the eight
LGAs as ADM1 under their region names (Banjul City Council, Kanifing Municipal Council, West Coast
= Brikama, Lower River = Mansakonko, North Bank = Kerewan, Central River North = Kuntaur, Central
River South = Janjanbureh, Upper River = Basse). geoBoundaries gbOpen GMB ADM1 (World Bank, 2020)
has the same eight under the LGA names. The census's Table B.3 prints each LGA's area, and the two
files split the difference differently around Kanifing, the municipality between Banjul and
Brikama that holds a fifth of the country:

    Kanifing   census 75.55 km2   COD 93.7   geoBoundaries 52.9
    Banjul     census 12.23       COD  9.3   geoBoundaries  9.4

The line between Kanifing and Brikama runs through continuous built-up land, so the area alone
does not say where people are. Kontur's 2023 grid is read on both layers and compared, LGA by LGA,
with the 2024 census's preliminary counts (the nearest count to Kontur's date) and with 2013; the
layer whose Kanifing, Banjul and Brikama sit with the national ratio is drawn (`LAYER`).

## THE JOIN

COD's pcodes are mapped to LGA names by hand below (`COD_ADM1`), and the witness that the mapping
does not decide is geoBoundaries: each COD unit must overlap its LGA-named geoBoundaries feature
more than any other, which the two files' different naming and linework cannot make true by
construction. Central River North and South are Kuntaur and Janjanbureh, which the Afrobarometer
region lists agree with (sources.md §11aq).

## KONTUR IS A WEIGHT INSIDE AN LGA

Hexes join on their centroid; one falling just offshore (the Atlantic coast, the river banks and
its islands) is snapped to the nearest LGA within SNAP_M rather than dropped.
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

from gm import B1_2013, B3_AREA, LGAS   # noqa: E402  the census module's own transcription

RAW = os.path.join(ROOT, "data", "raw", "gm")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "gm")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "gm.csv")

OUT_UNITS = os.path.join(GEO, "gm_lgas.gpkg")
OUT_HEXES = os.path.join(GEO, "gm_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "gm_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}
COD_ZIP = os.path.join(RAW, "gmb_admin_boundaries.shp.zip")
GB_ADM1 = os.path.join(RAW, "geoBoundaries-GMB-ADM1.geojson")
KONTUR_GZ = os.path.join(KONTUR, "kontur_population_GM_20231101.gpkg.gz")
KONTUR_GPKG = KONTUR_GZ[:-3]
DOWNLOADS = {
    COD_ZIP: ("https://data.humdata.org/dataset/9f2ce756-2e50-4042-a952-32160977d223/resource/"
              "4fd16b04-3f6f-4a90-823c-d91da12e960a/download/gmb_admin_boundaries.shp.zip",
              b"PK", 500_000),
    GB_ADM1: ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/GMB/ADM1/"
              "geoBoundaries-GMB-ADM1.geojson", b"{", 100_000),
    KONTUR_GZ: ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
                "kontur_population_GM_20231101.gpkg.gz", b"\x1f\x8b", 100_000),
}

UNITS = 8
# COD ADM1 pcode -> (COD's name, the census LGA).
COD_ADM1 = {
    "GM01": ("Banjul City Council", "Banjul"),
    "GM05": ("Kanifing Municipal Council", "Kanifing"),
    "GM03": ("West Coast", "Brikama"),
    "GM08": ("Lower River", "Mansakonko"),
    "GM06": ("North Bank", "Kerewan"),
    "GM07": ("Central River North", "Kuntaur"),
    "GM04": ("Central River South", "Janjanbureh"),
    "GM02": ("Upper River", "Basse"),
}

# GBoS, *The Gambia 2024 Population and Housing Census: Preliminary Report* (UNFPA Gambia copy,
# `gambia.unfpa.org/sites/default/files/pub-pdf/2024-09/...`), Table 12, p.15 (PDF p.22).
CENSUS_2024 = {"Banjul": 26_461, "Kanifing": 379_348, "Brikama": 1_151_128,
               "Mansakonko": 90_624, "Kerewan": 248_475, "Kuntaur": 118_104,
               "Janjanbureh": 147_412, "Basse": 261_160}
CENSUS_2024_TOTAL = 2_422_712

LAYER = "cod"            # "cod" or "geoboundaries"; set from the Kontur comparison (module doc)
AREA_BAND = (0.70, 1.30)          # drawn LGA area against Table B.3's
IOU_MIN = 0.85                    # rural LGAs against their geoBoundaries namesake
IOU_SMALL = ("Banjul", "Kanifing")  # printed, not banded: the two files disagree here (doc)
AREA_TOL = 0.002                  # COD's ADM1 against its own national polygon
SNAP_M = 500
# Kontur Nov 2023 against the April 2024 count. Measured 1.161 on the first build (2,811,983
# against 2,422,712): Kontur runs above this census, and the band is there to catch a wrong or
# empty download, not to judge Kontur's level, which only weights within an LGA.
KONTUR_NATIONAL_2024 = (0.85, 1.30)
# Per LGA on the drawn layer, as a multiple of that national ratio, so Kontur's overall excess
# does not read as a misplaced unit.
KONTUR_UNIT_2024 = (0.75, 1.30)


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
    if not os.path.exists(os.path.join(SHP_DIR, "gmb_admin1.shp")):
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


def hex_units(pts, layer, crs_utm="EPSG:32628"):
    """Centroid-in-polygon, then snap within SNAP_M. Returns (unit per hex, snapped, dropped)."""
    import geopandas as gpd

    joined = gpd.sjoin(pts, layer[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit_of = joined["unit"].copy()
    out = unit_of.isna()
    snapped = 0.0
    if out.any():
        near = gpd.sjoin_nearest(pts.loc[out].to_crs(crs_utm),
                                 layer[["unit", "geometry"]].to_crs(crs_utm),
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
    from shapely.ops import unary_union

    import geo_checks

    if "--fetch" in sys.argv:
        fetch()
    unpack()
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM}; run sources/gm.py first")
    census = pd.read_csv(NORM, keep_default_na=False, na_values=[""]).groupby(
        "geo_id")["count"].sum()
    if sorted(census.index) != sorted(LGAS) or census.to_dict() != B1_2013:
        raise SystemExit(f"gm.csv LGAs {census.to_dict()} are not Table B.1's {B1_2013}")
    if sum(CENSUS_2024.values()) != CENSUS_2024_TOTAL:
        raise SystemExit("the 2024 preliminary LGA figures do not sum to their total")
    eq = "EPSG:6933"

    # ---- 1. COD-AB ADM1, named by pcode
    a0 = gpd.read_file(os.path.join(SHP_DIR, "gmb_admin0.shp"), engine="fiona")
    a1 = gpd.read_file(os.path.join(SHP_DIR, "gmb_admin1.shp"), engine="fiona")
    if len(a1) != UNITS or a1["adm1_pcode"].nunique() != UNITS:
        raise SystemExit(f"COD-AB Gambia ADM1 has {len(a1)} features, expected {UNITS}")
    got = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    bad = {pc: (got.get(pc), nm) for pc, (nm, _l) in COD_ADM1.items() if fold(got.get(pc)) != fold(nm)}
    if bad:
        raise SystemExit(f"COD ADM1 names moved under their pcodes: {bad}")
    print(f"COD-AB Gambia: {len(a1)} ADM1 units, version {a1['version'].iloc[0]}, valid_on "
          f"{a1['valid_on'].iloc[0]}, crs={a1.crs}")
    a1["unit"] = a1["adm1_pcode"].map(lambda pc: COD_ADM1[pc][1])
    cod = a1[["unit", "geometry"]].copy()
    cod_area = cod.to_crs(eq).area / 1e6
    nat = float(a0.to_crs(eq).area.sum() / 1e6)
    if abs(cod_area.sum() / nat - 1) > AREA_TOL:
        raise SystemExit(f"COD's LGAs sum to {cod_area.sum():,.0f} km2 against its national {nat:,.0f}")
    cod["area_km2"] = cod_area.to_numpy()

    # ---- 2. geoBoundaries ADM1, and the join witness
    gb = geo_checks.read_layer(GB_ADM1, "geoBoundaries GMB ADM1")
    if len(gb) != UNITS:
        raise SystemExit(f"geoBoundaries GMB ADM1 has {len(gb)} features, expected {UNITS}")
    gb["unit"] = [next((l for l in LGAS if fold(l) == fold(n)), None) for n in gb["shapeName"]]
    if gb["unit"].isna().any() or gb["unit"].nunique() != UNITS:
        raise SystemExit(f"geoBoundaries names {sorted(gb['shapeName'])} are not the eight LGAs")
    gb = gb[["unit", "geometry"]].copy()
    gb["area_km2"] = (gb.to_crs(eq).area / 1e6).to_numpy()

    ce = cod.set_index("unit").to_crs(eq)
    ge = gb.set_index("unit").to_crs(eq)
    print("\n  LGA          B.3 km2    COD km2  (x B.3)   gB km2  (x B.3)   IoU COD-gB  best other")
    fails = []
    for u in LGAS:
        a = ce.geometry[u]
        ious = {v: a.intersection(g).area / a.union(g).area for v, g in ge.geometry.items()
                if a.intersects(g)}
        own = ious.get(u, 0.0)
        other = max(((v, x) for v, x in ious.items() if v != u), key=lambda t: t[1],
                    default=("-", 0.0))
        rc, rg = ce.loc[u, "area_km2"] / B3_AREA[u], ge.loc[u, "area_km2"] / B3_AREA[u]
        print(f"  {u:<12}{B3_AREA[u]:>9,.2f}{ce.loc[u, 'area_km2']:>11,.1f}  ({rc:4.2f})"
              f"{ge.loc[u, 'area_km2']:>9,.1f}  ({rg:4.2f})   {own:10.3f}  {other[0]} {other[1]:.3f}")
        if own <= other[1]:
            fails.append(f"{u}: COD overlaps geoBoundaries {other[0]} more than its namesake")
        if u not in IOU_SMALL and own < IOU_MIN:
            fails.append(f"{u}: IoU {own:.3f} < {IOU_MIN}")
        r = rc if LAYER == "cod" else rg
        if not AREA_BAND[0] <= r <= AREA_BAND[1]:
            fails.append(f"{u}: drawn area {r:.2f}x Table B.3's")
    if fails:
        raise SystemExit("LGA layer checks failed:\n  " + "\n  ".join(fails))

    # ---- 3. Kontur on both layers, against 2013 and the 2024 count
    hexes = geo_checks.read_layer(KONTUR_GPKG, "Kontur GM")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    pts = gpd.GeoDataFrame({"pop": hexes[popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(cod.crs)
    ktot = float(pts["pop"].sum())
    print(f"\nKontur GM 2023-11: {len(hexes):,} hexes, {ktot:,.0f} people; 2024 census "
          f"{CENSUS_2024_TOTAL:,} (x{ktot / CENSUS_2024_TOTAL:.3f}), 2013 {sum(B1_2013.values()):,}")
    if not KONTUR_NATIONAL_2024[0] <= ktot / CENSUS_2024_TOTAL <= KONTUR_NATIONAL_2024[1]:
        raise SystemExit(f"Kontur against the 2024 count outside {KONTUR_NATIONAL_2024}")
    per_layer = {}
    for name, layer in (("cod", cod), ("geoboundaries", gb)):
        unit_of, snapped, dropped = hex_units(pts, layer)
        per_layer[name] = (unit_of, pts.groupby(unit_of)["pop"].sum())
        print(f"  {name}: snapped {snapped:,.0f} people within {SNAP_M} m, dropped {dropped:,.0f}")
    nat24 = ktot / CENSUS_2024_TOTAL
    print(f"\n  Kontur per LGA over the 2024 count, divided by the national {nat24:.3f}:")
    print(f"  {'LGA':<12}{'2024':>11}{'COD':>11}{'x':>7}{'gB':>11}{'x':>7}{'2013':>11}")
    for u in LGAS:
        kc, kg = per_layer["cod"][1].get(u, 0.0), per_layer["geoboundaries"][1].get(u, 0.0)
        print(f"  {u:<12}{CENSUS_2024[u]:>11,}{kc:>11,.0f}{kc / CENSUS_2024[u] / nat24:>7.2f}"
              f"{kg:>11,.0f}{kg / CENSUS_2024[u] / nat24:>7.2f}{B1_2013[u]:>11,}")

    # ---- 4. the drawn layer
    units = cod if LAYER == "cod" else gb
    unit_of, per = per_layer[LAYER]
    keep = unit_of.notna()
    out = gpd.GeoDataFrame({"unit": unit_of[keep].to_numpy(), "pop": pts.loc[keep, "pop"].to_numpy()},
                           geometry=hexes.to_crs(units.crs).geometry[keep.to_numpy()].to_numpy(),
                           crs=units.crs)
    sizes = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(LGAS) - set(sizes.index))
    if missing:
        raise SystemExit(f"LGAs with no populated hex: {missing}")
    geo_checks.ratio_band(CENSUS_2024, {u: sizes.loc[u, "sum"] / nat24 for u in LGAS},
                          *KONTUR_UNIT_2024,
                          what="LGA (Kontur 2023 over the 2024 count, over the national ratio)")
    units = units.copy()
    units["census_pop"] = units["unit"].map(B1_2013).astype("int64")
    units["census_2024"] = units["unit"].map(CENSUS_2024).astype("int64")
    units["kontur_pop"] = units["unit"].map(sizes["sum"]).round().astype("int64")
    units["hexes"] = units["unit"].map(sizes["size"]).astype("int64")
    print(f"\n  drawn: {LAYER}, {len(out):,} hexes, {out['pop'].sum():,.0f} people")

    os.makedirs(GEO, exist_ok=True)
    cols = ["unit", "census_pop", "census_2024", "kontur_pop", "hexes", "area_km2", "geometry"]
    units[cols].to_file(OUT_UNITS, layer="lgas", driver="GPKG")
    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "census_2013", "census_2024_prelim", "area_b3_km2", "area_cod_km2",
                    "area_gb_km2", "kontur_2023_cod", "kontur_2023_gb", "drawn"])
        for u in LGAS:
            w.writerow([u, B1_2013[u], CENSUS_2024[u], B3_AREA[u],
                        round(float(ce.loc[u, "area_km2"]), 2), round(float(ge.loc[u, "area_km2"]), 2),
                        round(float(per_layer["cod"][1].get(u, 0.0))),
                        round(float(per_layer["geoboundaries"][1].get(u, 0.0))), LAYER])
    print(f"\nwrote {OUT_UNITS}\nwrote {OUT_HEXES}\nwrote {OUT_LOOKUP}")


if __name__ == "__main__":
    main()
