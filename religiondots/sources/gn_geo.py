"""Guinea — the eight régions administratives and the placement grid.

Writes:
    data/geo/gn/gn_regions.gpkg     the 8 counted units (`units`)
    data/geo/gn/gn_hexes.gpkg       Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/gn/gn_lookup.csv       unit -> census population, Kontur population

Usage:
    python sources/gn_geo.py --fetch    COD-AB shapefile zip (~2.7 MB) + Kontur (~7.4 MB)
    python sources/gn_geo.py            rebuild from data/raw/gn/

THE BOUNDARIES ARE OCHA COD-AB GUINEA (`cod-ab-gin`), the shapefile bundle on §12's Chile
rule, read with `engine="fiona"`. ADM1 is the eight régions administratives, which is
exactly the tier the census prints religion on: seven régions plus the Conakry special zone.

THE JOIN IS BY NAME, folded, and asserted to be a bijection. The census report writes
`N'Zérékoré` with an apostrophe (and `Nzérékoré` in its own chart two pages earlier);
folding out every non-letter makes them one key. Any unmatched name raises.

KONTUR IS NEEDED BECAUSE THE UNITS ARE HUGE. Eight units over about 245,000 km² is some
30,000 km² each. Conakry is a peninsula of 450 km² holding 1.66 million people (Tableau
2.08 of the census report) while Kankan is 72,000-odd km² of savannah with its people along
the Niger and the Milo, and Forest Guinea, where nearly every Christian and animist in the
country lives, is mountain and rainforest with the population in a few towns and the valleys.
"""

import csv
import gzip
import os
import re
import shutil
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gn")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "gn")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "gn.csv")

ZIP_NAME = "gin_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/f814a950-4d4e-4f46-a880-4da5522f14c4/resource/"
           "873df02f-febb-4916-abd2-c63f1831abe4/download/gin_admin_boundaries.shp.zip")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_GN_20231101.gpkg.gz")
GZ_NAME = "kontur_population_GN_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_GN_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "gn_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "gn_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "gn_lookup.csv")

UNITS = 8

# Census 2014 (April) against Kontur 2023-11: nine and a half years at the census's own
# 2.2% a year intercensal growth (Tableau 2.08) is about 1.23x. The band is wide around it.
KONTUR_RATIO_MIN = 0.90
KONTUR_RATIO_MAX = 1.70

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _get(url, dst, magic, min_size):
    import requests

    if os.path.exists(dst) and os.path.getsize(dst) > min_size:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers=UA)
    r.raise_for_status()
    with open(dst + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    with open(dst + ".part", "rb") as fh:
        head = fh.read(len(magic))
    # §5a: a 200 is not a download.
    if head != magic:
        raise SystemExit(f"{dst}: starts {head!r}, expected {magic!r}")
    os.replace(dst + ".part", dst)
    print(f"  got {os.path.getsize(dst):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    z = os.path.join(RAW, ZIP_NAME)
    _get(ZIP_URL, z, b"PK", 1_000_000)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(SHP_DIR)

    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    _get(GZ_URL, gz, b"\x1f\x8b", 5_000_000)
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000):
        with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    shp = os.path.join(SHP_DIR, "gin_admin1.shp")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    for p in (shp, gpkg, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/gn.py --fetch and "
                             "sources/gn_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the counted units, from the normalised file
    df = pd.read_csv(NORM)
    census = df.groupby("geo_name")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} units in gn.csv, expected {UNITS}")

    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    g = gpd.read_file(shp, engine="fiona")
    print(f"COD-AB Guinea admin1: {len(g)} features, crs={g.crs}")
    print(f"  columns: {[c for c in g.columns if c != 'geometry']}")
    if len(g) != UNITS:
        raise SystemExit(f"{shp} has {len(g)} features, expected {UNITS}")
    if "valid_on" in g.columns:
        print(f"  valid_on {sorted(set(map(str, g['valid_on'])))}")

    g["key"] = g["adm1_name"].map(norm)
    lut = {}
    for name in census.index:
        hits = g.index[g["key"] == norm(name)].tolist()
        if len(hits) != 1:
            raise SystemExit(f"census région {name!r} matched {len(hits)} COD polygons: "
                             f"{sorted(g['adm1_name'])}")
        lut[name] = hits[0]
    if len(set(lut.values())) != UNITS:
        raise SystemExit("two census régions matched the same polygon")
    print(f"  name join: {UNITS}/{UNITS} census régions matched one polygon each, "
          "0 polygons unused")
    for name, i in sorted(lut.items()):
        if g.loc[i, "adm1_name"] != name:
            print(f"    census {name!r} <-> COD {g.loc[i, 'adm1_name']!r} "
                  f"({g.loc[i, 'adm1_pcode']})")

    units = g.loc[list(lut.values())].copy()
    units["unit"] = list(lut.keys())
    units["census_pop"] = units["unit"].map(census).astype(int)
    units = units[["unit", "adm1_pcode", "adm1_name", "census_pop", "geometry"]]
    units.to_file(OUT_UNITS, layer="regions", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units)} régions)")

    eq = units.to_crs(6933)
    km2 = (eq.geometry.area / 1e6).to_numpy()
    for (_i, r), a in sorted(zip(units.iterrows(), km2), key=lambda t: -t[1]):
        print(f"    {r['unit']:<12} {a:>9,.0f} km²  {int(r['census_pop']):>10,} people  "
              f"{r['census_pop'] / a:>8,.1f}/km²")

    # ---- 2. Kontur, joined on hex CENTROIDS
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"\nKontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    # Take the centroid in the CRS the hexes were tiled in, then reproject the POINTS.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes whose centroid is in no région: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%), dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"régions with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"régions whose hexes sum to zero population: {zero}")
    print(f"  every one of the {UNITS} régions has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    census_total = int(units["census_pop"].sum())
    ratio = tot / census_total
    print(f"\n  Kontur 2023 {tot:,.0f} vs the 2014 drawn population {census_total:,}: "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, "
                         f"{KONTUR_RATIO_MAX}], check the download")
    print("     used only as a WITHIN-région weight, so the level does not matter and the "
          "shape does.")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lk = units[["unit", "adm1_pcode", "census_pop"]].merge(
        per, left_on="unit", right_index=True, how="left")
    lk["kontur_over_census"] = lk["kontur_pop"] / lk["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm1_pcode", "census_pop_2014", "kontur_pop_2023", "hexes",
                    "kontur_over_census"])
        for r in lk.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.adm1_pcode, int(r.census_pop), round(r.kontur_pop, 1),
                        int(r.hexes), round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")
    print("\n  per-région Kontur/census ratio:")
    for r in lk.sort_values("kontur_over_census").itertuples(index=False):
        print(f"    {r.unit:<12} {r.kontur_over_census:5.2f}x  {int(r.hexes):>7,} hexes")


if __name__ == "__main__":
    main()
