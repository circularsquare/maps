"""Republic of the Congo — the twelve départements of 2007 and the placement grid.

Writes:
    data/geo/cg/cg_departements.gpkg   the 12 counted units (`units`)
    data/geo/cg/cg_hexes.gpkg          Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/cg/cg_lookup.csv          unit -> census population, Kontur population

Usage:
    python sources/cg_geo.py --fetch    COD-AB shapefile zip (~0.6 MB) + Kontur (~1.1 MB)
    python sources/cg_geo.py            rebuild from data/raw/cg/

THE BOUNDARIES ARE OCHA COD-AB CONGO (`cod-ab-cog`, version 01, boundaries of 2017, valid
2019-06-17, reviewed 2025-10-30), the shapefile bundle, read with `engine="fiona"`. ADM1 is the
twelve départements, which is the tier Tableau 11 prints: ten départements plus the communes of
Brazzaville and Pointe-Noire, which have been départements in their own right since 2002.

THE 2007 DÉPARTEMENTS ARE COD-AB'S TWELVE, AND THE 2024 REFORM IS NOT ON EITHER. Congo went from
12 to 15 départements by laws 24-2024 to 34-2024 of 8 October 2024 (Journal officiel no. 42,
17 October 2024, pp1304-1308, read): Djoué-Léfini (chef-lieu Odziba) is Ignié, Mayama, Vindza,
Kimba, Ngabé and the new district of Odziba, all Pool in 2007; Nkéni-Alima (Gamboma) is Gamboma,
Abala, Allembé, Ollombo, Ongogni and Makotimpoko, all Plateaux in 2007; Congo-Oubangui (Mossaka)
is Mossaka and Loukoléla (Cuvette in 2007), Liranga (Likouala in 2007) and the new district of
Bokoma. Law 29-2024 also redefines Brazzaville's territory. The 2007 membership is Tableau 1 of
the census brochure. This map draws the 2007 twelve on the pre-reform COD-AB layer; a reader
looking for the three new names will not find them.

TWO BOUNDARY DIFFERENCES ARE KNOWN AND LEFT. COD-AB's Brazzaville is 246 km2 and Pointe-Noire
208 km2, against 100 and 43.7 km2 in Tableau 3 of the 2007 brochure. Both cities have grown their
arrondissements since (Brazzaville from 7 to 9, Pointe-Noire from 4 to 6). A hex whose centroid is
inside today's city line but was Pool or Kouilou in 2007 takes the city's religion mix. The
people affected are a small share of either city and the city mixes are close to their
neighbours' in every large answer.

THE JOIN IS BY NAME, folded, with one alias: COD-AB spells the port `Point-Noire`. It is asserted
to be a bijection. Any unmatched name raises.

KONTUR IS NEEDED BECAUSE THE UNITS ARE HUGE AND HALF-EMPTY. Congo is 342,000 km2 and 3.7 million
people in 2007, 2.1 million of them in Brazzaville and Pointe-Noire. Likouala is 66,000 km2 of
swamp forest with its people along the Oubangui; Sangha is 56,000 km2 with Ouesso and the logging
towns. An equal spread over those would put most of their dots in uninhabited forest.
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
RAW = os.path.join(ROOT, "data", "raw", "cg")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "cg")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "cg.csv")

ZIP_NAME = "cog_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/762e2263-c6f1-4ef6-a3bc-48338a6484a8/resource/"
           "e1d2d25f-65ca-4e1f-9c61-6b9c8ccfc230/download/cog_admin_boundaries.shp.zip")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_CG_20231101.gpkg.gz")
GZ_NAME = "kontur_population_CG_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_CG_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "cg_departements.gpkg")
OUT_HEXES = os.path.join(GEO, "cg_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "cg_lookup.csv")

UNITS = 12
ALIASES = {"pointnoire": "pointenoire"}      # COD-AB `Point-Noire`

# Census April 2007 against Kontur 2023-11: sixteen and a half years. The RGPH-5 preliminary
# total for 2023 is 6,142,180 (§11aq), 1.66x the 2007 figure. The band is wide around it.
KONTUR_RATIO_MIN = 1.15
KONTUR_RATIO_MAX = 2.30

# Tableau 3's areas, km2, printed beside the 2007 populations; compared, not asserted.
T3_KM2 = {"Kouilou": 13650, "Niari": 25941.7, "Lékoumou": 20950, "Bouenza": 12265.4,
          "Pool": 33955.2, "Plateaux": 38400, "Cuvette": 48250, "Cuvette-Ouest": 26600,
          "Sangha": 55800, "Likouala": 66044, "Brazzaville": 100, "Pointe-Noire": 43.7}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    k = re.sub(r"[^a-z0-9]+", "", s.lower())
    return ALIASES.get(k, k)


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
    if head != magic:                                   # §5a: a 200 is not a download.
        raise SystemExit(f"{dst}: starts {head!r}, expected {magic!r}")
    os.replace(dst + ".part", dst)
    print(f"  got {os.path.getsize(dst):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR, exist_ok=True)
    z = os.path.join(RAW, ZIP_NAME)
    _get(ZIP_URL, z, b"PK", 400_000)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(SHP_DIR)

    gz = os.path.join(KONTUR, GZ_NAME)
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    _get(GZ_URL, gz, b"\x1f\x8b", 800_000)
    if not (os.path.exists(gpkg) and os.path.getsize(gpkg) > 800_000):
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
    shp = os.path.join(SHP_DIR, "cog_admin1.shp")
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if not os.path.exists(gpkg) and os.path.exists(os.path.join(KONTUR, GZ_NAME)):
        with gzip.open(os.path.join(KONTUR, GZ_NAME), "rb") as src, \
                open(gpkg + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(gpkg + ".part", gpkg)
    if not os.path.exists(shp) and os.path.exists(os.path.join(RAW, ZIP_NAME)):
        with zipfile.ZipFile(os.path.join(RAW, ZIP_NAME)) as zf:
            zf.extractall(SHP_DIR)
    for p in (shp, gpkg, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/cg.py --fetch and "
                             "sources/cg_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the counted units, from the normalised file
    df = pd.read_csv(NORM)
    census = df.groupby("geo_name")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} units in cg.csv, expected {UNITS}")

    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    g = gpd.read_file(shp, engine="fiona")
    print(f"COD-AB Congo admin1: {len(g)} features, crs={g.crs}")
    if len(g) != UNITS:
        raise SystemExit(f"{shp} has {len(g)} features, expected {UNITS}")
    print(f"  valid_on {sorted(set(map(str, g['valid_on'])))}, "
          f"version {sorted(set(map(str, g['version'])))}")

    g["key"] = g["adm1_name"].map(norm)
    lut = {}
    for name in census.index:
        hits = g.index[g["key"] == norm(name)].tolist()
        if len(hits) != 1:
            raise SystemExit(f"census département {name!r} matched {len(hits)} COD polygons: "
                             f"{sorted(g['adm1_name'])}")
        lut[name] = hits[0]
    if len(set(lut.values())) != UNITS:
        raise SystemExit("two census départements matched the same polygon")
    print(f"  name join: {UNITS}/{UNITS} census départements matched one polygon each, "
          "0 polygons unused")
    for name, i in sorted(lut.items()):
        if g.loc[i, "adm1_name"] != name:
            print(f"    census {name!r} <-> COD {g.loc[i, 'adm1_name']!r} "
                  f"({g.loc[i, 'adm1_pcode']})")

    units = g.loc[list(lut.values())].copy()
    units["unit"] = list(lut.keys())
    units["census_pop"] = units["unit"].map(census).astype(int)
    units = units[["unit", "adm1_pcode", "adm1_name", "census_pop", "geometry"]]
    units.to_file(OUT_UNITS, layer="departements", driver="GPKG")
    print(f"wrote {OUT_UNITS} ({len(units)} départements)")

    eq = units.to_crs(6933)
    km2 = (eq.geometry.area / 1e6).to_numpy()
    print(f"    {'département':<14}{'COD km2':>10}{'2007 km2':>10}{'people':>11}{'/km2':>9}")
    for (_i, r), a in sorted(zip(units.iterrows(), km2), key=lambda t: -t[1]):
        print(f"    {r['unit']:<14}{a:>10,.0f}{T3_KM2[r['unit']]:>10,.0f}"
              f"{int(r['census_pop']):>11,}{r['census_pop'] / a:>9,.1f}")

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
    print(f"  hexes whose centroid is in no département: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%), dropped")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"départements with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"départements whose hexes sum to zero population: {zero}")
    print(f"  every one of the {UNITS} départements has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    census_total = int(units["census_pop"].sum())
    ratio = tot / census_total
    print(f"\n  Kontur 2023 {tot:,.0f} vs the 2007 drawn population {census_total:,}: "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, "
                         f"{KONTUR_RATIO_MAX}], check the download")
    print("     used only as a WITHIN-département weight, so the level does not matter and "
          "the shape does.")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lk = units[["unit", "adm1_pcode", "census_pop"]].merge(
        per, left_on="unit", right_index=True, how="left")
    lk["kontur_over_census"] = lk["kontur_pop"] / lk["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm1_pcode", "census_pop_2007", "kontur_pop_2023", "hexes",
                    "kontur_over_census"])
        for r in lk.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.adm1_pcode, int(r.census_pop), round(r.kontur_pop, 1),
                        int(r.hexes), round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")
    print("\n  per-département Kontur/census ratio:")
    for r in lk.sort_values("kontur_over_census").itertuples(index=False):
        print(f"    {r.unit:<14} {r.kontur_over_census:5.2f}x  {int(r.hexes):>7,} hexes")


if __name__ == "__main__":
    main()
