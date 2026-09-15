"""Guinea-Bissau — the nine regiões and the placement grid.

Writes:
    data/geo/gw/gw_regions.gpkg     the 9 counted units (`units`)
    data/geo/gw/gw_hexes.gpkg       Kontur 400 m hexes with `unit` and `pop` (`place`)
    data/geo/gw/gw_lookup.csv       unit -> census nationals, Kontur population

Usage:
    python sources/gw_geo.py --fetch    COD-AB shapefile zip (~0.6 MB) + Kontur (~1.1 MB)
    python sources/gw_geo.py            rebuild from data/raw/gw/

THE BOUNDARIES ARE OCHA COD-AB GUINEA-BISSAU (`cod-ab-gnb`, version 01, from SALB, valid
2021-06-09), the shapefile bundle, read with `engine="fiona"`. ADM1 is the eight regiões plus
the Sector Autónomo de Bissau, which is the tier the census prints religion on.

THE JOIN IS BY NAME, folded, through a short alias list, and asserted to be a bijection. The
census writes `SAB` and `B. Bijagós` / `Bolama/Bijagós`; COD-AB's names are tried against
`Bissau` and `Bolama` for those two. Any census região matching zero or two polygons raises.

KONTUR IS NEEDED because SAB is a city of 362,699 nationals on under 80 km² while Bolama/
Bijagós is some eighty islands, about twenty of them inhabited, and Oio and Gabú are each
several thousand km² of villages.
"""

import csv
import glob
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
RAW = os.path.join(ROOT, "data", "raw", "gw")
SHP_DIR = os.path.join(RAW, "shp")
GEO = os.path.join(ROOT, "data", "geo", "gw")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "gw.csv")

ZIP_NAME = "gnb_admin_boundaries.shp.zip"
ZIP_URL = ("https://data.humdata.org/dataset/3db6297f-55d7-4e79-8969-281b1838ef79/resource/"
           "95d2ae47-3731-4cb6-9663-07fd82dcb1c8/download/gnb_admin_boundaries.shp.zip")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_GW_20231101.gpkg.gz")
GZ_NAME = "kontur_population_GW_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_GW_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "gw_regions.gpkg")
OUT_HEXES = os.path.join(GEO, "gw_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "gw_lookup.csv")

UNITS = 9
ALIASES = {"SAB": ["Bissau", "Sector Autonomo de Bissau", "SAB"],
           "Bolama/Bijagós": ["Bolama", "Bolama/Bijagos", "Bijagos", "Bolama Bijagos"]}

# Census March 2009 (nationals, uncorrected for the 4.6% omission) against Kontur 2023-11:
# 14.7 years at about 2.5% a year is ~1.44x, and the omission and the 0.5% outside the
# table push it up. The band is wide around that.
KONTUR_RATIO_MIN = 1.00
KONTUR_RATIO_MAX = 2.10

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
    _get(ZIP_URL, z, b"PK", 300_000)
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
    shps = sorted(glob.glob(os.path.join(SHP_DIR, "**", "*admin1.shp"), recursive=True))
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if len(shps) != 1:
        raise SystemExit(f"expected one *admin1.shp under {SHP_DIR}, found {shps} — run "
                         "sources/gw_geo.py --fetch")
    shp = shps[0]
    for p in (gpkg, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run sources/gw.py --fetch and "
                             "sources/gw_geo.py --fetch")
    os.makedirs(GEO, exist_ok=True)

    # ---- 1. the counted units, from the normalised file
    df = pd.read_csv(NORM, keep_default_na=False, na_values=[""])
    census = df[df["source_category"] == "Total"].groupby("geo_id")["count"].sum()
    if len(census) != UNITS:
        raise SystemExit(f"{len(census)} units in gw.csv, expected {UNITS}")

    # engine="fiona": pyogrio is the engine that has silently returned zero features here.
    g = gpd.read_file(shp, engine="fiona")
    print(f"COD-AB Guinea-Bissau admin1: {len(g)} features, crs={g.crs}")
    print(f"  names: {sorted(g['adm1_name'])}")
    if len(g) != UNITS:
        raise SystemExit(f"{shp} has {len(g)} features, expected {UNITS}")
    if "valid_on" in g.columns:
        print(f"  valid_on {sorted(set(map(str, g['valid_on'])))}")

    g["key"] = g["adm1_name"].map(norm)
    lut = {}
    for name in census.index:
        keys = {norm(name)} | {norm(a) for a in ALIASES.get(name, [])}
        hits = g.index[g["key"].isin(keys)].tolist()
        if len(hits) != 1:
            raise SystemExit(f"census região {name!r} matched {len(hits)} COD polygons "
                             f"(tried {sorted(keys)}): {sorted(g['adm1_name'])}")
        lut[name] = hits[0]
    if len(set(lut.values())) != UNITS:
        raise SystemExit("two census regiões matched the same polygon")
    print(f"  name join: {UNITS}/{UNITS} census regiões matched one polygon each, "
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
    print(f"wrote {OUT_UNITS} ({len(units)} regiões)")

    eq = units.to_crs(6933)
    km2 = (eq.geometry.area / 1e6).to_numpy()
    for (_i, r), a in sorted(zip(units.iterrows(), km2), key=lambda t: -t[1]):
        print(f"    {r['unit']:<16} {a:>8,.0f} km²  {int(r['census_pop']):>9,} nationals  "
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

    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)
    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    # SNAP, DO NOT DROP, THE HEXES JUST OFFSHORE (spec §12's archipelago rule). COD-AB's
    # coastline is coarse against Kontur's 400 m hexes: on the first run 330 hexes holding
    # 79,221 people (3.67%) had centroids in no região, 48,348 of them on Bissau's own
    # shoreline 47-1,180 m outside the SAB polygon, and the rest along the Bijagós and the
    # mangrove estuaries. 52% of that population is within 400 m of a região, 95.7% within
    # 1 km and 99.9% within 2 km, which is coastline resolution, not a broken join. Dropping
    # it would thin Bissau exactly where its people live. Snapped to the nearest região
    # within SNAP_M; anything farther is dropped, and more than 2% dropped stops the build.
    SNAP_M = 2_000
    outside = joined["unit"].isna()
    stray = float(pts.loc[outside, popcol].sum())
    print(f"  hexes whose centroid is in no região: {int(outside.sum()):,} "
          f"({stray:,.0f} people, {100.0 * stray / float(pts[popcol].sum()):.3f}%)")
    if outside.any():
        utm = units[["unit", "geometry"]].to_crs(32628)
        near = gpd.sjoin_nearest(pts.loc[outside].to_crs(32628), utm, how="left",
                                 max_distance=SNAP_M, distance_col="dist")
        near = near[~near.index.duplicated(keep="first")]
        got = near["unit"].notna()
        joined.loc[near.index[got], "unit"] = near.loc[got, "unit"].to_numpy()
        by = near[got].groupby("unit")[popcol].agg(["size", "sum"])
        print(f"  snapped {int(got.sum()):,} of them within {SNAP_M:,} m "
              f"({float(near.loc[got, popcol].sum()):,.0f} people): "
              + ", ".join(f"{u} {int(r['size'])}/{r['sum']:,.0f}"
                          for u, r in by.sort_values("sum", ascending=False).iterrows()))
    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    share_lost = lost / float(pts[popcol].sum())
    print(f"  still in no região after the snap: {int(outside.sum()):,} hexes "
          f"({lost:,.0f} people, {100.0 * share_lost:.3f}%), dropped")
    if share_lost > 0.02:
        raise SystemExit(f"{100 * share_lost:.2f}% of Kontur falls outside every região "
                         f"even after a {SNAP_M:,} m snap; check the coastline")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"regiões with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"regiões whose hexes sum to zero population: {zero}")
    print(f"  every one of the {UNITS} regiões has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    census_total = int(units["census_pop"].sum())
    ratio = tot / census_total
    print(f"\n  Kontur 2023 {tot:,.0f} vs the 2009 census nationals {census_total:,}: "
          f"ratio {ratio:.3f}")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit(f"ratio {ratio:.3f} outside [{KONTUR_RATIO_MIN}, "
                         f"{KONTUR_RATIO_MAX}], check the download")
    print("     used only as a WITHIN-região weight, so the level does not matter and the "
          "shape does.")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")

    per = per.rename(columns={"size": "hexes", "sum": "kontur_pop"})
    lk = units[["unit", "adm1_pcode", "census_pop"]].merge(
        per, left_on="unit", right_index=True, how="left")
    lk["kontur_over_census"] = lk["kontur_pop"] / lk["census_pop"]
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "adm1_pcode", "census_nationals_2009", "kontur_pop_2023", "hexes",
                    "kontur_over_census"])
        for r in lk.sort_values("unit").itertuples(index=False):
            w.writerow([r.unit, r.adm1_pcode, int(r.census_pop), round(r.kontur_pop, 1),
                        int(r.hexes), round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")
    print("\n  per-região Kontur/census ratio:")
    for r in lk.sort_values("kontur_over_census").itertuples(index=False):
        print(f"    {r.unit:<16} {r.kontur_over_census:5.2f}x  {int(r.hexes):>6,} hexes")


if __name__ == "__main__":
    main()
