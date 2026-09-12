"""Guyana — boundaries and placement grid for the 10 administrative regions.

Writes:
    data/geo/gy/gy_regions.gpkg          the 10 regions (`units`)
    data/geo/gy/gy_grid_400m.gpkg        Kontur H3 r8 hexes with `unit` and `pop` (`place`)
    data/geo/gy/gy_lookup.csv            region number -> name, area, census and Kontur pop

Usage:
    python sources/gy_geo.py --fetch     two downloads, ~1 MB total
    python sources/gy_geo.py             rebuild from data/raw/gy/ and data/geo/kontur/

THE JOIN IS A PUBLISHED STANDARD, WHICH IS THE BEST CASE §12 DESCRIBES AND THE FIRST TIME
THIS PROJECT HAS HAD IT FOR FREE. The census names no region at all — Table 2.19's columns
are `Region 1` … `Region 10` and the compendium never prints `Barima-Waini` anywhere in its
66 pages — so there is no name to join on and nothing to transliterate. geoBoundaries carries
`shapeISO`, and **ISO 3166-2:GY is exactly the ten regions in region-number order**: GY-BA is
Region 1, GY-PM is Region 2, through GY-UD for Region 10. So the mapping below is a
transcription of a standard rather than a guess, and `--check-iso` re-derives it.

AND THE BOUNDARY FILE'S NAMES ARE WRONG IN ONE PLACE, WHICH IS WHY THEY ARE NOT USED AS KEYS.
geoBoundaries spells Region 1 **`Barina-Waini`** — an `n` for the `m` of Barima, the river.
§12's Chile rule says take names from the statistical source and not the boundary file; here
the statistical source publishes no names, so the official ISO 3166-2 spellings are used for
display and the file's own strings are used for nothing. Had the join been by name it would
have failed on exactly one region and looked like a vintage problem.

THE INDEPENDENT CHECK IS KONTUR AGAINST THE CENSUS, PER REGION. The ISO join determines which
polygon is Region N; it does not determine how many people a modelled population surface puts
there. A correct assignment keeps every region's Kontur/census ratio inside a tight band; a
scrambled one pairs Georgetown's 312,000 with Potaro-Siparuni's 11,000 and scatters the ratios
over orders of magnitude. That is §9i's North Macedonia check and it is the whole verification
here, because there is no population column in the boundary file and no code on the census
side to cross-match.

GUYANA IS THE CASE KONTUR EXISTS FOR. 90% of the country is forest and 90% of the people are
on a coastal strip a few km deep: Region 4 is 41.7% of the population on 1.0% of the land,
while Region 8 is 11,077 people spread over 20,000 km². Placing uniformly inside regions
would wash the interior in evenly spaced dots and — because Regions 1, 7, 8 and 9 are the
Amerindian interior and heavily Catholic and Anglican — that wash would be a colour, and the
loudest thing on the map. §12's Kenya entry, in the country it is most true of.

Inland water needs no special handling for the same reason it did not in Kenya: hexes exist
only where people are, so the Essequibo's islands are present and its water is not.
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gy")
OUT_DIR = os.path.join(ROOT, "data", "geo", "gy")
KONTUR_DIR = os.path.join(ROOT, "data", "geo", "kontur")

UNITS_OUT = os.path.join(OUT_DIR, "gy_regions.gpkg")
GRID_OUT = os.path.join(OUT_DIR, "gy_grid_400m.gpkg")
LOOKUP = os.path.join(OUT_DIR, "gy_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "gy.csv")

ADM1_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
            "GUY/ADM1/geoBoundaries-GUY-ADM1.geojson")
ADM1 = os.path.join(RAW, "geoBoundaries-GUY-ADM1.geojson")

KONTUR_URL = ("https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_population_GY_20231101.gpkg.gz")
KONTUR_GZ = os.path.join(KONTUR_DIR, "kontur_population_GY_20231101.gpkg.gz")
KONTUR = os.path.join(KONTUR_DIR, "kontur_population_GY_20231101.gpkg")

REGIONS = 10

# ISO 3166-2:GY, which is the ten regions in region-number order. This is the join.
# The names are the official ISO ones; geoBoundaries' `shapeName` is used for nothing
# because it misspells Region 1 (see the module docstring).
ISO_REGION = {
    "GY-BA": (1, "Barima-Waini"),
    "GY-PM": (2, "Pomeroon-Supenaam"),
    "GY-ES": (3, "Essequibo Islands-West Demerara"),
    "GY-DE": (4, "Demerara-Mahaica"),
    "GY-MA": (5, "Mahaica-Berbice"),
    "GY-EB": (6, "East Berbice-Corentyne"),
    "GY-CU": (7, "Cuyuni-Mazaruni"),
    "GY-PT": (8, "Potaro-Siparuni"),
    "GY-UT": (9, "Upper Takutu-Upper Essequibo"),
    "GY-UD": (10, "Upper Demerara-Berbice"),
}

# A Kontur/census ratio outside this band on any region means the polygons are not the
# regions the counts think they are. Kontur is a 2023 model of a 2012 census population and
# Guyana's population has been roughly flat, so the band is centred near 1; it is deliberately
# wide because a modelled surface over near-empty interior regions is genuinely poor there,
# and narrow enough that any swap of two regions breaks it (§9i).
RATIO_LO, RATIO_HI = 0.30, 3.0


def fetch():
    import gzip
    import shutil

    import requests

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(KONTUR_DIR, exist_ok=True)

    if not os.path.exists(ADM1) or os.path.getsize(ADM1) < 100_000:
        print("GET", ADM1_URL)
        r = requests.get(ADM1_URL, timeout=300, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        # §5a. A GeoJSON FeatureCollection, not an HTML error page.
        if b'"FeatureCollection"' not in r.content[:400]:
            raise SystemExit(f"not GeoJSON -- starts {r.content[:80]!r}")
        with open(ADM1, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(ADM1):,} bytes")

    if os.path.exists(KONTUR) and os.path.getsize(KONTUR) > 1_000_000:
        print("already have", KONTUR)
        return
    if not os.path.exists(KONTUR_GZ) or os.path.getsize(KONTUR_GZ) < 100_000:
        print("GET", KONTUR_URL)
        r = requests.get(KONTUR_URL, timeout=600, headers={"User-Agent": "religiondots/1.0"})
        r.raise_for_status()
        if r.content[:2] != b"\x1f\x8b":
            raise SystemExit(f"not gzip: first bytes {r.content[:40]!r}")
        with open(KONTUR_GZ, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(KONTUR_GZ):,} bytes")
    with gzip.open(KONTUR_GZ, "rb") as src, open(KONTUR, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  decompressed to {KONTUR}  {os.path.getsize(KONTUR):,} bytes")


def fiona_layers(path):
    import fiona
    return list(fiona.listlayers(path))


def build_units():
    import geopandas as gpd

    if not os.path.exists(ADM1):
        raise SystemExit(f"missing {ADM1} -- run: python sources/gy_geo.py --fetch")

    # §12's Chile rule: assert the feature count, not the absence of an exception.
    gdf = gpd.read_file(ADM1)
    if len(gdf) != REGIONS:
        raise SystemExit(f"{len(gdf)} features in {ADM1}, expected {REGIONS}")
    print(f"  geoBoundaries GUY ADM1: {len(gdf)} features")

    if "shapeISO" not in gdf.columns:
        raise SystemExit("no shapeISO column -- geoBoundaries has changed its schema and "
                         "the ISO 3166-2 join is gone; re-derive the region numbers before "
                         "trusting anything downstream")

    got = list(gdf["shapeISO"])
    want = set(ISO_REGION)
    if sorted(got) != sorted(want):
        raise SystemExit(f"shapeISO set mismatch.\n  file: {sorted(got)}\n  ISO : {sorted(want)}")
    if len(set(got)) != REGIONS:
        raise SystemExit(f"shapeISO is not unique: {got}")

    gdf["unit"] = [str(ISO_REGION[c][0]) for c in gdf["shapeISO"]]
    gdf["name"] = [ISO_REGION[c][1] for c in gdf["shapeISO"]]
    gdf["iso"] = gdf["shapeISO"]

    mis = [(a, b) for a, b in zip(gdf["shapeName"], gdf["name"]) if a != b]
    if mis:
        print(f"    {len(mis)} region name(s) differ from ISO 3166-2 and the ISO spelling "
              "is used:")
        for a, b in mis:
            print(f"      geoBoundaries {a!r} -> {b!r}")

    gdf = gdf.to_crs(4326)
    gdf["area_km2"] = gdf.to_crs(6933).area / 1e6
    gdf = gdf.sort_values("unit", key=lambda s: s.astype(int)).reset_index(drop=True)
    return gdf[["unit", "name", "iso", "area_km2", "geometry"]]


def census_pop():
    """Region -> census population, from the normalised file. The check's other side."""
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run: python sources/gy.py")
    pop = {}
    with open(NORM, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["geo_level"] == "region":
                pop[r["geo_id"]] = pop.get(r["geo_id"], 0) + int(r["count"])
    if len(pop) != REGIONS:
        raise SystemExit(f"{len(pop)} regions in {NORM}, expected {REGIONS}")
    return pop


def build_grid(units):
    import geopandas as gpd
    import numpy as np
    import shapely

    if not os.path.exists(KONTUR):
        raise SystemExit(f"missing {KONTUR} -- run: python sources/gy_geo.py --fetch")

    layers = fiona_layers(KONTUR)
    layer = "population" if "population" in layers else layers[0]
    print(f"  reading Kontur r8 hexes from layer {layer!r}…")
    hexes = gpd.read_file(KONTUR, layer=layer)
    if len(hexes) == 0:
        raise SystemExit("Kontur read returned ZERO features -- §12's pyogrio/fiona trap; "
                         "retry with engine='fiona'")
    print(f"    {len(hexes):,} hexes, {hexes['population'].sum():,.0f} people")
    hexes = hexes.to_crs(4326)

    # Join on hex CENTRES so no hex is split between two regions.
    centres = gpd.GeoDataFrame(geometry=hexes.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(centres, units[["unit", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    hexes["unit"] = hit["unit"].to_numpy()

    outside = hexes["unit"].isna()
    print(f"    {outside.sum():,} hexes ({100.0 * outside.mean():.2f}%) fall outside every "
          f"region, {hexes.loc[outside, 'population'].sum():,.0f} people — the coastal and "
          "border strip Kontur rounds outwards")
    hexes = hexes[~outside].copy()

    print(f"  clipping {len(hexes):,} hexes to their region…")
    poly = units.set_index("unit")["geometry"]
    geom = hexes.geometry.to_numpy()
    who = hexes["unit"].to_numpy()
    out = np.empty(len(hexes), dtype=object)
    n_clipped = 0
    for unit, parent in poly.items():
        idx = np.flatnonzero(who == unit)
        if idx.size == 0:
            continue
        shapely.prepare(parent)
        inside = shapely.contains_properly(parent, geom[idx])
        out[idx[inside]] = geom[idx[inside]]
        edge = idx[~inside]
        if edge.size:
            out[edge] = shapely.intersection(parent, geom[edge])
            n_clipped += edge.size
    print(f"    {n_clipped:,} boundary hexes clipped, {len(hexes) - n_clipped:,} left whole")

    hexes["geometry"] = gpd.GeoSeries(out, crs=4326, index=hexes.index)
    empty = hexes.geometry.is_empty | hexes.geometry.isna()
    if empty.any():
        print(f"    dropped {empty.sum():,} hexes whose clip came out empty")
        hexes = hexes[~empty].copy()

    hexes = hexes.rename(columns={"population": "pop"})[["unit", "pop", "geometry"]]

    # Every region must get hexes, or its dots have nowhere to go and it silently empties.
    missing = sorted(set(units["unit"]) - set(hexes["unit"]), key=int)
    if missing:
        raise SystemExit(
            f"regions with no hex centre: {missing}. At r8 over regions this large that "
            "should be impossible; something is wrong with the join, not with the grid.")
    return hexes


def check(units, hexes, pop):
    import pandas as pd

    ok = True
    k = hexes.groupby("unit")["pop"].sum()
    rows = []
    for _, u in units.iterrows():
        c = pop[u["unit"]]
        kp = float(k.get(u["unit"], 0.0))
        rows.append((u["unit"], u["name"], u["area_km2"], c, kp, kp / c if c else float("nan")))
    df = pd.DataFrame(rows, columns=["unit", "name", "area_km2", "census", "kontur", "ratio"])
    df = df.sort_values("unit", key=lambda s: s.astype(int))

    print(f"\n  {'reg':>3}  {'region':<32} {'area km2':>10} {'census':>9} {'kontur':>9} "
          f"{'ratio':>6}")
    for _, r in df.iterrows():
        flag = "" if RATIO_LO <= r["ratio"] <= RATIO_HI else "   <-- OUT OF BAND"
        print(f"  {r['unit']:>3}  {r['name']:<32} {r['area_km2']:>10,.0f} "
              f"{r['census']:>9,} {r['kontur']:>9,.0f} {r['ratio']:>6.2f}{flag}")

    bad = df[(df["ratio"] < RATIO_LO) | (df["ratio"] > RATIO_HI)]
    ok &= bad.empty
    print(f"\n  {'OK ' if bad.empty else 'BAD'} every region's Kontur/census ratio is inside "
          f"[{RATIO_LO}, {RATIO_HI}] ({len(bad)} outside)")

    tot = df["kontur"].sum() / df["census"].sum()
    print(f"      national ratio {tot:.3f} — Kontur models {df['kontur'].sum():,.0f} against "
          f"a 2012 census {df['census'].sum():,}")

    # The surface is worst where its inputs are thinnest, and that is not at random (§12).
    worst = df.reindex(df["ratio"].sub(tot).abs().sort_values(ascending=False).index).head(3)
    print("      furthest from the national ratio, i.e. where placement inside the region "
          "is least trustworthy:")
    for _, r in worst.iterrows():
        print(f"        Region {r['unit']} {r['name']} — {r['ratio']:.2f}x")

    if not ok:
        raise SystemExit("boundary check FAILED")
    return df


def main():
    if "--fetch" in sys.argv:
        fetch()
    os.makedirs(OUT_DIR, exist_ok=True)

    units = build_units()
    pop = census_pop()
    hexes = build_grid(units)
    df = check(units, hexes, pop)

    units.to_file(UNITS_OUT, driver="GPKG", layer="regions")
    print(f"\nwrote {UNITS_OUT} ({len(units)} regions)")
    hexes.to_file(GRID_OUT, driver="GPKG", layer="grid")
    print(f"wrote {GRID_OUT} ({len(hexes):,} hexes)")
    df.to_csv(LOOKUP, index=False)
    print(f"wrote {LOOKUP}")


if __name__ == "__main__":
    main()
