"""Thailand — the placement layer: Kontur 400 m population hexagons, keyed to province.

Writes data/geo/th/th_hexes.gpkg.

**76 provinces for 66.0M people is ~868,000 each, and the provinces are wildly uneven**, so
this is §8.2's ordinary case rather than a special one: Bangkok is 8.3M in 1,571 km² while Mae
Hong Son is 209,200 in 12,681 km² of forested mountain. An equal share per polygon would spray
the northwestern highlands with evenly spaced dots and squeeze an eighth of the country into
one speck.

**IT MATTERS MOST FOR THE MUSLIM SOUTH, AND FOR THE OPPOSITE REASON TO CAMBODIA'S.** There the
grid was needed because the distinctive provinces were nearly empty; here Pattani, Yala and
Narathiwat are dense, wet-rice and rubber country with populations concentrated along the
coast and the Pattani river, and their interiors are the Sankalakhiri range. A wash would
paint the mountains the same as the towns and put the map's sharpest religious boundary in the
wrong place inside each province.

**THE GULF AND THE ANDAMAN ARE INSIDE THE PROVINCES.** Every peninsular changwat's polygon
runs out to a maritime limit, and so do the Gulf provinces around Bangkok. A population grid
has no hexes on open water, so §8.2c's problem does not arise rather than being patched —
Cambodia's Tonle Sap finding, on a coastline.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two provinces.

Usage:
    python sources/th_grid.py --fetch    one ~30 MB gzipped gpkg from Kontur
    python sources/th_grid.py            rebuild from data/raw/th/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "th")
GEO = os.path.join(ROOT, "data", "geo", "th")
PROVINCES = os.path.join(GEO, "th_provinces.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "th.csv")
OUT = os.path.join(GEO, "th_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TH_20231101.gpkg.gz")
GZ_NAME = "kontur_population_TH_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_TH_20231101.gpkg"

EXPECTED_PROVINCES = 76
CENSUS_POPULATION = 65_981_658

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Only a within-province weight is used,
# so the level does not matter and the shape does. Kontur's vintage is 2023 against a 2010
# census, thirteen years at ~0.4%/yr, so this should read somewhat above 1.0 — and Thailand's
# census counts *de facto* residents while Kontur models built-up population, which in a
# country with several million registered-elsewhere migrant workers pushes the same way.
NATIONAL_TOLERANCE = 0.45

# 76 uneven units give the band a real null. Two outliers tolerated, on Cambodia's reasoning.
UNIT_BAND = 2.0
MAX_OUTSIDE_BAND = 3


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print("  %s bytes" % "{:,}".format(os.path.getsize(gz)))
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    # §5a: a 200 is not a download, and a gunzip that runs is not a GeoPackage.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit("%s is not a GeoPackage -- starts %r" % (gpkg, magic))
    print("  unpacked %s bytes" % "{:,}".format(os.path.getsize(gpkg)))


def main():
    import random

    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit("no %s -- run with --fetch" % gpkg)

    prov = gpd.read_file(PROVINCES)
    print("provinces: %d" % len(prov))
    if len(prov) != EXPECTED_PROVINCES:
        print("  !! expected %d" % EXPECTED_PROVINCES)

    hexes = gpd.read_file(gpkg)
    print("kontur: %s hexes, crs=%s" % ("{:,}".format(len(hexes)), hexes.crs))
    hexes = hexes.to_crs(prov.crs)

    pts = hexes.copy()
    pts["geometry"] = pts.geometry.centroid
    joined = gpd.sjoin(pts, prov[["unit", "geometry"]], how="left", predicate="within")
    hexes["unit"] = joined["unit"].to_numpy()

    outside = hexes["unit"].isna().sum()
    lost = hexes.loc[hexes["unit"].isna(), "population"].sum()
    print("  %s hexes (%s people) fall outside every province -- the sea edge and the "
          "Malaysian, Lao, Cambodian and Myanmar overrun"
          % ("{:,}".format(int(outside)), "{:,.0f}".format(lost)))
    hexes = hexes[hexes["unit"].notna()].copy()

    tot = hexes["population"].sum()
    ratio = tot / CENSUS_POPULATION
    print("\nkontur/census nationally: %.3fx (%s vs %s)"
          % (ratio, "{:,.0f}".format(tot), "{:,}".format(CENSUS_POPULATION)))
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit("national ratio %.3f is outside 1 +/- %.2f -- the join moved"
                         % (ratio, NATIONAL_TOLERANCE))

    # Per-province ratios: the check that the join is right, not that Kontur is.
    census = (pd.read_csv(NORM, dtype={"geo_id": str})
              .query("geo_level == 'province'")
              .groupby("geo_id")["count"].sum())
    k = hexes.groupby("unit")["population"].sum()
    cmp = pd.DataFrame({"kontur": k, "census": census}).dropna()
    cmp["ratio"] = cmp["kontur"] / cmp["census"] / ratio
    bad = cmp[(cmp["ratio"] < 1 / UNIT_BAND) | (cmp["ratio"] > UNIT_BAND)]
    print("per-province ratio (normalised): median %.2f, %d of %d outside %.1fx"
          % (cmp["ratio"].median(), len(bad), len(cmp), UNIT_BAND))
    for u, r in bad.sort_values("ratio").iterrows():
        print("    %-5s %.2fx  kontur=%s census=%s"
              % (u, r["ratio"], "{:,.0f}".format(r["kontur"]),
                 "{:,.0f}".format(r["census"])))
    if len(bad) > MAX_OUTSIDE_BAND:
        raise SystemExit("%d provinces outside the band -- the join is suspect" % len(bad))

    # The null: a shuffled join should fail the same band badly (§12, Benin).
    rng = random.Random(7)
    shuffled = list(cmp.index)
    rng.shuffle(shuffled)
    null = cmp["kontur"].to_numpy() / cmp["census"].reindex(shuffled).to_numpy() / ratio
    n_bad = int(((null < 1 / UNIT_BAND) | (null > UNIT_BAND)).sum())
    print("  null (province labels shuffled): %d of %d outside the band" % (n_bad, len(cmp)))

    # **THE COLUMN MUST BE CALLED `pop`.** countries.py's `_kontur_place_weight` tests for
    # exactly that name and, not finding it, prints one line and falls back to equal shares
    # per polygon — the scatter still runs, the map still draws, and the placement is
    # silently wrong. Writing Kontur's own `population` cost one build before it was caught.
    out = hexes[["unit", "population", "geometry"]].rename(columns={"population": "pop"})
    out.to_file(OUT, driver="GPKG", layer="hexes")
    print("\nwrote %s -- %s hexes over %d provinces"
          % (OUT, "{:,}".format(len(out)), out["unit"].nunique()))


if __name__ == "__main__":
    main()
