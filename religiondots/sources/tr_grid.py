"""Türkiye — the placement layer: Kontur 400 m population hexagons, keyed to İBBS-1.

Writes data/geo/tr/tr_hexes.gpkg. `countries.py` uses it to weight where a region's dots
land, never to change how many there are.

**TÜRKİYE NEEDS THIS MORE THAN ALMOST ANY COUNTRY HERE, because twelve units for 85 million
people is 7.1 million each and the units are enormous in area as well as in population.**
TR7 Orta Anadolu is 90,000 km² of which most is the Konya-Aksaray steppe; TRB Ortadoğu
Anadolu runs from Malatya to the Iranian border across the Eastern Anatolian plateau, and its
people are in a handful of basins — Malatya, Elazığ, Van, Muş — with mountains between them.
An equal-share wash over those polygons would put a third of Turkey's dots on empty high
ground, and at this unit size that is most of what the reader would see. §8.2's emptiness
case, at the largest scale it has come up.

**IT IS A POPULATION WEIGHT AND NOT A RELIGIOUS ONE.** The Diyanet measures nothing about
where inside a region a Shafi'i lives, so a Hanafi dot and a Shafi'i dot in TRC are spread
identically. Read a cluster as "this region, drawn where Turks actually live", never as a
neighbourhood reading. The map is the same twelve mixtures however finely the dots are
placed.

**THE PER-REGION CHECK HAS REAL POWER HERE, UNLIKE FIJI'S FIFTEEN UNITS**, because the two
sides share no lineage: Kontur is modelled from GHSL, HRSL and building footprints, and the
counts come from OCHA COD-PS, which is TÜİK's address register. Two independent estimates of
how many people are in each of twelve regions have to agree, and if the spatial join were
wrong they would not. Twelve is still few enough that the shuffled null is weak evidence
rather than strong, and it is printed with that caveat rather than dressed up.

**THE VINTAGE GAP IS TEN YEARS AND IT MOVES NOTHING.** Counts are 2013 shares on 2022
population; the grid is Kontur's 2023 snapshot. The grid only ever decides where inside a
region a dot sits, so no count is affected by it at all.

Usage:
    python sources/tr_grid.py --fetch    one ~33 MB gz from Kontur
    python sources/tr_grid.py            rebuild from data/raw/tr/
"""

import gzip
import os
import random
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tr")
GEO = os.path.join(ROOT, "data", "geo", "tr")
REGIONS = os.path.join(GEO, "tr_ibbs1.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "tr.csv")
OUT = os.path.join(GEO, "tr_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_TR_20231101.gpkg.gz")
GZ_NAME = "kontur_population_TR_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_TR_20231101.gpkg"

EXPECTED_UNITS = 12
REGISTER_POPULATION = 85_279_553          # COD-PS 2022, the same total sources/tr.py draws

# Kontur is modelled and is not a census; it must never be asserted equal to one (§12, North
# Macedonia). This band is about whether the JOIN worked, not whether Kontur is right.
NATIONAL_TOLERANCE = 0.35

# Per region, after normalising by the national ratio. Twelve large units, none tiny, so a
# generous band that a broken join would still fail.
UNIT_BAND = 2.0
MAX_OUTSIDE_BAND = 1


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
        r = requests.get(GZ_URL, timeout=3600, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(REGIONS):
        raise SystemExit(f"missing {REGIONS} -- run sources/tr_geo.py first")

    reg = gpd.read_file(REGIONS)
    if len(reg) != EXPECTED_UNITS:
        raise SystemExit(f"{REGIONS} has {len(reg)} regions, expected {EXPECTED_UNITS}")
    print(f"regions: {len(reg)}, crs={reg.crs}")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"kontur: {len(hexes):,} hexes, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    # Centroid in the CRS the hexes were TILED in, then reproject the POINTS. Reprojecting
    # the polygons first and taking centroids after moves them.
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid, crs=hexes.crs)
    pts = pts.to_crs(reg.crs)

    joined = gpd.sjoin(pts, reg[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]      # a border point, once only
    hexes["unit"] = joined["unit"].reindex(hexes.index).to_numpy()

    outside = int(hexes["unit"].isna().sum())
    lost = hexes.loc[hexes["unit"].isna(), popcol].sum()
    print(f"  {outside:,} hexes ({lost:,.0f} people) fall outside every region -- the sea "
          "edge and the Syrian, Iraqi, Iranian, Armenian, Georgian, Greek and Bulgarian "
          "overrun in Kontur's country extract")
    hexes = hexes[hexes["unit"].notna()].copy()

    tot = hexes[popcol].sum()
    ratio = tot / REGISTER_POPULATION
    print(f"\nkontur/register nationally: {ratio:.3f}x "
          f"({tot:,.0f} vs {REGISTER_POPULATION:,})")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"national ratio {ratio:.3f} is outside 1 +/- {NATIONAL_TOLERANCE} "
                         "-- the join moved")

    counts = (pd.read_csv(NORM, dtype={"geo_id": str})
              .query("geo_level == 'region'")
              .groupby("geo_id")["count"].sum())
    k = hexes.groupby("unit")[popcol].sum()
    cmp = pd.DataFrame({"kontur": k, "register": counts}).dropna()
    if len(cmp) != EXPECTED_UNITS:
        raise SystemExit(f"only {len(cmp)} regions have both sides -- the join dropped one")
    cmp["ratio"] = cmp["kontur"] / cmp["register"] / ratio
    bad = cmp[(cmp["ratio"] < 1 / UNIT_BAND) | (cmp["ratio"] > UNIT_BAND)]
    print(f"per-region ratio (normalised): median {cmp['ratio'].median():.2f}, "
          f"{len(bad)} of {len(cmp)} outside {UNIT_BAND:.1f}x")
    for u, r in cmp.sort_values("ratio").iterrows():
        mark = "  <--" if u in bad.index else ""
        print(f"    {u:5s} {r['ratio']:.2f}x  kontur={r['kontur']:>12,.0f} "
              f"register={r['register']:>12,.0f}{mark}")
    if len(bad) > MAX_OUTSIDE_BAND:
        raise SystemExit(f"{len(bad)} regions outside the band -- the join is suspect")

    # The null, and it is WEAK EVIDENCE at twelve units: shuffling twelve labels can easily
    # leave several in place by chance. Printed because a null that does NOT fire would be a
    # warning, not because passing it proves much (fj_grid.py's point about fifteen).
    rng = random.Random(7)
    shuffled = list(cmp.index)
    rng.shuffle(shuffled)
    null = cmp["kontur"].to_numpy() / cmp["register"].reindex(shuffled).to_numpy() / ratio
    n_bad = int(((null < 1 / UNIT_BAND) | (null > UNIT_BAND)).sum())
    print(f"  null (region labels shuffled): {n_bad} of {len(cmp)} outside the band "
          "-- weak at twelve units, see the docstring")

    # **THE COLUMN MUST BE CALLED `pop`** -- countries.py's `_kontur_place_weight` tests for
    # exactly that name and silently falls back to equal shares per polygon without it,
    # which for a twelve-unit country would be very wrong and would still draw a map.
    out = hexes[["unit", popcol, "geometry"]].rename(columns={popcol: "pop"})
    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, driver="GPKG", layer="hexes")
    print(f"\nwrote {OUT} -- {len(out):,} hexes over {out['unit'].nunique()} regions")


if __name__ == "__main__":
    main()
