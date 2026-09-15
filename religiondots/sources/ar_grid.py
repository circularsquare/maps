"""Argentina — the placement layer: Kontur 400 m population hexagons, keyed to the six regions.

Writes data/geo/ar/ar_hexes.gpkg. `countries.py` uses it to weight where a region's dots land,
never to change how many there are.

**SIX UNITS FOR 46 MILLION PEOPLE, 7.6 MILLION EACH, and most of their area is empty.**
Patagonia is a third of the country's land and 5.6% of its people; NOA's 5.9 million are in a
handful of valleys along the Andes' eastern edge; Centro holds the Pampas and also Córdoba,
Rosario and Mar del Plata. An equal-share wash would put most of Argentina's dots on steppe and
puna. §8.2's emptiness case, as acute as Türkiye's (sources/tr_grid.py).

**IT IS A POPULATION WEIGHT AND NOT A RELIGIOUS ONE.** The survey says nothing about where
inside a region an evangelical lives, so a Catholic dot and an evangelical dot in NEA are spread
identically. The map is the same six mixtures however finely the dots are placed.

**THE PER-REGION CHECK HAS REAL POWER ON THE JOIN BUT THE SHUFFLED NULL DOES NOT.** Kontur is
modelled from GHSL, HRSL and building footprints; the counts are INDEC's census, so the two
share no lineage and a broken join would show. Six labels shuffled leave too few ways to be
wrong for the null to mean much, and it is printed with that caveat.

**AND IT ASSERTS THAT NOTHING LANDS ON THE MALVINAS.** Kontur's country extract may carry hexes
there; sources/ar_geo.py dropped the islands from Patagonia, so any such hex must fall outside
every region here.

Usage:
    python sources/ar_grid.py --fetch    one ~30 MB gz from Kontur
    python sources/ar_grid.py            rebuild from data/raw/ar/
"""

import os

# [[feedback_cap_cpu]]: Anita is using the machine. Set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "6")

import gzip
import random
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ar")
GEO = os.path.join(ROOT, "data", "geo", "ar")
REGIONS = os.path.join(GEO, "ar_regions.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "ar.csv")
OUT = os.path.join(GEO, "ar_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_AR_20231101.gpkg.gz")
GZ_NAME = "kontur_population_AR_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_AR_20231101.gpkg"

EXPECTED_UNITS = 6
REGISTER_POPULATION = 45_892_285          # INDEC census 2022, the total sources/ar.py draws

# Kontur is modelled and is not a census; never assert it equal to one (§12, North Macedonia).
NATIONAL_TOLERANCE = 0.35
UNIT_BAND = 2.0
MAX_OUTSIDE_BAND = 0                      # six very large units; none may miss by 2x

MALVINAS = (-62.5, -53.5, -57.0, -50.5)   # lon/lat box


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
        r = requests.get(GZ_URL, timeout=3600, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        print(f"  {os.path.getsize(gz):,} bytes")


def unpack():
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 5_000_000:
        return gpkg
    if not os.path.exists(gz):
        raise SystemExit(f"missing {gz} -- run with --fetch first")
    tmp = gpkg + ".part"
    with gzip.open(gz, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(tmp, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{tmp} is not a GeoPackage")
    os.replace(tmp, gpkg)            # [[reference_wb_truncates]]: temp then replace
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")
    return gpkg


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    gpkg = unpack()
    if not os.path.exists(REGIONS):
        raise SystemExit(f"missing {REGIONS} -- run sources/ar_geo.py first")

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

    # Centroid in the CRS the hexes were TILED in, then reproject the POINTS.
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(reg.crs)

    x0, y0, x1, y1 = MALVINAS
    on_islands = ((pts.geometry.x > x0) & (pts.geometry.x < x1) &
                  (pts.geometry.y > y0) & (pts.geometry.y < y1)).to_numpy()
    print(f"  {int(on_islands.sum())} Kontur hexes ({hexes.loc[on_islands, popcol].sum():,.0f} "
          "people) sit on the Malvinas in Kontur's extract")

    joined = gpd.sjoin(pts, reg[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    hexes["unit"] = joined["unit"].reindex(hexes.index).to_numpy()

    placed_on_islands = int(pd.Series(hexes["unit"].to_numpy()[on_islands]).notna().sum())
    if placed_on_islands:
        raise SystemExit(f"{placed_on_islands} Malvinas hexes joined to a region -- "
                         "ar_geo.py's offshore drop did not hold")

    outside = int(hexes["unit"].isna().sum())
    lost = hexes.loc[hexes["unit"].isna(), popcol].sum()
    print(f"  {outside:,} hexes ({lost:,.0f} people) fall outside every region -- the "
          "coast, the islands, and Kontur's overrun into Chile, Bolivia, Paraguay, Brazil "
          "and Uruguay")
    hexes = hexes[hexes["unit"].notna()].copy()

    tot = hexes[popcol].sum()
    ratio = tot / REGISTER_POPULATION
    print(f"\nkontur/census nationally: {ratio:.3f}x ({tot:,.0f} vs {REGISTER_POPULATION:,})")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"national ratio {ratio:.3f} is outside 1 +/- {NATIONAL_TOLERANCE}")

    counts = (pd.read_csv(NORM, dtype={"geo_id": str})
              .query("geo_level == 'region'").groupby("geo_id")["count"].sum())
    k = hexes.groupby("unit")[popcol].sum()
    cmp = pd.DataFrame({"kontur": k, "census": counts}).dropna()
    if len(cmp) != EXPECTED_UNITS:
        raise SystemExit(f"only {len(cmp)} regions have both sides -- the join dropped one")
    cmp["ratio"] = cmp["kontur"] / cmp["census"] / ratio
    bad = cmp[(cmp["ratio"] < 1 / UNIT_BAND) | (cmp["ratio"] > UNIT_BAND)]
    print(f"per-region ratio (normalised): {len(bad)} of {len(cmp)} outside {UNIT_BAND:.1f}x")
    for u, r in cmp.sort_values("ratio").iterrows():
        mark = "  <--" if u in bad.index else ""
        print(f"    {u:10s} {r['ratio']:.3f}x  kontur={r['kontur']:>12,.0f} "
              f"census={r['census']:>12,.0f}{mark}")
    if len(bad) > MAX_OUTSIDE_BAND:
        raise SystemExit(f"{len(bad)} regions outside the band -- the join is suspect")

    rng = random.Random(7)
    worst = []
    labels = list(cmp.index)
    for _ in range(200):
        s = labels[:]
        rng.shuffle(s)
        null = cmp["kontur"].to_numpy() / cmp["census"].reindex(s).to_numpy() / ratio
        worst.append(float(max(null.max(), 1 / null.min())))
    real = float(max(cmp["ratio"].max(), 1 / cmp["ratio"].min()))
    beaten = sum(1 for w in worst if w <= real) / len(worst)
    print(f"  null (region labels shuffled 200 times): the real join's worst miss is "
          f"{real:.2f}x; {beaten:.0%} of shuffles do as well -- six units, weak evidence")

    # **THE COLUMN MUST BE CALLED `pop`** -- countries.py's `_kontur_place_weight` tests for
    # exactly that name and silently falls back to equal shares per polygon without it.
    out = hexes[["unit", popcol, "geometry"]].rename(columns={popcol: "pop"})
    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, driver="GPKG", layer="hexes")
    print(f"\nwrote {OUT} -- {len(out):,} hexes over {out['unit'].nunique()} regions")


if __name__ == "__main__":
    main()
