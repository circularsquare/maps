"""South Africa — the placement layer: Kontur 400 m population hexagons, keyed to municipality.

Writes data/geo/za/za_hexes.gpkg.

**THE UNITS WENT FROM 9 TO 213 ON 2026-09-09 AND THIS LAYER MATTERS LESS THAN IT DID, WHICH
IS THE POINT.** With nine provinces every dot in Northern Cape was placed by this grid and by
nothing else, on a polygon the size of Germany. With 213 municipalities the units are a
median 123,419 people and a median 2,861 km², so the grid is doing ordinary within-unit work
rather than carrying the whole geography. It is still needed and the reason is unchanged:
South African municipalities are fine in people and wild in area. Dawid Kruiper is 44,231 km²
with 107,161 people; Mandeni is 545 km² with 147,808. An equal share over the outlines would
still draw the Karoo and the Kalahari as populous as the coast.

THE JOIN IS A SPATIAL ONE, on hex CENTROIDS, so a hex on a municipal line belongs wholly to
one side and no population is split or double-counted.

**LESOTHO IS THE ONE TO WATCH, AND ESWATINI BESIDE IT.** Lesotho is entirely surrounded by
South Africa and Kontur's ZA extract does not respect that as reliably as the boundary file
does; any hex whose centroid lands inside the Lesotho hole falls outside all 213 municipalities
and is dropped, which is correct, but it means the dropped-population line is expected to be
non-zero and must not be read as a fault. It is reported rather than silently discarded. The
same goes for the Mozambique and Zimbabwe borders in the north east.

**PRINCE EDWARD AND MARION ISLANDS** are South African territory 1,700 km south east of Cape
Town, inside the City of Cape Town municipality. They have no permanent population, only a
rotating research station, so Kontur has nothing there and nothing is drawn; they are
mentioned because a bounding box computed from this layer will not include them and that is
the right answer.

**AND AT 213 UNITS THE JOIN CAN BE TESTED RATHER THAN ASSUMED.** With nine provinces the only
available check was that the national ratio was not absurd. Here Kontur's population per
municipality is compared against the people this map draws there, two ways, following Benin
and Zimbabwe: the log-log correlation against the best of 500 shuffles of the same numbers
over the same polygons, and the per-unit ratio band. The correlation is the discriminating one
on a country with this many similar-sized units.

Usage:
    python sources/za_grid.py --fetch    one ~39 MB gzipped gpkg from Kontur
    python sources/za_grid.py            rebuild from data/raw/za/
"""

import gzip
import os
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "za")
GEO = os.path.join(ROOT, "data", "geo", "za")
UNITS = os.path.join(GEO, "za_munics.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "za.csv")
OUT = os.path.join(GEO, "za_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_ZA_20231101.gpkg.gz")
GZ_NAME = "kontur_population_ZA_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_ZA_20231101.gpkg"

EXPECTED_UNITS = 213

# The two ReligionBelief rows sources/za.py emits and taxonomy/za2016.py excludes. They are in
# the normalised file so tools/gap_share.py can see them, so they have to come back out here:
# counting them would compare Kontur against 707,296 people this map does not draw.
NOT_DRAWN = {"Religion: Do not know", "Religion: Unspecified"}

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a count and must not
# be asserted equal to one (§12, North Macedonia). It is used ONLY as a within-unit weight, so
# what matters is that it is not wildly out, and the band is deliberately wide.
#
# The comparison is against the people actually drawn from data/normalized/za.csv, which is
# the CS 2016 answer universe. Kontur's grid is dated November 2023 against a 2016 survey on a
# country growing about 1.3%/yr, and the CSV also drops the `Do not know` and `Unspecified`
# answers, so the ratio SHOULD read high and a value near 1.10 is the expected result rather
# than a problem.
KONTUR_TOLERANCE = 0.30
# Measured 2026-09-09: r=0.9829 as built against a best of 0.8026 over 500 shuffles. The floor
# is set well below the built value and well above the shuffled one, so it fails on a broken
# join rather than on Kontur being revised.
MIN_LOGLOG_R = 0.90
SHUFFLES = 500


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
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(UNITS):
        raise SystemExit(f"missing {UNITS} -- run sources/za_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    units = gpd.read_file(UNITS)
    if len(units) != EXPECTED_UNITS:
        raise SystemExit(f"{UNITS} has {len(units)} units, expected {EXPECTED_UNITS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(units.crs)
    hexes = hexes.to_crs(units.crs)

    joined = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every municipality: "
          f"{int(outside.sum()):,} ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Kontur's ZA extract overruns the border and, more to the point, covers the\n"
          "     Lesotho enclave and the Eswatini salient, neither of which is a South\n"
          "     African municipality; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    names = {u: n for u, n in zip(units["unit"], units["name"])}
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit("municipalities with no populated hex: "
                         + ", ".join(f"{names.get(u, u)} ({u})" for u in missing))
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"municipalities whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} municipalities has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/za.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[(df["geo_level"] == "municipality") & (~df["source_category"].isin(NOT_DRAWN))]
    drawn_per = df.groupby("geo_id")["count"].sum()
    drawn = float(drawn_per.sum())
    tot = float(out["pop"].sum())
    ratio = tot / drawn
    print(f"\n  Kontur {tot:,.0f} vs the drawn CS 2016 answers {drawn:,.0f} - "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and CS 2016 disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much even for a weight -- check the download")
    print("     a November 2023 model against a 2016 survey whose non-answers are not "
          "drawn, so\n     it should and does read high; used only as a WITHIN-unit weight, "
          "so the level\n     does not matter and the shape does.")

    # ---- the shape, tested rather than assumed (Benin's check, which 213 units can carry) ----
    common = sorted(set(per.index) & set(drawn_per.index))
    if len(common) != EXPECTED_UNITS:
        raise SystemExit(f"{len(common)} municipalities in both Kontur and za.csv")
    k = np.log(np.array([per.loc[u, "sum"] for u in common], dtype=float))
    d = np.log(np.array([drawn_per[u] for u in common], dtype=float))
    r = float(np.corrcoef(k, d)[0, 1])
    rng = np.random.default_rng(20260909)
    best = max(float(np.corrcoef(k, rng.permutation(d))[0, 1]) for _ in range(SHUFFLES))
    print(f"\n  SHAPE CHECK: log-log correlation between Kontur's population per "
          f"municipality and\n    the people drawn there is r={r:.4f}, against a best of "
          f"{best:.4f} over {SHUFFLES} shuffles of the\n    same numbers over the same "
          "polygons. The gap is the evidence that the join is right.")
    if r < MIN_LOGLOG_R:
        raise SystemExit(f"log-log r={r:.4f} is under the floor {MIN_LOGLOG_R}; the "
                         "municipality join is wrong, or Kontur has been revised")

    band = np.array([per.loc[u, "sum"] / drawn_per[u] for u in common])
    order = np.argsort(band)
    print(f"\n  per-municipality ratio band: {band.min():.2f} to {band.max():.2f}, "
          f"median {np.median(band):.2f}; "
          f"{int((band < 0.5).sum() + (band > 2.0).sum())} of {len(band)} outside 0.5-2.0")
    for i in list(order[:4]) + list(order[-4:]):
        u = common[i]
        print(f"    {names.get(u, u):28s} {u:7s} Kontur {per.loc[u, 'sum']:>10,.0f}  "
              f"drawn {drawn_per[u]:>10,}  {band[i]:5.2f}x")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
