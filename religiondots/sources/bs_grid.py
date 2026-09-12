"""The Bahamas — the placement layer: Kontur 400 m population hexagons, keyed to island.

Writes data/geo/bs/bs_hexes.gpkg.

**THE PROBLEM HERE IS THE OPPOSITE OF TRINIDAD'S: ONE UNIT HOLDS THREE QUARTERS OF THE
COUNTRY.** New Providence is 296,732 of 398,165 people — **74.5%** — on 282 km², while
Andros is 6,859 km² of mangrove and blue hole holding 7,695 people. So the counting tier is
lopsided in both directions at once, and neither end is served by an equal share over the
polygon:

  * On **New Providence**, three quarters of every religion's dots land in one polygon, and
    where they land inside it is the only spatial information the map has left for most of
    the country. Nassau and its suburbs occupy the north-east; the pine barren of the
    south-west is close to empty. Uniform scatter would put a quarter of Nassau's Baptists
    in the pine forest.
  * On the **Family Islands**, settlement is a thread of villages along one road on a long
    thin cay. Andros at 7,695 people over 6,859 km² is 1.1 people per km² averaged, and
    essentially all of them are on the eastern shore.

Kontur handles both without a special case, which is why it is used rather than the
census's own enumeration districts — those exist (BNSI publishes population by settlement
for 2010) but not as boundaries anyone publishes.

**THE ISLANDS ARE ISLANDS, SO THE SPATIAL JOIN CANNOT SPILL.** Every unit here is separated
from every other by open sea; a hex whose centroid misses every island polygon is a
coastline disagreement between Kontur and COD-AB, not a border overrun. `water.py` is not
involved.

**AND THAT DISAGREEMENT IS 6% OF THE COUNTRY, SO IT IS SNAPPED RATHER THAN DROPPED.** A
plain `within` join leaves 836 hexes and **25,095 modelled people** — 6.08% — outside every
island. COD-AB's Bahamas geometry is GDAMS 2009 and its coastline is generalised, while a
Kontur hex is 400 m across, so a shoreline settlement's centroid routinely lands just off
the polygon. Measured, before deciding:

    <100 m from an island   303 hexes   12,606 people
    100-250 m               367 hexes   10,595 people
    250-500 m               163 hexes    1,891 people
    500 m and beyond          3 hexes        3 people

**Three people, in three hexes, are genuinely offshore.** Everything else is the coast
itself — and in the Bahamas the coast is where the population is, so dropping it would tilt
every island's dots inland, hardest on exactly the settlements that matter: nine of the
twelve heaviest unclaimed hexes are Nassau's own waterfront. So the join is `within` first,
which is authoritative and cannot spill, and then `sjoin_nearest` **capped at 1 km** for
what is left; past that, dropped. The cap is what keeps the operation honest — it is a snap
to the coastline, not a nearest-neighbour fill, and the distance histogram above is printed
on every run so a future vintage with a real offshore population would show up rather than
be absorbed.

Distances are measured in **UTM 18N**. The archipelago spans zones 17 and 18, so the
westernmost islands sit a few degrees off the central meridian; at a 1 km threshold that
scale error is a few metres and cannot change any assignment.

**TWO UNITS ARE SMALLER THAN THIS GRID IS REALLY MEANT FOR.** Spanish Wells is 4 km² and
Harbour Island 6 km², so at 0.67 km² per hex they hold single-figure counts. That is thin
and it is also where it matters least — an island small enough to walk across is close to
uniform anyway. It is asserted rather than assumed: every island must come out with at
least one populated hex, or its dots would silently fall back to §8.2's equal share.

Usage:
    python sources/bs_grid.py --fetch    one ~130 KB gzipped gpkg from Kontur
    python sources/bs_grid.py            rebuild from data/raw/bs/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bs")
GEO = os.path.join(ROOT, "data", "geo", "bs")
ISLANDS = os.path.join(GEO, "bs_islands.gpkg")
OUT = os.path.join(GEO, "bs_hexes.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "bs.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_BS_20231101.gpkg.gz")
GZ_NAME = "kontur_population_BS_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_BS_20231101.gpkg"

EXPECTED_ISLANDS = 18

# The coastline snap. A hex centroid this far or less from an island is that island's;
# beyond it, the hex is offshore and is dropped. 1 km is chosen against the measured
# distribution — the slop tops out at 500 m and the next hex is 1.3 km away — so the
# threshold sits in an empty gap rather than through the middle of the data.
SNAP_M = 1000
UTM = 32618                     # UTM 18N; see the module docstring on the zone-17 islands

# Kontur is modelled from GHSL, HRSL and building footprints; it is not the census and must
# not be asserted equal to it (§12, North Macedonia). Used ONLY as a within-island weight,
# so what matters is that it is not wildly out — assert the RELATIONSHIP, and MEASURE the
# band rather than copying one from another country (§9u).
CENSUS_POPULATION = 398_165
KONTUR_TOLERANCE = 0.30

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 100_000:
        print("already have", gpkg)
        return
    if not os.path.exists(gz):
        print("GET", GZ_URL)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(gz + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(gz + ".part", gz)
        print(f"  {os.path.getsize(gz):,} bytes")
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)
    # §5a: a 200 is not a download, and a gunzip that runs is not a gpkg.
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage -- starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(ISLANDS):
        raise SystemExit(f"missing {ISLANDS} -- run sources/bs_geo.py first")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    islands = gpd.read_file(ISLANDS)
    if len(islands) != EXPECTED_ISLANDS:
        raise SystemExit(f"{ISLANDS} has {len(islands)} islands, "
                         f"expected {EXPECTED_ISLANDS}")

    # Kontur ships in EPSG:3857. Take the centroid in the CRS the hexes were tiled in, then
    # reproject the POINTS -- reprojecting first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(islands.crs)
    hexes = hexes.to_crs(islands.crs)

    joined = gpd.sjoin(pts, islands[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)
    unit = joined["unit"].copy()        # NaN where the hex is outside every island

    outside = joined["unit"].isna().to_numpy()
    adrift = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every island polygon: "
          f"{int(outside.sum()):,} ({adrift:,.0f} people, "
          f"{100.0 * adrift / pts[popcol].sum():.3f}%)")

    # ---- snap the coastal slop, and measure it rather than assuming it ----
    m_pts = pts[outside].to_crs(UTM)
    near = gpd.sjoin_nearest(m_pts, islands.to_crs(UTM)[["unit", "geometry"]],
                             how="left", distance_col="dist_m")
    near = near[~near.index.duplicated(keep="first")].reindex(m_pts.index)

    print("     distance from one of those to the nearest island:")
    edges = [0, 100, 250, 500, 1000, SNAP_M, float("inf")]
    names = ["<100 m", "100-250 m", "250-500 m", "500 m-1 km",
             f"1 km-{SNAP_M / 1000:g} km", f">{SNAP_M / 1000:g} km"]
    for lo, hi, lab in zip(edges[:-1], edges[1:], names):
        sel = near[(near["dist_m"] >= lo) & (near["dist_m"] < hi)]
        if len(sel):
            print(f"       {lab:<12} {len(sel):>5,} hexes  {sel[popcol].sum():>9,.0f} people")

    snap = near["dist_m"] <= SNAP_M
    idx = near.index[snap]
    unit.loc[idx] = near.loc[idx, "unit"]
    snapped = float(near.loc[snap, popcol].sum())
    print(f"     snapped to the nearest island within {SNAP_M:,} m: "
          f"{int(snap.sum()):,} hexes ({snapped:,.0f} people). COD-AB's coastline is "
          "generalised\n     GDAMS 2009 and a hex is 400 m across, so a shoreline "
          "settlement lands just off it;\n     the Bahamian population IS the coastline, "
          "and dropping this tilts every island inland.")

    # `unit` is a pandas Series and its unmatched entries are NaN, NOT None — so the
    # surviving-hex mask has to be `notna()`. Written as `unit != None` this silently kept
    # the two genuinely-offshore hexes with a null unit, and the only symptom downstream was
    # scatter.py reporting one extra unit "with polygons but no religion rows".
    keep = unit.notna().to_numpy()
    lost = float(pts.loc[~keep, popcol].sum())
    print(f"     still outside after the snap: {int((~keep).sum()):,} hexes "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%) — "
          "genuinely offshore; dropped.")

    out = gpd.GeoDataFrame(
        {"unit": unit[keep].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep].to_numpy(), crs=islands.crs)

    # The layer must carry exactly the units bs_geo.py built and nothing else — no null,
    # no stray. This is the assertion the bug above walked straight past.
    got = set(out["unit"].unique())
    want = set(islands["unit"])
    if got != want or out["unit"].isna().any():
        raise SystemExit(f"hex layer carries units {sorted(got, key=str)}, expected "
                         f"{sorted(want)}")

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(islands["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"islands with no populated hex: {missing} -- their dots would "
                         "fall back to an equal share over the whole polygon")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"islands whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_ISLANDS} islands has hexes: "
          f"{per['size'].min():,}-{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > KONTUR_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")
    print("     a 2023 modelled grid against a 2022 census, one year apart; used only as\n"
          "     a WITHIN-island weight.")

    # The per-unit ratio is the one that would bias a share, and it is printed rather than
    # asserted: an island Kontur models badly gets its dots on a worse surface, it does not
    # get the wrong number of them (§9t). It is ALSO the check on sources/bs_geo.py's
    # district -> island grouping — a district assigned to the wrong island would move its
    # people between two of these rows and show up here as a pair of bad ratios.
    cen = pd.read_csv(NORM, dtype={"geo_id": str}, keep_default_na=False, na_values=[])
    cen = cen[(cen["geo_level"] == "island") & (cen["source_category"] == "TOTAL")]
    cen = dict(zip(cen["geo_id"], pd.to_numeric(cen["count"])))
    print("\n  per-island Kontur/census ratio (also the check on the district grouping):")
    rows = []
    for unit, row in per.iterrows():
        nm = islands.loc[islands["unit"] == unit, "name"].iloc[0]
        rows.append((row["sum"] / cen[unit], nm, int(row["size"]), cen[unit]))
    for r, nm, n, c in sorted(rows):
        print(f"      {nm:<26} {r:5.2f}x   {n:>6,} hexes   census {c:>8,}")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
