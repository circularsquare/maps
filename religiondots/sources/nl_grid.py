"""The Netherlands — the placement layer: Kontur 400 m population hexagons, keyed to the
gemeente the counts are on.

Writes data/geo/nl/nl_hexes.gpkg.

WHY A GRID AND NOT AN EQUAL SHARE PER POLYGON. Dutch gemeenten are not the problem Austria's
Alpine ones were, but they have their own version of it and it is water. A gemeente's polygon
here runs out into the IJsselmeer, the Markermeer, the Zeeland delta and the Waddenzee: Urk's
takes in a slab of open lake, Noordoostpolder and Lelystad reach far out into the Markermeer,
and half of Zeeland is estuary. Spreading dots evenly over those polygons would put a
noticeable part of the country's most Reformed population on water. A population grid has no
hexes on open water, so spec §8.2c's problem does not arise rather than being patched.

It also does the thing the grid is normally for. Noordoostpolder, Hollands Kroon and
Súdwest-Fryslân are 400 to 500 km² of mostly empty polder with the people in a handful of
villages, and Amsterdam is 165 km² with 800,000 people in it; at 103 km² average against a
0.14 km² hex there are of the order of seven hundred hexes to a gemeente, well clear of
[[reference_kontur_resolution_floor]]'s point where the grid stops paying.

THE GRID IS 2023 AND THE COUNTS ARE 2010/2014, and that is fine for one reason only: Kontur
is a WITHIN-unit weight. How many dots a gemeente gets is CBS's answer and nothing here can
change it; the grid only decides where inside the gemeente they land. The Netherlands added
about 1.1 million people between 2014 and 2023 and that growth is not evenly spread, so do
not read the national ratio below as an error term.

THE JOIN IS SPATIAL, on hex CENTROIDS, so no hex is split between two gemeenten.

Usage:
    python sources/nl_grid.py --fetch    one gzipped gpkg from Kontur
    python sources/nl_grid.py            rebuild from data/raw/nl/
"""

import gzip
import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "nl")
GEO = os.path.join(ROOT, "data", "geo", "nl")
UNITS = os.path.join(GEO, "nl_units.gpkg")
OUT = os.path.join(GEO, "nl_hexes.gpkg")
CSV = os.path.join(ROOT, "data", "normalized", "nl.csv")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_NL_20231101.gpkg.gz")
GZ_NAME = "kontur_population_NL_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_NL_20231101.gpkg"

EXPECTED_UNITS = 403
POP_2014 = 16_829_289
RD = 28992          # Amersfoort / RD New, the Dutch national grid — metres
SNAP_M = 300        # how far a hex may be from a gemeente and still be that gemeente's


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
    with open(gpkg, "rb") as fh:
        magic = fh.read(16)
    if magic[:4] != b"SQLi":
        raise SystemExit(f"{gpkg} is not a GeoPackage — starts {magic!r}")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    for p in (gpkg, UNITS, CSV):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run nl_grid.py --fetch, nl_geo.py and nl.py")

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
    # reproject the POINTS — reprojecting first and taking the centroid after moves it. The
    # join itself is done in EPSG:28992, the Dutch national grid, because the snap below
    # needs metres and a nearest-join in degrees is silently wrong.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs).to_crs(RD)
    hexes = hexes.to_crs(units.crs)
    u_rd = units[["unit", "name", "geometry"]].to_crs(RD)

    joined = gpd.sjoin(pts, u_rd[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    # TWO DIFFERENT THINGS FALL OUTSIDE AND ONLY ONE OF THEM IS DUTCH, so they are separated
    # rather than both dropped. [[reference_archipelago_grid_snap]]: a stray coastal hex is a
    # boundary-generalisation artefact and must be SNAPPED, because the loss is seaward and
    # dropping it drifts a coastal town's dots inland. But Kontur's NL extract is not clipped
    # to the Netherlands: it carries a slab of Belgium off Zeeuws-Vlaanderen — Brugge, Knokke
    # and Zeebrugge, 371,100 people whose nearest gemeente is Sluis and which are 5 to 25 km
    # from it — and those are foreign and must be dropped. A distance cap separates the two
    # cleanly, because the generalisation error is a few hundred metres and Belgium is not.
    outside = joined["unit"].isna()
    far = gpd.sjoin_nearest(pts[outside.to_numpy()], u_rd, how="left",
                            distance_col="dist")
    far = far[~far.index.duplicated(keep="first")]
    snap = far["dist"] <= SNAP_M
    print(f"\n  hexes whose centroid is outside every gemeente: {int(outside.sum()):,} "
          f"({float(far[popcol].sum()):,.0f} people, "
          f"{100.0 * float(far[popcol].sum()) / pts[popcol].sum():.3f}%)")
    print(f"     within {SNAP_M} m of one, snapped to it: {int(snap.sum()):,} hexes, "
          f"{float(far.loc[snap, popcol].sum()):,.0f} people")
    dropped = far.loc[~snap].groupby("name")[popcol].agg(["size", "sum"])
    dropped = dropped.sort_values("sum", ascending=False)
    print(f"     further than that, dropped as not Dutch: {int((~snap).sum()):,} hexes, "
          f"{float(far.loc[~snap, popcol].sum()):,.0f} people — nearest to "
          + ", ".join(f"{k} {v['sum']:,.0f}" for k, v in dropped.head(4).iterrows()))

    unit_of = joined["unit"].copy()
    unit_of.loc[far.index[snap]] = far.loc[snap, "unit"]
    keep = unit_of.notna()
    out = gpd.GeoDataFrame(
        {"unit": unit_of[keep].to_numpy(),
         "pop": pts.loc[keep.to_numpy(), popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=units.crs)

    # A unit absent from the placement layer has no geometry to draw into and its people are
    # DROPPED, not spread. No Dutch gemeente is anywhere near the 400 m floor — the smallest,
    # Rozendaal, is 6.6 km² — so this is an assertion rather than a repair.
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"{len(missing)} gemeenten with no populated hex: {missing[:12]}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"{len(zero)} gemeenten whose hexes sum to zero: {zero[:12]}")
    print(f"  every one of the {EXPECTED_UNITS} gemeenten has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each, median "
          f"{int(per['size'].median()):,}")

    tot = float(out["pop"].sum())
    print(f"\n  Kontur 2023 {tot:,.0f} vs CBS 1 January 2014 {POP_2014:,} — "
          f"ratio {tot / POP_2014:.3f}")

    # ---- the two nulls, per unit: does the grid agree with CBS about where people are?
    df = pd.read_csv(CSV)
    cbs = df.groupby("geo_id")["count"].sum()
    rows = [(u, float(cbs[u]), float(per.loc[u, "sum"]))
            for u in units["unit"] if u in cbs.index and per.loc[u, "sum"] > 0]
    ratios = sorted((k / c, u) for u, c, k in rows)
    names = units.set_index("unit")["name"]
    print(f"\n  ratio band, Kontur 2023 vs CBS 2014, over {len(rows)} drawn gemeenten:")
    print("    lowest : " + ", ".join(f"{names[u]} {r:.2f}" for r, u in ratios[:4]))
    print("    highest: " + ", ".join(f"{names[u]} {r:.2f}" for r, u in ratios[-4:]))
    print("    median : %.3f" % ratios[len(ratios) // 2][0])
    print("     A gemeente far from 1.0 is a place that grew or shrank in nine years, or a "
          "place whose\n     hexes the border cuts. It changes nothing: the grid is a "
          "within-unit weight only.")

    out.to_file(OUT, driver="GPKG")
    print(f"\nwrote {OUT}  ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
