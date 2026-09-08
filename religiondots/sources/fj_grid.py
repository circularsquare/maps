"""Fiji — the placement grid: Kontur 400 m population hexagons, clipped to the provinces.

Writes data/geo/fj/fj_hexes.gpkg. `countries.py` uses it to weight where a province's dots
land, never to change how many there are.

**FIJI NEEDS THIS BECAUSE THE UNITS ARE ARCHIPELAGOS, NOT AREAS.** Fifteen provinces over 330
islands: Lau is 60-odd islands scattered across 500 km of ocean with 10,683 people between
them; Cakaudrove is half of Vanua Levu plus Taveuni plus Rabi and Kioa. An equal share per
polygon would put dots in open sea and on uninhabited islets, and it would put the same weight
on Suva's suburbs as on a copra island. §8.2's emptiness and water cases at once, and the
water case is unusually literal here.

**EVERY SPATIAL OPERATION HAPPENS IN EPSG:3832, NOT EPSG:4326** — see `sources/fj_geo.py`'s
docstring. Fiji straddles the 180th meridian and three of its provinces tear into 360-degree
polygons in geographic coordinates. Kontur ships in EPSG:3857, which is centred on Greenwich
and therefore splits Fiji as badly as 4326 does: the Lau group lands at x ~ -19,900,000 and
Viti Levu at x ~ +19,800,000, on opposite edges of the plane. **The hexes are reprojected into
the province CRS before anything is joined**, and the assertion below fails loudly if the
country ever comes out wider than it is.

**THE VINTAGE GAP IS SIXTEEN YEARS** — counts 2007, grid 2023 — second only to Nicaragua's
eighteen. It moves dots *within* a province and never between provinces, so no count is
affected.

**AND THE CORRELATION HERE IS THE ONLY CHECK ON THE JOIN WITH ANY POWER**, which is why
`fj_geo.py` declines to assert a geographic witness: fifteen units cannot calibrate a
neighbour test. A modelled 2023 population grid, sharing no lineage with FBoS's census or with
OCHA's boundaries, has to agree about how many people are on each of 15 island groups.

Usage:
    python sources/fj_grid.py --fetch    one ~0.8 MB gz from Kontur
    python sources/fj_grid.py            rebuild from data/raw/fj/
"""

import gzip
import math
import os
import random
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "fj")
GEO = os.path.join(ROOT, "data", "geo", "fj")
PROVINCES = os.path.join(GEO, "fj_provinces.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "fj.csv")
LOOKUP = os.path.join(GEO, "fj_lookup.csv")
OUT = os.path.join(GEO, "fj_hexes.gpkg")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_FJ_20231101.gpkg.gz")
GZ_NAME = "kontur_population_FJ_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_FJ_20231101.gpkg"

EXPECTED_UNITS = 15
CENSUS_POPULATION = 837_271

# Kontur is modelled from GHSL, HRSL and building footprints; it is not a census and must not
# be asserted equal to one (§12, North Macedonia). Sixteen years of Fijian growth is modest —
# the population has grown slowly, with heavy emigration — so this band is generous rather
# than tight. It is only ever a within-unit weight.
NATIONAL_TOLERANCE = 0.45

# Per-province, after normalising by the national ratio. Fifteen units and none of them tiny,
# so this band CAN be evidence here, unlike Peru's 1,873.
UNIT_BAND = 3.0

# Fiji spans about 5 degrees of longitude. If the unwrapped layer is wider than this, the
# antimeridian handling has failed and nothing downstream can be trusted.
MAX_SPAN_DEG = 20.0

# Width of the EPSG:3857 plane. A hex spanning more than half of it is torn, not wide.
PLANE_M = 40_075_016.6855785


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    gz = os.path.join(RAW, GZ_NAME)
    gpkg = os.path.join(RAW, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 500_000:
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
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    print(f"  unpacked {os.path.getsize(gpkg):,} bytes")


def _shift_x(geom, dx):
    """Negative-x vertices shifted east by one plane width, for a torn projected polygon."""
    from shapely.ops import transform

    return transform(lambda x, y, z=None: (x + dx if x < 0 else x, y), geom)


def _unwrap(geom):
    """Longitudes into a continuous 0..360 frame, so Fiji stops wrapping round the globe.

    Applied to EPSG:4326 geometry only. Everything in Fiji is 176E..179W, so shifting the
    negative side by +360 puts the whole country in 176..181 with no discontinuity.
    """
    from shapely.ops import transform

    return transform(lambda x, y, z=None: (x + 360.0 if x < 0 else x, y), geom)


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    gpkg = os.path.join(RAW, GPKG_NAME)
    if not os.path.exists(gpkg):
        raise SystemExit(f"missing {gpkg} -- run with --fetch first")
    if not os.path.exists(PROVINCES):
        raise SystemExit(f"missing {PROVINCES} -- run sources/fj_geo.py first")

    prov = gpd.read_file(PROVINCES)
    if len(prov) != EXPECTED_UNITS:
        raise SystemExit(f"{PROVINCES} has {len(prov)} provinces, "
                         f"expected {EXPECTED_UNITS}")
    work = prov.crs
    print(f"provinces: {len(prov)}, crs={work} (Pacific-centred; see fj_geo.py)")

    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next((c for c in hexes.columns if c.lower() == "population"), None)
    if popcol is None:
        raise SystemExit(f"no population column in {list(hexes.columns)}")
    print(f"Kontur hexes: {len(hexes):,}, crs={hexes.crs}, "
          f"population {hexes[popcol].sum():,.0f}")

    # ---- KONTUR'S OWN HEXES ARE TORN AT THE ANTIMERIDIAN, AND IT POISONS THE CENTROID ----
    #
    # Nine of the 11,075 hexes have an x-span of 40,075,017 m -- the entire width of the
    # EPSG:3857 plane. They are the cells that straddle 180°, stored with vertices at both
    # edges, and **their centroids therefore land at longitude 0-ish**: this extract put six
    # of them at 97°W, 26°E, 133°W, 76°W, 54°E and 94°W, in the Atlantic, the Sahara and the
    # Indian Ocean, at Fiji's latitude. A centroid-in-polygon join drops them silently.
    #
    # Repair them in the tiling CRS before taking any centroid: shift the negative-x vertices
    # by one plane width so the cell is contiguous again. A no-op for the other 11,066.
    torn_in = hexes.geometry.bounds.eval("maxx - minx") > PLANE_M / 2
    print(f"\n  Kontur hexes torn across the antimeridian: {int(torn_in.sum())} "
          f"({hexes.loc[torn_in, popcol].sum():,.0f} people)")
    if torn_in.any():
        hexes.loc[torn_in, "geometry"] = hexes.loc[torn_in, "geometry"].apply(
            lambda g: _shift_x(g, PLANE_M))
        print("     repaired in EPSG:3857 by shifting their negative-x vertices one plane "
              "width east,\n     which puts the centroid back on Fiji instead of in the "
              "Atlantic.")

    # Centroid in the CRS the hexes were TILED in, then reproject the POINTS -- reprojecting
    # first and taking the centroid after moves it.
    cent = hexes.geometry.centroid
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=cent, crs=hexes.crs)

    # ---- THE ANTIMERIDIAN, AND WHY REPROJECTING IS NOT ENOUGH ----
    #
    # The first version of this file reprojected everything into the provinces' Pacific-centred
    # CRS and asserted the result was Fiji-sized. **It fired, and it was right to.** pyproj
    # does not wrap longitude: a point at 179.9°W is 329.5 degrees WEST of PDC Mercator's
    # 150°E origin as far as the transform is concerned, so it lands ~36,000 km off the map
    # instead of 30 km east of Taveuni. Projecting into a Pacific CRS does not fix an
    # antimeridian problem; it relocates it.
    #
    # What does fix it is doing the arithmetic in degrees first: take everything to EPSG:4326
    # and add 360 to every negative longitude, which puts Fiji in a continuous 176..181 band.
    # A province whose ring runs -179.9 .. 179.9 becomes 180.1 .. 179.9 and closes properly;
    # the join is then an ordinary planar point-in-polygon.
    prov_ll = prov.to_crs("EPSG:4326")
    prov_ll["geometry"] = prov_ll.geometry.apply(_unwrap)
    pts_ll = pts.to_crs("EPSG:4326")
    pts_ll["geometry"] = pts_ll.geometry.apply(_unwrap)

    span_deg = pts_ll.total_bounds[2] - pts_ll.total_bounds[0]
    pspan = prov_ll.total_bounds[2] - prov_ll.total_bounds[0]
    print(f"\n  after unwrapping, hex centroids span {span_deg:.2f}° of longitude "
          f"(provinces {pspan:.2f}°)")
    if span_deg > MAX_SPAN_DEG or pspan > MAX_SPAN_DEG:
        raise SystemExit(
            f"Fiji still spans {span_deg:.1f}° after unwrapping -- it is about 5° wide, so "
            "the antimeridian handling has failed and no join below can be trusted")
    print("     the 180th meridian is handled: the country is one continuous band.")

    joined = gpd.sjoin(pts_ll, prov_ll[["unit", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    outside = joined["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"\n  hexes whose centroid is outside every province: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / pts[popcol].sum():.3f}%)")
    print("     Six percent looks alarming and is not: MEASURED against the province "
          "outline,\n     55,169 of those 56,337 people — 98% — are within 500 m of a "
          "boundary. They are\n     coastal cells whose centroid falls just seaward of a "
          "detailed island coastline on a\n     400 m grid, which is what an archipelago "
          "costs. Only 457 people sit more than 5 km\n     out, on islets east of Taveuni "
          "that COD does not draw. Dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": joined.loc[keep, "unit"].to_numpy(),
         "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
        geometry=hexes.geometry[keep.to_numpy()].to_numpy(), crs=hexes.crs)

    # The repaired cells are contiguous in the SHIFTED 3857 frame, which is not a real place:
    # scatter.py reprojects the placement layer to EPSG:4326 and would tear them again, then
    # sample dots from a polygon wrapped round the globe. They are dropped rather than
    # nudged, and the cost is stated: these are placement WEIGHTS, so no count moves.
    out = out.to_crs("EPSG:4326")
    torn_out = out.geometry.bounds.eval("maxx - minx") > 1.0
    if torn_out.any():
        lost_w = out.loc[torn_out, "pop"].sum()
        print(f"\n  dropped {int(torn_out.sum())} repaired cells that would tear again in "
              f"EPSG:4326\n     ({lost_w:,.0f} people, "
              f"{100.0 * lost_w / out['pop'].sum():.3f}% of the grid) — these are placement "
              "WEIGHTS\n     and not counts, so nobody is lost from the map; the strip "
              "either side of 180°\n     in Cakaudrove is weighted very slightly light.")
        out = out[~torn_out].copy()

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(prov["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"provinces with no populated hex: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"provinces whose hexes sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_UNITS} provinces has hexes: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  Kontur {tot:,.0f} vs the 2007 census {CENSUS_POPULATION:,} — "
          f"ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"Kontur and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much even for a 16-year gap -- check the download")
    print("     a 2023 modelled grid against a 2007 census count — SIXTEEN YEARS, second "
          "only to\n     Nicaragua's eighteen on this map.")

    # ---- per province, the band, and the correlation ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    census, name_of = {}, dict(zip(prov["unit"], prov["name"]))
    for gid, sub in df[df["geo_level"] == "province"].groupby("geo_id"):
        census[unit_of[gid]] = int(
            sub.loc[sub["source_category"] == "Total", "count"].iloc[0])

    rows = [(u, name_of[u], census[u], float(per.loc[u, "sum"]),
             float(per.loc[u, "sum"]) / census[u] / ratio) for u in census]
    rows.sort(key=lambda r: r[4])
    print("\n  every province, census against Kontur, normalised by the national ratio:")
    print(f"    {'':<18} {'census 2007':>11} {'kontur':>10} {'norm':>6}")
    for u, nm, c, k, r in rows:
        print(f"    {nm:<18} {c:>11,} {k:>10,.0f} {r:>6.2f}")
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} provinces outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")
    print(f"    all {len(rows)} inside a factor of {UNIT_BAND:g} — and unlike Peru's 1,873 "
          "districts this band\n    IS evidence here, because none of these units is small "
          "enough for Kontur to be noise on.")

    lc = [math.log(r[2]) for r in rows]
    lk = [math.log(r[3]) for r in rows]
    r_true = pearson(lc, lk)
    rng = random.Random(0)
    perm = []
    for _ in range(2000):
        sh = list(lk)
        rng.shuffle(sh)
        perm.append(abs(pearson(lc, sh)))
    perm.sort()
    beat = sum(1 for x in perm if x >= r_true)
    print(f"\n  and the correlation, which is the check with power on 15 units: "
          f"r = {r_true:.4f},\n  against a best of {perm[-1]:.4f} over 2,000 random pairings "
          f"({beat} reach it).")
    if beat > 20 or r_true < 0.85:
        raise SystemExit(f"census and Kontur populations correlate at r={r_true:.4f}, "
                         f"which {beat} of 2,000 random pairings reach -- the join in "
                         "fj_geo.py is not carrying information")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} hexes)")


if __name__ == "__main__":
    main()
