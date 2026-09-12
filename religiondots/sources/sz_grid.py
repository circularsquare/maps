"""Eswatini — the placement layer: WorldPop 100 m population, blocked to ~370 m, by region.

Writes data/geo/sz/sz_cells.gpkg.

**KONTUR IS WRONG ABOUT ESWATINI AND THIS IS THE COUNTRY THAT SHOWS IT.** Every other
coarse-geography country here (Zimbabwe, Kenya, Malawi, Benin, Myanmar, Cambodia) places its
dots on the Kontur 400 m grid, and doing that here would have been the obvious move. It fails
the per-unit band and not by a little.

Singapore is the only other country off that grid and it left for the opposite kind of reason
(sources.md §9bp): Kontur counts everybody physically present and Singapore's census counts
residents, so an ACCURATE grid was measuring the wrong people. **Here the grid is simply
inaccurate on its own terms** — same universe, same idea, wrong answer — which is the failure
mode the per-unit band exists to catch and the first time on this map that it has fired.
`kontur_population_SZ_20231101.gpkg`, normalised by its own national ratio, against the 2017
census:

      Hhohho      136,475 against a census   320,651     0.38x
      Manzini     281,632 against a census   355,945     0.71x
      Shiselweni  265,051 against a census   204,111     1.17x
      Lubombo     527,600 against a census   212,531     2.24x

Kontur puts **43% of Eswatini in Lubombo**, which the census counts at 19%, and its fifteen
largest hexes are almost all in the northern Lowveld around Simunye, Mhlume and Tshaneni. That
is the sugar-estate belt, which HOT mapped building by building; the Highveld *imiti*, the
dispersed homesteads that most Swazis actually live in, are barely in OSM at all. **Kontur is
built from building footprints, so it inherits where the mapping happened rather than where
the people are** — and in a small country one mapping campaign is enough to tip a whole
region. The boundaries were cleared first and are not the problem: Mbabane, Piggs Peak,
Manzini, Matsapha, Nhlangano, Hlatikulu, Siteki, Big Bend and Simunye each fall in the region
they belong to, and COD's four polygons reproduce the CSO's published areas to within 0.4%.

**WORLDPOP GETS IT RIGHT, AND THE MARGIN IS NOT CLOSE.** The constrained 100 m
`maxar_v1` release is built from machine-extracted Maxar building footprints rather than from
volunteered mapping, so it does not have Kontur's blind spot:

      Hhohho      0.976     Manzini  1.025     Shiselweni  0.953     Lubombo  1.039

**AND A SECOND WORLDPOP RELEASE IS THE CONTROL, NOT A SECOND OPINION.** The unconstrained
`swz_ppp_2017_UNadj` raster is a different model of a different year, and it is the year the
census was taken: 0.964 / 1.013 / 0.991 / 1.041. Two independently built rasters agreeing to
within about two points on all four regions is what makes this a weight worth trusting; a
single grid agreeing with the census could be luck, and Kontur shows what disagreement looks
like. Both are fetched and both are checked on every run.

**THE CONSTRAINED RASTER IS THE ONE DRAWN ON, and the year mismatch is the price.** Constrained
means WorldPop places nobody outside mapped built-up land, so no dots land in Malolotja, Hlane
or Mlawula, and that is §8.2's whole purpose. The 2017 unconstrained raster matches the census
year and spreads people across open country, which is the failure the grid exists to avoid. So
the better year loses to the better placement, deliberately, and the control keeps the choice
honest.

The 100 m raster is summed in 4x4 blocks to ~370 m cells, which is Kontur's resolution and
keeps the layer at a comparable size. It is only a within-region weight, so the absolute level
never matters and only the shape does.

THE JOIN IS SPATIAL, on cell CENTROIDS, so no cell is split between two regions.

Usage:
    python sources/sz_grid.py --fetch    two GeoTIFFs from WorldPop, ~11 MB, seconds
    python sources/sz_grid.py            rebuild from data/raw/sz/
"""

import os
import sys

# [[feedback_cap_cpu]] — before numpy is imported anywhere below it.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sz")
GEO = os.path.join(ROOT, "data", "geo", "sz")
REGIONS = os.path.join(GEO, "sz_regions.gpkg")
NORM = os.path.join(ROOT, "data", "normalized", "sz.csv")
LOOKUP = os.path.join(GEO, "sz_lookup.csv")
OUT = os.path.join(GEO, "sz_cells.gpkg")

WP = "https://data.worldpop.org/GIS/Population/"
# The one drawn on: constrained to mapped built-up land, Maxar footprints, 2020.
DRAWN_URL = WP + "Global_2000_2020_Constrained/2020/maxar_v1/SWZ/swz_ppp_2020_UNadj_constrained.tif"
DRAWN_NAME = "swz_ppp_2020_UNadj_constrained.tif"
# The control: a different model of a different year, and that year is the census's.
CONTROL_URL = WP + "Global_2000_2020/2017/SWZ/swz_ppp_2017_UNadj.tif"
CONTROL_NAME = "swz_ppp_2017_UNadj.tif"

BLOCK = 4                       # 100 m cells -> ~370 m, Kontur's resolution
EXPECTED_REGIONS = 4
CENSUS_POPULATION = 1_093_238

# WorldPop is modelled and is not the census; it must not be asserted equal to it (§12,
# North Macedonia). Only a within-region weight, so the level does not matter and the shape
# does. The national ratio is 1.06 for the 2020 raster (three years of growth) and 1.02 for
# the 2017 one, so 0.30 is generous on both.
NATIONAL_TOLERANCE = 0.30

# Measured: the drawn raster reads 0.953-1.039 and the control 0.964-1.041, so 1.15 is
# roughly three times the worst observed miss and still nowhere near Kontur's 0.38/2.24.
UNIT_BAND = 1.15


def _get(url, name, floor):
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, name)
    if os.path.exists(dest) and os.path.getsize(dest) > floor:
        print("already have", dest)
        return
    print("GET", url)
    r = requests.get(url, timeout=1800, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # §5a: a 200 is not a download. Both byte orders and both TIFF versions — WorldPop
    # ships the constrained rasters as BigTIFF (`II+\0`, version 43) and the unconstrained
    # ones as classic TIFF (`II*\0`, version 42), so checking only for `*` rejects the very
    # file this script is built on.
    with open(dest, "rb") as fh:
        magic = fh.read(4)
    if magic not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        raise SystemExit(f"{dest} is not a TIFF -- starts {magic!r}, "
                         f"{os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def fetch():
    _get(DRAWN_URL, DRAWN_NAME, 1_000_000)
    _get(CONTROL_URL, CONTROL_NAME, 5_000_000)


def _blocks(path):
    """Sum a WorldPop raster into BLOCK x BLOCK cells. Returns (pop, west, north, dx, dy).

    `pop` is a 2-D array of block sums; the rest describe the block grid's geo-transform,
    which is the raster's own scaled by BLOCK. WorldPop ships EPSG:4326 with -99999 for the
    sea and for unmodelled land; anything not finite or negative is zeroed rather than
    propagated, because a NaN in a weight silently removes a whole region's dots.
    """
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise SystemExit(f"{path}: expected EPSG:4326, got {src.crs}")
        a = src.read(1).astype("float64")
        t = src.transform
        nodata = src.nodata
    a[~np.isfinite(a)] = 0.0
    if nodata is not None:
        a[a == nodata] = 0.0
    a[a < 0] = 0.0

    h, w = a.shape
    hh, ww = h // BLOCK * BLOCK, w // BLOCK * BLOCK
    if hh != h or ww != w:
        # Trimming would drop people. Pad to a whole number of blocks instead.
        pad = np.zeros(((h + BLOCK - 1) // BLOCK * BLOCK,
                        (w + BLOCK - 1) // BLOCK * BLOCK))
        pad[:h, :w] = a
        a = pad
        h, w = a.shape
    pop = a.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).sum(axis=(1, 3))
    return pop, t.c, t.f, t.a * BLOCK, t.e * BLOCK


def _cells(path):
    """The populated blocks of a raster as a GeoDataFrame of square polygons."""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box

    pop, west, north, dx, dy = _blocks(path)
    rows, cols = np.nonzero(pop > 0)
    vals = pop[rows, cols]
    x0 = west + cols * dx
    y0 = north + rows * dy          # dy is negative in a north-up raster
    geom = [box(a, min(b, b + dy), a + dx, max(b, b + dy))
            for a, b in zip(x0, y0)]
    return gpd.GeoDataFrame({"pop": vals}, geometry=geom, crs="EPSG:4326")


def _by_region(cells, reg):
    """Cell populations summed per region, joined on centroids.

    The centroid is taken from the BOUNDS rather than from `.centroid`, which is both exact
    (these are axis-aligned boxes) and silences geopandas' geographic-CRS warning instead of
    ignoring it. Reprojecting to take a centroid and reprojecting back would move it.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    b = cells.geometry.bounds
    pts = gpd.GeoDataFrame(
        {"pop": cells["pop"].to_numpy()},
        geometry=[Point(x, y) for x, y in zip((b["minx"] + b["maxx"]) / 2.0,
                                              (b["miny"] + b["maxy"]) / 2.0)],
        crs=cells.crs)
    j = gpd.sjoin(pts, reg[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    return j


def _band(label, per, census, ratio, name_of):
    rows = [(u, name_of[u], census[u], float(per[u]), float(per[u]) / census[u] / ratio)
            for u in census]
    rows.sort(key=lambda r: r[4])
    print(f"\n  {label}, normalised by its own national ratio {ratio:.3f}:")
    print(f"    {'':<14} {'census':>10} {'grid':>10} {'norm':>6}")
    for u, nm, c, k, r in rows:
        print(f"    {nm:<14} {c:>10,} {k:>10,.0f} {r:>6.2f}")
    return rows


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    drawn = os.path.join(RAW, DRAWN_NAME)
    control = os.path.join(RAW, CONTROL_NAME)
    for p in (drawn, control):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run with --fetch first")
    if not os.path.exists(REGIONS):
        raise SystemExit(f"missing {REGIONS} -- run sources/sz_geo.py first")

    reg = gpd.read_file(REGIONS)
    if len(reg) != EXPECTED_REGIONS:
        raise SystemExit(f"{REGIONS} has {len(reg)} regions, expected {EXPECTED_REGIONS}")
    name_of = dict(zip(reg["unit"], reg["name"]))

    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(LOOKUP, dtype=str)
    unit_of = dict(zip(lut["geo_id"], lut["unit"]))
    census = {}
    for gid, sub in df[df["geo_level"] == "region"].groupby("geo_id"):
        census[unit_of[gid]] = int(
            sub.loc[sub["source_category"] == "Total", "count"].iloc[0])
    if sum(census.values()) != CENSUS_POPULATION:
        raise SystemExit(f"sz.csv region totals sum to {sum(census.values()):,}, "
                         f"expected {CENSUS_POPULATION:,}")

    cells = _cells(drawn)
    print(f"WorldPop {DRAWN_NAME}: {len(cells):,} populated {BLOCK}x{BLOCK} blocks "
          f"(~370 m), population {cells['pop'].sum():,.0f}")

    j = _by_region(cells, reg)
    outside = j["unit"].isna()
    lost = float(j.loc[outside, "pop"].sum())
    print(f"\n  cells whose centroid is outside every region: {int(outside.sum()):,} "
          f"({lost:,.0f} people, {100.0 * lost / cells['pop'].sum():.3f}%)")
    print("     WorldPop's SWZ raster is clipped to the national outline, so this is the "
          "sliver\n     between the raster edge and COD's polygons; dropped.")

    keep = ~outside
    out = gpd.GeoDataFrame(
        {"unit": j.loc[keep, "unit"].to_numpy(),
         "pop": j.loc[keep, "pop"].to_numpy(dtype=float)},
        geometry=cells.geometry[keep.to_numpy()].to_numpy(), crs=reg.crs)

    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(reg["unit"]) - set(per.index))
    if missing:
        raise SystemExit(f"regions with no populated cell: {missing}")
    zero = sorted(per.index[per["sum"] <= 0])
    if zero:
        raise SystemExit(f"regions whose cells sum to zero population: {zero}")
    print(f"  every one of the {EXPECTED_REGIONS} regions has cells: "
          f"{per['size'].min():,}–{per['size'].max():,} each")

    tot = float(out["pop"].sum())
    ratio = tot / CENSUS_POPULATION
    print(f"\n  WorldPop {tot:,.0f} vs census {CENSUS_POPULATION:,} — ratio {ratio:.3f}")
    if abs(ratio - 1.0) > NATIONAL_TOLERANCE:
        raise SystemExit(f"WorldPop and the census disagree by {abs(ratio - 1) * 100:.0f}%, "
                         "which is too much for a weight -- check the download")

    rows = _band("the drawn raster (constrained, 2020)",
                 {u: per.loc[u, "sum"] for u in census}, census, ratio, name_of)
    worst = [r for r in rows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if worst:
        raise SystemExit(f"{len(worst)} regions outside a factor of {UNIT_BAND:g}: "
                         f"{[(w[1], round(w[4], 2)) for w in worst]}")

    # ---- THE CONTROL: a different WorldPop model, of the census's own year ----
    cc = _cells(control)
    jc = _by_region(cc, reg)
    cper = jc[jc["unit"].notna()].groupby("unit")["pop"].sum()
    cratio = float(cper.sum()) / CENSUS_POPULATION
    crows = _band("the control raster (unconstrained, 2017, the census year)",
                  {u: cper[u] for u in census}, census, cratio, name_of)
    cworst = [r for r in crows if r[4] < 1 / UNIT_BAND or r[4] > UNIT_BAND]
    if cworst:
        raise SystemExit(f"the CONTROL raster puts {len(cworst)} regions outside a factor "
                         f"of {UNIT_BAND:g}: {[(w[1], round(w[4], 2)) for w in cworst]} -- "
                         "two WorldPop releases disagreeing about a region means the weight "
                         "is not trustworthy, whichever one is drawn")
    a = {r[0]: r[4] for r in rows}
    b = {r[0]: r[4] for r in crows}
    gap = max(abs(a[u] - b[u]) for u in a)
    print(f"\n  the two rasters agree to within {gap:.3f} on every region — different "
          f"models, different\n  years, built from different inputs. THAT is what makes "
          "this weight worth trusting;\n  Kontur reads 0.38 to 2.24 on the same four "
          "polygons and this script's docstring says why.")
    if gap > 0.10:
        raise SystemExit(f"the two WorldPop releases differ by {gap:.3f} on some region, "
                         "which is too much for one to be a control on the other")

    os.makedirs(GEO, exist_ok=True)
    out.to_file(OUT, layer="cells", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells)")


if __name__ == "__main__":
    main()
