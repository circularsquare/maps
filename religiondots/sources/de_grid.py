"""Germany — the 1km INSPIRE grid as a PLACEMENT layer, with measured per-religion weights.

Writes data/geo/de/de_grid_1km.gpkg, which replaces de_gemeinden.gpkg as the country's
`place` layer.  Counts still come from the Gemeinde table; this file only decides WHERE
inside a Gemeinde a dot goes.

WHY, IN ONE NUMBER: Berlin is 3,596,999 people in a single Gemeinde polygon of 891 km².
Placed on that polygon, its 3,596 dots are scattered uniformly over the whole city, which
says nothing about Berlin and actively misleads — Neukölln and Zehlendorf come out
identical.  Every large German city has the same problem and 78 Gemeinden hold 31.6% of
the country (sources/de_geo.md §4).

WHAT MAKES GERMANY DIFFERENT FROM EVERY OTHER COUNTRY HERE.  spec §8.2 separates two jobs:
the magnitude per unit, and the placement within it.  Everywhere else the placement layer
is a *proxy* — US census tracts are engineered to ~4,000 people so an equal share is a
population weighting (§8.2), and §8.4 goes further and FITS a demographic model to guess
where a denomination sits inside a county.  Germany needs neither, because destatis
publishes **the same three categories on the grid itself**:

    GITTER_ID_1km;x_mp_1km;y_mp_1km;Insgesamt_Bevoelkerung;
    Roemisch_katholisch;Evangelisch;Sonstige_keine_ohneAngabe

So a Catholic dot in Munich can be placed on where Munich's *Catholics* are, not on where
Munich's people are — and that is a measurement, not a model.  Germany is the only country
on the map where §8.2's approximation is dropped outright rather than bounded.

WHY 1km AND NOT THE 100m FILE.  The 100m grid has 3,088,036 populated cells and Germany
draws 82,710 dots, so it would be 37 cells per dot — the placement layer would be far finer
than anything that can be shown, at 14× the file.  At 1km there are 210,555 cells, a median
of about 390 people each, which is finer than every count unit in this project except the
UK's Output Areas; Berlin gets ~890 cells for ~3,596 dots.  The dot value is the binding
constraint, not the grid.

TWO APPROXIMATIONS, both stated rather than hidden:

  * A cell is assigned to the Gemeinde containing its CENTRE.  Cells straddle boundaries —
    destatis assigns each address to the cell holding its coordinate and never clips — so a
    boundary cell's people may belong to either side.  The effect is bounded by the cell
    size and it only moves WEIGHT within a Gemeinde, never a count.
  * Each cell is then CLIPPED to its assigned Gemeinde, so a dot can never land outside its
    own unit, in the sea, or in Poland.  Without this the coastal and border squares hang
    over the edge and dots fall in the Baltic.

The grid and the Gemeinde table are perturbed INDEPENDENTLY (Cell-Key, sources/de.md §3),
so grid sums per Gemeinde do not equal the Gemeinde table and are not meant to.  Only
relative shares within a Gemeinde are used, and the reconciliation below reports the
agreement rather than asserting it.

Usage:
    python sources/de_grid.py           # needs data/raw/de/Religion_gitter.zip and
                                        # data/geo/de/de_gemeinden.gpkg
    python sources/de_grid.py --fetch   # download the 27MB grid zip first
"""

import os
import shutil
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "de")
GEO_DIR = os.path.join(ROOT, "data", "geo", "de")
GEMEINDEN = os.path.join(GEO_DIR, "de_gemeinden.gpkg")
OUT = os.path.join(GEO_DIR, "de_grid_1km.gpkg")

ZIP_NAME = "Religion_gitter.zip"
ZIP_URL = "https://www.destatis.de/static/DE/zensus/gitterdaten/Religion.zip"
MIN_BYTES = 25_000_000
CSV_NAME = "Zensus2022_Religion_1km-Gitter.csv"

DASH = "–"          # "Genau Null oder auf Null geändert"
CELL = 1000         # metres
GRID_CRS = 3035     # ETRS89-LAEA Europe, per the INSPIRE cell id
N_CELLS = 210_556      # the file has no trailing newline, so `wc -l` reports one fewer

# source column -> the column name written into the gpkg, and the node it weights
COLUMNS = {
    "Roemisch_katholisch": "kath",
    "Evangelisch": "ev",
    "Sonstige_keine_ohneAngabe": "son",
}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ZIP_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_BYTES:
        print("already have", dest)
        return
    print("downloading", ZIP_URL)
    urllib.request.urlretrieve(ZIP_URL, dest)
    size = os.path.getsize(dest)
    if size < MIN_BYTES or not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is {size:,} bytes and "
                         f"{'is' if zipfile.is_zipfile(dest) else 'is NOT'} a zip -- "
                         f"expected >= {MIN_BYTES:,} (spec §12)")
    print(f"  {size:,} bytes")


def csv_path():
    p = os.path.join(RAW, CSV_NAME)
    if os.path.exists(p):
        return p
    src = os.path.join(RAW, ZIP_NAME)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    with zipfile.ZipFile(src) as z:
        with z.open(CSV_NAME) as fh, open(p, "wb") as out:
            shutil.copyfileobj(fh, out)
    print(f"  extracted {CSV_NAME}")
    return p


def cell(v, where):
    """A count cell -> int.  The dash is a true zero; anything else unrecognised raises."""
    s = v.strip()
    if s == DASH:
        return 0
    if s.lstrip("-").isdigit():
        return int(s)
    raise ValueError(f"{where}: unrecognised count {v!r}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    path = csv_path()

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely import box

    df = pd.read_csv(path, sep=";", encoding="utf-8", dtype=str)
    print(f"\n  {len(df):,} grid cells in {CSV_NAME}")
    if len(df) != N_CELLS:
        print(f"  NOTE expected {N_CELLS:,} cells; the file has been reissued")

    for src, dst in COLUMNS.items():
        df[dst] = [cell(v, f"{i} {src}") for i, v in zip(df["GITTER_ID_1km"], df[src])]
    df["pop"] = [cell(v, "pop") for v in df["Insgesamt_Bevoelkerung"]]

    # The cell id encodes the LOWER-LEFT corner: CRS3035RES1000mN2689000E4337000.
    # It is the authoritative geometry; the x_mp/y_mp columns are the centre and are used
    # only to check the parse, so a change in either is caught rather than assumed.
    ids = df["GITTER_ID_1km"].str.extract(
        r"^CRS3035RES(?P<res>\d+)mN(?P<n>\d+)E(?P<e>\d+)$")
    if ids.isna().any().any():
        bad = df.loc[ids.isna().any(axis=1), "GITTER_ID_1km"].head(3).tolist()
        raise SystemExit(f"cell ids not in the INSPIRE form: {bad}")
    res = ids["res"].astype(int)
    if not (res == CELL).all():
        raise SystemExit(f"cell ids claim resolutions {sorted(res.unique())}, expected {CELL}")
    x0 = ids["e"].astype(np.int64).to_numpy()
    y0 = ids["n"].astype(np.int64).to_numpy()

    dx = np.abs((x0 + CELL // 2) - df["x_mp_1km"].astype(np.int64).to_numpy()).max()
    dy = np.abs((y0 + CELL // 2) - df["y_mp_1km"].astype(np.int64).to_numpy()).max()
    print(f"  OK  cell id parses to the published centre (max off-by {dx}m, {dy}m)"
          if dx == 0 and dy == 0 else
          f"  BAD cell id vs published centre differs by up to {dx}m, {dy}m")
    if dx or dy:
        raise SystemExit("cell id geometry disagrees with the published centre columns")

    g = gpd.GeoDataFrame(
        df[["kath", "ev", "son", "pop"]].copy(),
        geometry=box(x0, y0, x0 + CELL, y0 + CELL), crs=GRID_CRS)

    # ---- assign each cell to the Gemeinde containing its CENTRE
    gem = gpd.read_file(GEMEINDEN).to_crs(GRID_CRS)
    print(f"  {len(gem):,} Gemeinde polygons")
    centres = gpd.GeoDataFrame(geometry=g.geometry.centroid, crs=GRID_CRS)
    hit = gpd.sjoin(centres, gem[["ars", "geometry"]], how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]          # a centre on a shared edge
    g["ars"] = hit["ars"].to_numpy()

    lost = g["ars"].isna()
    print(f"\n  cells whose centre is in no Gemeinde: {lost.sum():,} "
          f"({g.loc[lost, 'pop'].sum():,} people, "
          f"{100.0 * g.loc[lost, 'pop'].sum() / g['pop'].sum():.3f}%)")
    print("      these are gemeindefreie Gebiete and cells centred just off the coast or "
          "over a border; their people are still counted in the Gemeinde table and are "
          "placed by their Gemeinde's other cells")
    g = g[~lost].copy()

    # ---- clip each cell to its own Gemeinde, so no dot leaves its unit
    print(f"  clipping {len(g):,} cells to their Gemeinde…")
    poly = gem.set_index("ars")["geometry"]
    g["geometry"] = g.geometry.intersection(
        gpd.GeoSeries(g["ars"].map(poly).to_numpy(), crs=GRID_CRS, index=g.index))
    empty = g.geometry.is_empty | g.geometry.isna()
    if empty.any():
        print(f"  dropped {empty.sum():,} cells whose clip came out empty "
              f"({g.loc[empty, 'pop'].sum():,} people)")
        g = g[~empty].copy()

    # ---- reconciliation: REPORTED, not asserted.  The two tables are perturbed apart.
    import csv as _csv
    cen = {}
    with open(os.path.join(ROOT, "data", "normalized", "de.csv"), encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if r["geo_level"] == "gemeinde":
                cen.setdefault(r["geo_id"], {})[r["source_category"]] = int(r["count"])
    key = {"kath": "Römisch-katholische Kirche (öffentlich-rechtlich)",
           "ev": "Evangelische Kirche (öffentlich-rechtlich)",
           "son": "Sonstige, keine, ohne Angabe",
           "pop": "Einwohnerzahl"}
    print("\n  grid total vs Gemeinde table (independently perturbed; both are the source's):")
    for col, label in key.items():
        grid = int(g[col].sum())
        table = sum(v.get(label, 0) for v in cen.values())
        print(f"    {col:5s} {grid:>12,}  vs {table:>12,}   "
              f"{100.0 * (grid / table - 1):+7.3f}%")

    covered = g.groupby("ars")["pop"].sum()
    n_gem = len(cen)
    print(f"\n  Gemeinden with at least one cell: {covered.size:,} of {n_gem:,}")

    # THE LAYER MUST COVER EVERY GEMEINDE, or scatter.py has nowhere to put those units'
    # dots and they vanish without a word.  A Gemeinde can end up with no cell for two
    # reasons — it is small enough that no 1km centre lands inside it, or its cells were
    # all centred just outside — so the fallback is its own polygon, carrying its own
    # census counts, added as a single "cell".  That restores exactly the behaviour the
    # Gemeinde-only layer had, for the 0.01% of the country that needs it.
    missing = sorted(set(cen) - set(covered.index))
    if missing:
        rows = gem[gem["ars"].isin(missing)].copy()
        for col, label in (("kath", key["kath"]), ("ev", key["ev"]),
                           ("son", key["son"]), ("pop", key["pop"])):
            rows[col] = rows["ars"].map(lambda a: cen[a].get(label, 0))
        pop_missing = int(rows["pop"].sum())
        print(f"  {len(missing):,} Gemeinden have NO 1km cell ({pop_missing:,} people, "
              f"{100.0 * pop_missing / sum(v['Einwohnerzahl'] for v in cen.values()):.4f}%)"
              f" -- their own polygon is added as a single cell so nothing is unplaceable")
        g = pd.concat([g, rows[["ars", "kath", "ev", "son", "pop", "geometry"]]],
                      ignore_index=True)
        g = gpd.GeoDataFrame(g, geometry="geometry", crs=GRID_CRS)

    still = set(cen) - set(g["ars"])
    if still:
        raise SystemExit(f"{len(still)} Gemeinden still have no placement geometry: "
                         f"{sorted(still)[:5]}")
    print(f"  OK  every one of the {n_gem:,} Gemeinden has placement geometry")

    big = covered.sort_values(ascending=False).head(6)
    print("\n  cells per polygon, where it matters most:")
    for ars in big.index:
        n = int((g["ars"] == ars).sum())
        print(f"      {ars}  {n:>6,} cells  ({cen[ars]['Einwohnerzahl']:>10,} people)")

    out = g[["ars", "kath", "ev", "son", "pop", "geometry"]].to_crs(4326)
    out.to_file(OUT, layer="grid1km", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} cells, EPSG:4326)")


if __name__ == "__main__":
    main()
