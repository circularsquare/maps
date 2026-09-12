"""Bulgaria — the 265 obshtini, and the census's own 1 km grid to place dots inside them.

Writes:
    data/geo/bg/bg_obshtini.gpkg    265 polygons, `lau` + `name`
    data/geo/bg/bg_grid_1km.gpkg    cell/obshtina pieces, `cellcode` + `unit` + `pop`

**THE JOIN IS FREE, WHICH IS RARE HERE.** GISCO's LAU 2021 file carries Bulgaria as **265
units whose `LAU_ID` is NSI's own obshtina code verbatim** — `VID09`, `SML31`, `KRZ07` — the
same strings the census workbook keys on. So the crosswalk is the identity function and
§12's second shape of failure (a confident wrong pairing) cannot arise: `check()` asserts the
two sets are equal, and 265 = 265 with no leftovers on either side. es_geo.py's note that
GISCO LAU is the boundary answer for the EU holds for one more country.

**AND THE PLACEMENT LAYER IS MEASURED, NOT MODELLED.** NSI publishes `POPGRID2021_1000M`,
Census 2021 on 1 km cells of the ETRS89-LAEA grid (EPSG:3035), as a file geodatabase. It is
aggregated from **the point location of every census record** rather than disaggregated from
areas, which is what the 2011 grid did and why NSI says the two are not comparable. So
Bulgaria joins Germany and Slovakia on the short list of countries here whose dots sit on a
real population surface rather than on Kontur's model or on equal shares.

**IT HOLDS 6,461,591 OF THE 6,519,789, AND THE 58,198 MISSING ARE DOCUMENTED RATHER THAN
LOST.** NSI's own metadata (§15.1, table *Unallocated population*) reports exactly this
figure: 0.9% of the census has no usable point location and is on no cell. `_grid()` asserts
the published number, so a silently different grid fails the build instead of quietly
reweighting the country. Nothing is dropped from the counts by this: the missing 0.9% affects
only the *shape* of the placement weight inside an obshtina, and every obshtina's dot total
still comes from the workbook.

**THE LICENCE PERMITS THIS USE AND WOULD NOT PERMIT A GRID MAP.** NSI's terms (§8.3) allow
"analytical developments and mapping products based on the statistical information in the
dataset ... but without showing the original values in the individual grid cells", and forbid
redistributing the dataset in its original form. The grid is used here only as a within-unit
weight; no cell value is published, and the tiles carry dots per religion.

**But it is not an independent check on the counts** (§9av). The grid is the same
enumeration as the workbook, so the ratio band in `check()` says nothing about whether the
census is right; it validates the cell-to-obshtina assignment, which is the only join.

**CELLS ARE SPLIT BY AREA, NOT ASSIGNED BY CENTRE — sk_geo.py's rule, and Bulgaria needs it
for the opposite reason.** Slovakia's problem was municipalities smaller than a cell.
Bulgaria's obshtini are large (median 340 km²), so few cells straddle a boundary inland, but
the Danube and Black Sea edges are all border cells: assigning by centre throws away the
share of a cell whose square lies over Romania or the sea. Intersecting and renormalising
against the area actually inside the country keeps those people with the obshtina they live
in.

**THE GRID IS CLIPPED TO NON-EMPTY CELLS FIRST AND THAT IS A 40x SAVING.** 112,883 cells
cover Bulgaria's bounding box; **28,507 of them hold anybody**. Overlaying only those against
265 polygons is seconds rather than minutes, and an all-zero cell contributes nothing to a
population weight by construction. An obshtina left with no cell at all still gets its own
polygon as a fallback area, same as Slovakia.

Usage:
    python sources/bg_geo.py          build from data/raw/bg/ (sources/bg.py --fetch got it)
"""

import os
import sys
import zipfile

# [[feedback_cap_cpu]] — before numpy is imported by anything below, or BLAS takes the box.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bg")
GEO = os.path.join(ROOT, "data", "geo", "bg")
NORMALIZED = os.path.join(ROOT, "data", "normalized", "bg.csv")

LAU = os.path.join(ROOT, "data", "geo", "lau2021", "shp4326",
                   "LAU_RG_01M_2021_4326.shp")
GRID_ZIP = os.path.join(RAW, "GRID2021.zip")
GRID_GDB = os.path.join(RAW, "Grid_POP2021.gdb")
GRID_LAYER = "grid_1km_poly_BG_POP_Census2021"

UNITS_GPKG = os.path.join(GEO, "bg_obshtini.gpkg")
PLACE_GPKG = os.path.join(GEO, "bg_grid_1km.gpkg")

WGS84 = "EPSG:4326"
METRIC = "EPSG:3035"             # the grid's own CRS, so the cells stay square
NATIONAL = 6_519_789
# NSI's metadata §15.1: 58,198 people (0.9%) have no point location and are on no cell.
GRID_TOTAL = 6_461_591
EXPECTED_UNITS = 265
EXPECTED_CELLS = 112_883


def _units():
    import geopandas as gpd

    if not os.path.exists(LAU):
        raise SystemExit(f"missing {LAU} — the GISCO LAU 2021 shapefile is a shared asset, "
                         "see sources/es_geo.py for where it comes from")
    g = gpd.read_file(LAU, where="CNTR_CODE = 'BG'")
    g = g.rename(columns={"LAU_ID": "lau", "LAU_NAME": "name"})
    g["lau"] = g["lau"].astype(str).str.strip()
    g["name"] = g["name"].astype(str).str.strip()
    if len(g) != EXPECTED_UNITS:
        raise SystemExit(f"{len(g)} BG LAU polygons, expected {EXPECTED_UNITS}")
    if g["lau"].duplicated().any():
        raise SystemExit("duplicate LAU_ID in the Bulgarian polygons")
    return g[["lau", "name", "geometry"]].to_crs(WGS84)


def _grid():
    import geopandas as gpd
    import pandas as pd

    if not os.path.isdir(GRID_GDB):
        if not os.path.exists(GRID_ZIP):
            raise SystemExit(f"missing {GRID_ZIP} — run `python sources/bg.py --fetch`")
        zipfile.ZipFile(GRID_ZIP).extractall(RAW)
        print(f"  extracted {os.path.basename(GRID_ZIP)}")
    g = gpd.read_file(GRID_GDB, layer=GRID_LAYER)
    if len(g) != EXPECTED_CELLS:
        raise SystemExit(f"{len(g)} grid cells, expected {EXPECTED_CELLS:,}")
    g["pop"] = pd.to_numeric(g["T"], errors="coerce").fillna(0.0)
    total = float(g["pop"].sum())
    if int(round(total)) != GRID_TOTAL:
        raise SystemExit(f"the grid holds {total:,.0f} people, NSI's metadata says "
                         f"{GRID_TOTAL:,} — this is not the Census 2021 1 km grid")
    print(f"  grid: {len(g):,} cells summing to {total:,.0f}, NSI's published allocated "
          f"total ({100.0 * total / NATIONAL:.1f}% of the census; the rest has no point "
          f"location)")
    g = g[g["pop"] > 0].copy()
    print(f"  {len(g):,} cells hold anybody; the empty ones are dropped before the overlay")
    return g.rename(columns={"GRD_ID": "cellcode"})[["cellcode", "pop", "geometry"]]


def build():
    import geopandas as gpd
    import pandas as pd

    units = _units()
    grid = _grid()

    m_units = units.to_crs(METRIC)
    m_grid = grid.to_crs(METRIC) if str(grid.crs) != METRIC else grid

    pieces = gpd.overlay(m_grid[["cellcode", "pop", "geometry"]],
                         m_units[["lau", "geometry"]],
                         how="intersection", keep_geom_type=True)
    pieces = pieces[pieces.geometry.notna() & ~pieces.geometry.is_empty].copy()
    pieces["piece_area"] = pieces.geometry.area

    # A BORDER CELL'S PEOPLE ARE ALL IN BULGARIA EVEN WHERE ITS SQUARE IS NOT. Dividing by
    # the full 1 km² would lose the share lying over Romania, Serbia, Greece, Türkiye or the
    # Black Sea. Normalise against the area actually inside an obshtina instead.
    inside = pieces.groupby("cellcode")["piece_area"].transform("sum")
    pieces["pop"] = pieces["pop"] * (pieces["piece_area"] / inside)
    pieces = pieces.rename(columns={"lau": "unit"})
    # a cell split across two obshtini yields two pieces, so the id must carry both
    pieces["cellcode"] = pieces["cellcode"].astype(str) + "|" + pieces["unit"].astype(str)

    placed = float(pieces["pop"].sum())
    lost = GRID_TOTAL - placed
    print(f"  {len(pieces):,} cell/obshtina pieces; {placed:,.0f} of the grid's "
          f"{GRID_TOTAL:,} fall inside an obshtina ({lost:,.0f} outside, "
          f"{100.0 * lost / GRID_TOTAL:.3f}%)")

    missing = sorted(set(units["lau"]) - set(pieces["unit"]))
    if missing:
        print(f"  {len(missing)} obshtina(s) intersect no populated cell; adding their own "
              f"polygon as a placement area")
        patch = m_units[m_units["lau"].isin(missing)][["lau", "geometry"]].copy()
        patch = patch.rename(columns={"lau": "unit"})
        patch["cellcode"] = "obshtina:" + patch["unit"]
        patch["pop"] = 1.0            # its only cell, so the share is 1 whatever the number
        pieces = gpd.GeoDataFrame(pd.concat([pieces, patch], ignore_index=True),
                                  geometry="geometry", crs=m_units.crs)

    place = pieces[["cellcode", "unit", "pop", "geometry"]].to_crs(WGS84)

    os.makedirs(GEO, exist_ok=True)
    units.to_file(UNITS_GPKG, layer="obshtini", driver="GPKG")
    place.to_file(PLACE_GPKG, layer="grid", driver="GPKG")
    print(f"  wrote {UNITS_GPKG} ({len(units):,} polygons)")
    print(f"  wrote {PLACE_GPKG} ({len(place):,} cells)")
    return units, place, lost


def check(units, place, lost):
    import pandas as pd

    ok = True
    counts = pd.read_csv(NORMALIZED, dtype={"geo_id": str})
    codes = set(counts.loc[counts["geo_level"] == "obshtina", "geo_id"])

    good = set(units["lau"]) == codes
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} GISCO's {len(units):,} LAU codes are exactly the "
          f"{len(codes):,} obshtina codes in bg.csv")
    if not good:
        print(f"      only in GISCO: {sorted(set(units['lau']) - codes)[:6]}")
        print(f"      only in bg.csv: {sorted(codes - set(units['lau']))[:6]}")

    covered = set(place["unit"])
    good = covered == set(units["lau"])
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} every obshtina has somewhere to put its dots "
          f"({len(covered):,} of {len(units):,})")

    good = abs(lost) / GRID_TOTAL < 0.001
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} grid population falling outside every obshtina is "
          f"{lost:,.0f} ({100.0 * lost / GRID_TOTAL:.3f}%, want <0.1%)")

    # THE RATIO BAND, AND WHAT IT CANNOT SAY (§9av). The grid IS Census 2021, so agreement
    # is not evidence about the census. It is evidence about the cell-to-obshtina join.
    per = place.groupby("unit")["pop"].sum()
    cen = counts[(counts["geo_level"] == "obshtina") & (counts["source_category"] == "Общо")]
    cen = cen.set_index("geo_id")["count"]
    both = pd.concat([per.rename("grid"), cen.rename("census")], axis=1).dropna()
    both = both[both["census"] > 0]
    both["ratio"] = both["grid"] / both["census"]
    inband = both["ratio"].between(0.5, 2.0)
    good = float(inband.mean()) > 0.97
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} grid/census ratio within 0.5-2.0 for "
          f"{100.0 * inband.mean():.2f}% of obshtini (median {both['ratio'].median():.3f})")
    print("      NOT an independent check on the counts (§9av) — the grid IS Census 2021. "
          "It checks the\n      cell-to-obshtina assignment and nothing else.")
    worst = both.reindex(both["ratio"].sub(1).abs().sort_values(ascending=False).index).head(5)
    for code, r in worst.iterrows():
        name = units.loc[units["lau"] == code, "name"]
        label = str(name.iloc[0]) if len(name) else "?"
        print(f"      {code} {label[:26]:26s} grid {r['grid']:>9,.0f} "
              f"census {r['census']:>9,.0f}  ratio {r['ratio']:.3f}")

    if not ok:
        raise SystemExit("geometry checks FAILED")


def main():
    units, place, lost = build()
    check(units, place, lost)


if __name__ == "__main__":
    main()
