"""Poland placement layer: Kontur hexes keyed to religiondots' 2,477 gminas.

    python sources/pl_geo.py

-> data/geo/pl/pl_hexes.gpkg (unit = six-digit TERYT gmina code, pop = Kontur people)

UNITS. religiondots' `data/geo/pl/pl_gminy.gpkg` (Eurostat GISCO LAU 2021, keyed on the first six
digits of TERYT; religiondots/sources/pl_geo.md explains the decoding and its name check), read
only. The language table (sources/pl_nsp.py) is on the same 2,477 gminas of 2021, so the join is
`geo_id[:6]` -> `kod`, asserted both ways here and in countries/pl.py.

WHY HEXES AND NOT THE GMINA POLYGONS religiondots draws on. religiondots spreads a gmina's dots
evenly over its polygon, which puts Warsaw's 1.8 million evenly over its forests and the Vistula
(religiondots/sources/pl_geo.md §4). Kontur's population hexes place them where people live; each
gmina's dot count still comes from the census. A gmina averages 15,000 people over 126 km², far
above Kontur's 0.74 km² hex, so a centroid join loses no gmina (checked below).

A GMINA WITH NO POPULATED HEX would draw nothing, so it gets its own polygon at pop 0 (placed on
equal shares). None is expected; the count is printed.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
from _grid import hex_layer  # noqa: E402

RD_UNITS = HERE.parent / "religiondots" / "data" / "geo" / "pl" / "pl_gminy.gpkg"
NORM = HERE / "data" / "normalized" / "pl.csv"
OUT = HERE / "data" / "geo" / "pl" / "pl_hexes.gpkg"


def main():
    units = gpd.read_file(RD_UNITS)
    units["unit"] = units["kod"].astype(str)
    assert len(units) == 2477 and units["unit"].is_unique and (units["unit"].str.len() == 6).all()

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    df["unit"] = df["geo_id"].str[:6]
    census = df.groupby("unit")["count"].sum()       # persons with a language, T - N
    a, b = set(census.index), set(units["unit"])
    assert a == b, (sorted(a - b)[:5], sorted(b - a)[:5])
    print(f"join: {len(a)} gminas in the table and in the boundaries, none left over either way")

    # OVERLAPS. At GISCO's 1:1M a town's rural gmina (the "doughnut", gm. w. Elbląg round gm. m.
    # Elbląg) often has no hole: both polygons hold the town, and the first match took the town's
    # hexes for the ring (Elbląg city 0.13 of its census in Kontur, the ring 16x). Each polygon
    # is cut by every smaller polygon it overlaps, so a point belongs to the smallest unit
    # holding it. religiondots draws on the uncut polygons; this cut is languagedots' own.
    units = units.to_crs(2180)
    units["area"] = units.geometry.area
    units = units.sort_values("area").reset_index(drop=True)
    sidx = units.sindex
    cut, n_cut, lost = [], 0, 0.0
    for i, g in enumerate(units.geometry):
        smaller = [j for j in sidx.query(g, predicate="intersects") if j < i]
        if smaller:
            ov = units.geometry.iloc[smaller].union_all()
            if g.intersection(ov).area > 1e5:          # more than 0.1 km² shared
                n_cut += 1
                lost += g.intersection(ov).area
                g = g.difference(ov)
        cut.append(g)
    units = gpd.GeoDataFrame(units[["unit"]], geometry=cut, crs=2180)
    print(f"overlaps: {n_cut} gminas cut by smaller ones they overlapped, "
          f"{lost / 1e6:,.0f} km² removed in all")

    layer = hex_layer("pl", units[["unit", "geometry"]], census=census.round().astype(int).to_dict(),
                      out=OUT)
    have = set(layer.loc[layer["pop"] > 0, "unit"])
    empty = units[~units["unit"].isin(have)]
    if len(empty):
        extra = gpd.GeoDataFrame({"unit": empty["unit"], "pop": 0.0}, geometry=empty.geometry,
                                 crs=units.crs).to_crs(layer.crs)
        layer = pd.concat([layer, extra], ignore_index=True)
        layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"{len(empty)} gminas with no populated hex given their own polygon at pop 0")
    print(f"placement layer: {len(layer):,} features, {layer['unit'].nunique():,} gminas")


if __name__ == "__main__":
    main()
