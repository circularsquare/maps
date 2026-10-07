"""Hungary placement layer: Kontur hexes keyed to religiondots' 3,177 settlements.

    python sources/hu_geo.py

-> data/geo/hu/hu_hexes.gpkg (unit = KSH settlement code, pop = Kontur people)

UNITS. religiondots' `data/geo/hu/hu_settlements.gpkg` (GISCO LAU 2021 for the settlements, and
Budapest's 23 districts from geoBoundaries ADM2 clipped to GISCO's Budapest;
religiondots/sources/hu_geo.md), read only. Its `kod` is KSH's five-digit settlement code, the
same code the census database uses for TERUL_GEO5, Budapest's districts included, so the join is
on the code itself; it is asserted both ways below.

WHY HEXES. Most settlements are villages, but the towns of the Great Plain (Hódmezővásárhely,
Kecskemét, Szeged) have huge outlying areas of farmland and tanyák, and the polygon would spread
a town's dots evenly over them. Kontur puts them where people live; each settlement's dot count
still comes from the census.

A UNIT WITH NO POPULATED HEX would draw nothing, so it gets its own polygon at pop 0 (placed on
equal shares); the count is printed.
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

RD_UNITS = HERE.parent / "religiondots" / "data" / "geo" / "hu" / "hu_settlements.gpkg"
NORM = HERE / "data" / "normalized" / "hu.csv"
OUT = HERE / "data" / "geo" / "hu" / "hu_hexes.gpkg"


def main():
    units = gpd.read_file(RD_UNITS)
    units["unit"] = units["kod"].astype(str)
    assert len(units) == 3177 and units["unit"].is_unique, len(units)
    assert units["unit"].str.fullmatch(r"\d{5}").all()

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    everyone = df.groupby("geo_id")["count"].sum()          # the whole population, answered or not
    a, b = set(everyone.index), set(units["unit"])
    assert a == b, (sorted(a - b)[:5], sorted(b - a)[:5])
    print(f"join: {len(a):,} settlements in the table = {len(b):,} polygons, both ways")

    layer = hex_layer("hu", units[["unit", "geometry"]],
                      census=everyone.round().astype(int).to_dict(), out=OUT)
    have = set(layer.loc[layer["pop"] > 0, "unit"])
    empty = units[~units["unit"].isin(have)]
    if len(empty):
        extra = gpd.GeoDataFrame({"unit": empty["unit"], "pop": 0.0}, geometry=empty.geometry,
                                 crs=units.crs).to_crs(layer.crs)
        layer = pd.concat([layer, extra], ignore_index=True)
        layer.to_file(OUT, layer="hexes", driver="GPKG")
        print(f"  their census people: {everyone.loc[sorted(empty['unit'])].sum():,.0f}; largest: "
              f"{everyone.loc[sorted(empty['unit'])].sort_values().tail(3).round().to_dict()}")
    print(f"{len(empty)} units with no populated hex given their own polygon at pop 0")
    print(f"placement layer: {len(layer):,} features, {layer['unit'].nunique():,} units")


if __name__ == "__main__":
    main()
