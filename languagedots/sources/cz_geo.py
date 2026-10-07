"""Czechia placement layer: Kontur hexes keyed to religiondots' finest cover (6,250 obce + 142
city districts).

    python sources/cz_geo.py

-> data/geo/cz/cz_hexes.gpkg (unit = ČSÚ obec or city-district code, pop = Kontur people)

UNITS. religiondots' `data/geo/cz/cz_finest.gpkg` (ČSÚ's own generalised obce and městské části,
stamped with the census date; religiondots/sources/cz_geo.md), read only: every obec that is not
subdivided, plus the 142 city districts of the 8 statutory cities that are. The language table
(sources/cz_sldb.py) is on the same codes; the join is asserted both ways below. The 4 military
districts (vojenské újezdy) have polygons and no census rows, as in religiondots.

WHY HEXES AND NOT THE POLYGONS religiondots draws on. Most obce are villages of a few hundred, where
the polygon is fine, but 130 obce over 10,000 hold half the country, and those not subdivided
(Olomouc, České Budějovice, Hradec Králové, Zlín...) would spread their dots over their fields and
woods. Kontur's hexes place them where people live; each unit's dot count still comes from the
census.

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

RD_UNITS = HERE.parent / "religiondots" / "data" / "geo" / "cz" / "cz_finest.gpkg"
REPLACED = HERE.parent / "religiondots" / "data" / "geo" / "cz" / "cz_replaced.csv"
NORM = HERE / "data" / "normalized" / "cz.csv"
OUT = HERE / "data" / "geo" / "cz" / "cz_hexes.gpkg"
MILITARY = {"503941", "545422", "555177", "592935"}   # Libavá, Boletice, Hradiště, Březina


def main():
    units = gpd.read_file(RD_UNITS)
    units["unit"] = units["kod"].astype(str)
    assert len(units) == 6392 and units["unit"].is_unique, len(units)
    assert str(units.crs).endswith("5514"), units.crs

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    rep = set(pd.read_csv(REPLACED, dtype=str)["kod"])
    df = df[((df["geo_level"] == "municipality") & ~df["geo_id"].isin(rep))
            | (df["geo_level"] == "city_district")]
    df = df[df["source_category"] != "Nezjištěno"]
    census = df.groupby("geo_id")["count"].sum()            # people with a stated mother tongue
    a, b = set(census.index), set(units["unit"])
    assert a <= b, sorted(a - b)[:5]
    extra = b - a
    assert extra == MILITARY, sorted(extra)
    print(f"join: {len(a):,} units in the table, all in the boundaries; the {len(extra)} left over "
          "are the military districts, which have no census rows")
    units = units[units["unit"].isin(a)]

    layer = hex_layer("cz", units[["unit", "geometry"]], census=census.round().astype(int).to_dict(),
                      out=OUT)
    have = set(layer.loc[layer["pop"] > 0, "unit"])
    empty = units[~units["unit"].isin(have)]
    if len(empty):
        extra = gpd.GeoDataFrame({"unit": empty["unit"], "pop": 0.0}, geometry=empty.geometry,
                                 crs=units.crs).to_crs(layer.crs)
        layer = pd.concat([layer, extra], ignore_index=True)
        layer.to_file(OUT, layer="hexes", driver="GPKG")
        print(f"  their census people: {census.loc[sorted(empty['unit'])].sum():,.0f}; "
              f"largest: {census.loc[sorted(empty['unit'])].sort_values().tail(3).round().to_dict()}")
    print(f"{len(empty)} units with no populated hex given their own polygon at pop 0")
    print(f"placement layer: {len(layer):,} features, {layer['unit'].nunique():,} units")


if __name__ == "__main__":
    main()
