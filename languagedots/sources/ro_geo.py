"""Romania placement layer: Kontur hexes keyed to religiondots' 3,181 UATs.

    python sources/ro_geo.py

-> data/geo/ro/ro_hexes.gpkg (unit = SIRUTA code as religiondots' `kod`, pop = Kontur people)

UNITS. religiondots' `data/geo/ro/ro_uat.gpkg` (Eurostat GISCO LAU 2021, 1:1M, `kod` = SIRUTA) and
its `ro_uat_lookup.csv`, which resolves the census's "JUDEŢ|NAME" keys to SIRUTA
(religiondots/sources/ro_geo.md: folded names, 4 hyphen fixes, 4 by elimination within a county).
Both read only. The language table (sources/ro_census.py) is the same INS release's same 3,181
UATs under the same keys, so the lookup applies unchanged; the join is asserted both ways here and
in countries/ro.py.

WHY HEXES. religiondots spreads a UAT's dots evenly over its polygon; a commune of 75 km² is
mostly fields and forest around a few villages. Kontur's hexes put the dots where people live;
each UAT's dot count still comes from the census.

OVERLAPS. As for Poland (sources/pl_geo.py), each polygon is cut by every smaller polygon it
overlaps, so a hex in two generalised polygons goes to the smaller. A UAT with no populated hex
gets its own polygon at pop 0 (equal shares); the count is printed.
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

RD_RO = HERE.parent / "religiondots" / "data" / "geo" / "ro"
RD_UNITS = RD_RO / "ro_uat.gpkg"
RD_LOOKUP = RD_RO / "ro_uat_lookup.csv"
NORM = HERE / "data" / "normalized" / "ro.csv"
OUT = HERE / "data" / "geo" / "ro" / "ro_hexes.gpkg"
TOTAL = "POPULATIA REZIDENTA TOTAL"
CRS = 3844      # Stereo 70, Romania's national projection, metres


def census_by_unit():
    """The census resident total per UAT, keyed by SIRUTA. Asserts the lookup covers the table."""
    df = pd.read_csv(NORM, dtype={"geo_id": str})
    df = df[(df["geo_level"] == "uat") & (df["source_category"] == TOTAL)]
    lut = pd.read_csv(RD_LOOKUP, dtype=str)
    m = dict(zip(lut["geo_id"], lut["kod"]))
    df["unit"] = df["geo_id"].map(m)
    miss = df.loc[df["unit"].isna(), "geo_id"].tolist()
    assert not miss, f"{len(miss)} census UATs not in religiondots' lookup: {miss[:5]}"
    assert df["unit"].is_unique, "two census UATs on one SIRUTA code"
    return df.set_index("unit")["count"]


def main():
    census = census_by_unit()
    units = gpd.read_file(RD_UNITS)
    units["unit"] = units["kod"].astype(str)
    assert len(units) == 3181 and units["unit"].is_unique
    a, b = set(census.index), set(units["unit"])
    assert a == b, (sorted(a - b)[:5], sorted(b - a)[:5])
    print(f"join: {len(a):,} UATs in the table and in the boundaries, none left over either way; "
          f"{census.sum():,} people")

    units = units.to_crs(CRS)
    units["area"] = units.geometry.area
    print(f"UAT area: median {units['area'].median() / 1e6:,.1f} km², "
          f"{(units['area'].median() / 1e6) / 0.74:,.0f} Kontur hexes at the median")
    units = units.sort_values("area").reset_index(drop=True)
    sidx = units.sindex
    cut, n_cut, lost = [], 0, 0.0
    for i, g in enumerate(units.geometry):
        smaller = [j for j in sidx.query(g, predicate="intersects") if j < i]
        if smaller:
            ov = units.geometry.iloc[smaller].union_all()
            shared = g.intersection(ov).area
            if shared > 1e5:                         # more than 0.1 km² shared
                n_cut += 1
                lost += shared
                g = g.difference(ov)
        cut.append(g)
    units = gpd.GeoDataFrame(units[["unit"]], geometry=cut, crs=CRS)
    print(f"overlaps: {n_cut} UATs cut by smaller ones they overlapped, "
          f"{lost / 1e6:,.1f} km² removed in all")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    layer = hex_layer("ro", units[["unit", "geometry"]], census=census.astype(int).to_dict(),
                      out=OUT)
    have = set(layer.loc[layer["pop"] > 0, "unit"])
    empty = units[~units["unit"].isin(have)]
    if len(empty):
        extra = gpd.GeoDataFrame({"unit": empty["unit"], "pop": 0.0}, geometry=empty.geometry,
                                 crs=units.crs).to_crs(layer.crs)
        layer = pd.concat([layer, extra], ignore_index=True)
        layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"{len(empty)} UATs with no populated hex given their own polygon at pop 0")
    print(f"placement layer: {len(layer):,} features, {layer['unit'].nunique():,} UATs")
    assert layer["unit"].nunique() == 3181

    # Bucharest, the one UAT holding 9% of the country: Kontur should agree with the census
    buc = census.idxmax()
    share_c = census[buc] / census.sum()
    share_k = layer.loc[layer["unit"] == buc, "pop"].sum() / layer["pop"].sum()
    print(f"largest UAT {buc}: census share {share_c:.4f}, Kontur share {share_k:.4f}")


if __name__ == "__main__":
    main()
