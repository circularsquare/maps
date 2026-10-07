"""Latvia placement layer: Kontur hexes keyed to religiondots' 119 LAUs (the municipalities of 2011).

    python sources/lv_geo.py

-> data/geo/lv/lv_hexes.gpkg (unit = seven-digit ATVK code, pop = Kontur people)

UNITS. religiondots' `data/geo/lv/lv_lau.gpkg` (GISCO LAU 2021, dated 1 January 2021, so still
the 119 municipalities from before the July 2021 reform: 9 republican cities and 110 novadi;
religiondots/sources/lv_geo.py), read only. Its `lau` is GISCO's LAU_ID, which for Latvia is the
ATVK code; the census's territorial code is "LV" + the same seven digits. The 2011 census and the
2021 LAUs have the same 119 municipalities (the 2009 reform's map lasted until 2021), which the
join below asserts both ways, and by name as well, since a set of codes can match while pairing
the wrong twins.

WHY HEXES. Religiondots places Latvia on the LAU polygons with LAU population; that spreads a
novads's dots evenly over its forests and bogs. Kontur puts them where people live; each
municipality's dot count still comes from the census.

A UNIT WITH NO POPULATED HEX would draw nothing, so it gets its own polygon at pop 0 (placed on
equal shares); the count is printed.
"""
import os
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
from _grid import hex_layer  # noqa: E402

RD_UNITS = HERE.parent / "religiondots" / "data" / "geo" / "lv" / "lv_lau.gpkg"
NORM = HERE / "data" / "normalized" / "lv.csv"
OUT = HERE / "data" / "geo" / "lv" / "lv_hexes.gpkg"


def _key(name):
    """'Rēzekne city' / 'Rēzeknes novads' / 'Rēzekne' -> 'rezekne', for the name check only."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    for suffix in (" city", " county", " novads", " pilseta", "s novads"):
        s = s.replace(suffix, "")
    return s.strip()


def main():
    units = gpd.read_file(RD_UNITS)
    units["unit"] = units["lau"].astype(str).str.zfill(7)
    assert len(units) == 119 and units["unit"].is_unique, len(units)
    assert units["unit"].str.fullmatch(r"\d{7}").all()

    df = pd.read_csv(NORM, dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"]
    pop = df[df["source_category"] == "Population"].set_index("geo_id")["count"]
    pop.index = pop.index.str.removeprefix("LV")
    a, b = set(pop.index), set(units["unit"])
    assert a == b, (sorted(a - b)[:5], sorted(b - a)[:5])
    print(f"join: {len(a)} municipalities in the table = {len(b)} polygons, both ways")

    # the same code must also be the same place: compare names, and print the few that differ
    cname = df.drop_duplicates("geo_id").set_index("geo_id")["geo_name"]
    cname.index = cname.index.str.removeprefix("LV")
    differ = []
    for _, r in units.iterrows():
        x, y = _key(cname[r["unit"]]), _key(r["name"])
        if not (x == y or y.startswith(x) or x.startswith(y[:-1])):
            differ.append((r["unit"], cname[r["unit"]], r["name"]))
    for d in differ:
        print(f"  name differs: {d}")
    assert len(differ) <= 3, f"{len(differ)} code joins disagree on name"
    print(f"names agree on {119 - len(differ)} of 119 codes")

    # GISCO's 2021 population against the 2011 census, per municipality: a slow drift is
    # expected (Latvia lost 9% in ten years), a wrong pairing is not
    ratio = (units.set_index("unit")["pop"] / pop).sort_values()
    print(f"GISCO 2021 / census 2011 per municipality: min {ratio.iloc[0]:.2f} "
          f"({ratio.index[0]}), median {ratio.median():.2f}, max {ratio.iloc[-1]:.2f} "
          f"({ratio.index[-1]})")
    # GISCO'S WORKBOOK SWAPS TWO CITIES' POPULATIONS: Jelgava (0090000) carries 21,629 and
    # Jēkabpils (0110000) 50,248, the other way round from every census (2011: 59,511 and
    # 24,635). Only the `pop`
    # column is affected (religiondots weights by it; this layer does not, it weights by Kontur),
    # so the check skips the pair after asserting that they are each other's figure.
    jj = units.set_index("unit")["pop"]
    swap = ("0090000", "0110000")
    assert abs(jj[swap[0]] / pop[swap[1]] - 1) < 0.25 and abs(jj[swap[1]] / pop[swap[0]] - 1) < 0.25, \
        "Jelgava and Jēkabpils are no longer simply swapped in GISCO's workbook"
    rest = ratio.drop(list(swap))
    print(f"  without the Jelgava/Jēkabpils swap: min {rest.iloc[0]:.2f} ({rest.index[0]}), "
          f"max {rest.iloc[-1]:.2f} ({rest.index[-1]})")
    assert 0.6 < rest.iloc[0] and rest.iloc[-1] < 1.6, "a municipality's population moved too far"

    layer = hex_layer("lv", units[["unit", "geometry"]],
                      census=pop.round().astype(int).to_dict(), out=OUT)
    have = set(layer.loc[layer["pop"] > 0, "unit"])
    empty = units[~units["unit"].isin(have)]
    if len(empty):
        extra = gpd.GeoDataFrame({"unit": empty["unit"], "pop": 0.0}, geometry=empty.geometry,
                                 crs=units.crs).to_crs(layer.crs)
        layer = pd.concat([layer, extra], ignore_index=True)
        layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"{len(empty)} units with no populated hex given their own polygon at pop 0")
    print(f"placement layer: {len(layer):,} features, {layer['unit'].nunique():,} units")


if __name__ == "__main__":
    main()
