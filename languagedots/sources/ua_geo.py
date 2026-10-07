"""Ukraine placement layer: the census's 2001 raions and cities, with Kontur hexes.

    python sources/ua_geo.py        (after sources/ua_c01.py)

UNITS. The U.S. Census Bureau's `UA_GEOG_ADM2_2001_uscb_201905` (data/raw/ua/ukraine.gdb.zip, the
same HDX release as the counts, CC BY): 672 polygons, the raions and cities of oblast significance
of 2001 plus Kyiv's 10 districts and one polygon for Sevastopol, keyed by `GEO_MATCH`, which is
what sources/ua_c01.py writes as `geo_id`. The join is an identity; asserted both ways.

Why not religiondots' layer: religiondots draws Ukraine at its 27 oblasts on COD-AB's 2025 lines.
The census table is per 2001 raion, and the 2020 reform merged raions into 136 new ones, so the
2001 boundaries are what the table was published on (playbook: "a boundary file of the wrong
vintage drops units without an error"). Kontur is read in place from religiondots' copy.

PLACEMENT. Kontur 2023-11-01 (pre-invasion inputs) hexes go to the 2001 unit holding their
centroid (sources/_grid.py). The checks it prints: people outside every unit, units with no
populated hex, Kontur against the 2001 census per unit and a shuffled-join control.

CRIMEA AND SEVASTOPOL are drawn from Russia's 2021 census (Anita, 2026-10-05; sources/ua.md), so
the 26 Crimean units (UKR_01_*, the Autonomous Republic's raions and cities, and UKR_02_01,
Sevastopol) are taken out of Ukraine's layer. The hexes are still assigned over all 672 units first,
so the centroid split at Perekop and Chonhar and the coast snap are the same as before, and then
the layer is cut in two: data/geo/ua/ua_hexes.gpkg holds the 646 mainland units, and
data/geo/ua/crimea_hexes.gpkg the Crimean units' hexes, which sources/ru_geo.py re-keys to
Russia's two subjects. Every hex is in exactly one of the two files (asserted).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
GDB = "zip://" + str(ROOT / "data" / "raw" / "ua" / "ukraine.gdb.zip").replace("\\", "/") + "!Ukraine.gdb"
LAYER = "UA_GEOG_ADM2_2001_uscb_201905"
N_UNITS = 672


def main():
    import pandas as pd
    import pyogrio
    from _grid import hex_layer

    units = pyogrio.read_dataframe(GDB, layer=LAYER)
    if len(units) != N_UNITS or units["GEO_MATCH"].duplicated().any():
        raise SystemExit(f"{len(units)} ADM2 polygons (want {N_UNITS}), "
                         f"{int(units['GEO_MATCH'].duplicated().sum())} duplicate ids")
    empty = units.geometry.is_empty | units.geometry.isna()
    if empty.any():
        raise SystemExit(f"empty geometry: {list(units.loc[empty, 'GEO_MATCH'])}")
    bad = ~units.geometry.is_valid
    if bad.any():
        print(f"  {int(bad.sum())} invalid polygons, repaired with make_valid")
        units.loc[bad, "geometry"] = units.loc[bad, "geometry"].make_valid()
    units = units.rename(columns={"GEO_MATCH": "unit"})[["unit", "AREA_NAME", "ADM1_NAME", "geometry"]]
    by_id = set(units.loc[units["unit"].map(is_crimean), "unit"])
    by_name = set(units.loc[units["ADM1_NAME"].isin(["AVTONOMNA RESPUBLIKA KRYM", "MISTO SEVASTOPOL’"]), "unit"])
    if by_id != by_name or len(by_id) != 26:
        raise SystemExit(f"Crimean units by id {len(by_id)} vs by region name {len(by_name)}")

    counts = pd.read_csv(ROOT / "data" / "normalized" / "ua.csv")
    pop = counts.groupby("geo_id")["count"].sum()
    a, b = set(units["unit"]), set(pop.index)
    if a != b:
        raise SystemExit(f"join: {len(b - a)} counted units with no polygon {sorted(b - a)[:5]}, "
                         f"{len(a - b)} polygons with no count {sorted(a - b)[:5]}")
    print(f"  join: {len(a)} units, every counted unit has its polygon and every polygon a count")

    area = units.to_crs(6933).area / 1e6
    print(f"  unit area: median {area.median():,.0f} km2 (about {area.median() / 0.74:,.0f} Kontur hexes), "
          f"smallest {area.min():,.1f} km2 ({units.loc[area.idxmin(), 'AREA_NAME']})")
    layer = hex_layer("ua", units, census={u: int(p) for u, p in pop.items()})
    snap_coast(units, layer)


SNAP_M = 2000


def snap_coast(units, layer):
    """USCB's lines are EuroGlobalMap's, generalised: 1,670 Kontur hexes (126,337 people) have
    centroids just outside every unit, 99% of them within 1 km (Sevastopol, Berdiansk, Yalta and
    Odesa's shores, the Danube, the Tisza). The Kontur UA extract holds only Ukraine, so these are
    Ukrainian hexes: snap each within SNAP_M to its nearest unit, in a metric CRS (playbook: hex
    centroids fall just offshore). The few beyond (Tuzla spit and the like, ~600 people) are dropped."""
    import geopandas as gpd
    import pandas as pd
    from _grid import kontur_path

    hexes = gpd.read_file(kontur_path("ua"))
    pts = gpd.GeoDataFrame({"pop": hexes["population"].to_numpy(dtype=float)},
                           geometry=hexes.geometry.centroid, crs=hexes.crs)
    um = units[["unit", "geometry"]].to_crs(hexes.crs)
    inside = gpd.sjoin(pts, um, how="left", predicate="within")
    inside = inside[~inside.index.duplicated()]
    out = pts[inside["unit"].isna().to_numpy()]
    near = gpd.sjoin_nearest(out, um, distance_col="d", max_distance=SNAP_M)
    near = near[~near.index.duplicated()]
    snapped = gpd.GeoDataFrame({"unit": near["unit"].astype(str).to_numpy(), "pop": near["pop"].to_numpy()},
                               geometry=hexes.geometry.loc[near.index].to_numpy(), crs=hexes.crs).to_crs(4326)
    print(f"  snapped {len(snapped):,} outside hexes ({snapped['pop'].sum():,.0f} people) within "
          f"{SNAP_M} m; dropped {len(out) - len(snapped):,} ({out['pop'].sum() - snapped['pop'].sum():,.0f} people)")
    full = gpd.GeoDataFrame(pd.concat([layer, snapped], ignore_index=True), crs=4326)
    crim = full["unit"].map(is_crimean)
    main_, cr = full[~crim], full[crim]
    # every hex in exactly one of the two layers
    key = full.geometry.to_wkb()
    if key.duplicated().any():
        raise SystemExit(f"{int(key.duplicated().sum())} hexes placed twice in Ukraine's layer")
    if len(main_) + len(cr) != len(full):
        raise SystemExit("split lost hexes")
    path = ROOT / "data" / "geo" / "ua" / "ua_hexes.gpkg"
    main_.to_file(path, layer="hexes", driver="GPKG")
    print(f"  rewrote {path} ({len(main_):,} hexes, {main_['unit'].nunique()} units, "
          f"{main_['pop'].sum():,.0f} people)")
    cpath = ROOT / "data" / "geo" / "ua" / "crimea_hexes.gpkg"
    cr.to_file(cpath, layer="hexes", driver="GPKG")
    print(f"  wrote {cpath} ({len(cr):,} hexes, {cr['unit'].nunique()} Crimean units, "
          f"{cr['pop'].sum():,.0f} people), for sources/ru_geo.py")


def is_crimean(unit):
    """The Autonomous Republic of Crimea's 25 raions and cities, and Sevastopol."""
    return unit.startswith("UKR_01_") or unit == "UKR_02_01"


if __name__ == "__main__":
    main()
