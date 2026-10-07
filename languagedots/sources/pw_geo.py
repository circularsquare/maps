"""Palau placement layer: religiondots' Kontur hexes for PW (175, read only), re-keyed to the 16
states -> data/geo/pw/pw_hexes.gpkg (unit = ISO 3166-2 code, as in data/normalized/pw.csv).

    python sources/pw_geo.py [--fetch]

States: geoBoundaries gbOpen PLW ADM1 (OpenStreetMap, ODbL, 16 units, shapeISO PW-002 ...),
commit 9469f09. Each hex goes to the state its centroid falls in; a hex whose centroid falls in
no state polygon (OSM's state outlines follow the land and reef loosely, so coastal and
small-island hexes miss) goes to the NEAREST state, and every such hex is printed with its
distance. The 2015 census populations are printed beside Kontur's per state; Kontur is poor in
Palau (it puts more people on Angaur than the census does), but only placement inside a state is
borrowed from it.
"""
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import geopandas as gpd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "pw"
ADM = RAW / "geoBoundaries-PLW-ADM1.geojson"
ADM_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/PLW/ADM1/"
           "geoBoundaries-PLW-ADM1.geojson")
RD_HEX = HERE.parent / "religiondots" / "data" / "geo" / "pw" / "pw_hexes.gpkg"
OUT = HERE / "data" / "geo" / "pw" / "pw_hexes.gpkg"
CRS_M = 32653  # UTM 53N


def main():
    if "--fetch" in sys.argv and not ADM.exists():
        req = urllib.request.Request(ADM_URL, headers={"User-Agent": "Mozilla/5.0"})
        ADM.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    from pw_census import COLS, T, NAMES
    units = gpd.read_file(ADM).rename(columns={"shapeISO": "unit"})[["unit", "shapeName", "geometry"]]
    census = {c: T["Total"][i] for i, c in enumerate(COLS) if c.startswith("PW-")}
    assert len(units) == 16 and set(units["unit"]) == set(census), "state ids do not match"
    for u, n in zip(units["unit"], units["shapeName"]):
        assert NAMES[u] == n, (u, n, NAMES[u])
    print("  16 states, ids and names match the census table both ways")

    hexes = gpd.read_file(RD_HEX)
    assert len(hexes) == 175, len(hexes)
    um = units.to_crs(CRS_M)
    pts = gpd.GeoDataFrame({"i": range(len(hexes))},
                           geometry=hexes.to_crs(CRS_M).geometry.centroid, crs=CRS_M)
    j = gpd.sjoin(pts, um[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    unit = j["unit"].copy()
    miss = unit.isna()
    if miss.any():
        nn = gpd.sjoin_nearest(pts[miss], um[["unit", "geometry"]], how="left", distance_col="d")
        nn = nn[~nn.index.duplicated(keep="first")]
        unit[miss] = nn["unit"]
        print(f"  {int(miss.sum())} hexes ({hexes.loc[miss, 'pop'].sum():,.0f} people) outside every "
              "state polygon, given to the nearest:")
        for u, g in nn.groupby("unit"):
            print(f"    {NAMES[u]:13s} {len(g):3d} hexes, "
                  f"{hexes.loc[g.index, 'pop'].sum():6,.0f} people, max {g['d'].max() / 1000:.1f} km")
    layer = gpd.GeoDataFrame({"unit": unit.astype(str).to_numpy(), "pop": hexes["pop"].to_numpy()},
                             geometry=hexes.geometry.to_numpy(), crs=hexes.crs)
    per = layer.groupby("unit")["pop"].sum()
    kt, ct = per.sum(), sum(census.values())
    print(f"  {'state':13s} {'census':>7s} {'kontur':>7s}  ratio (normalised)")
    for u in sorted(census, key=lambda u: -census[u]):
        k = per.get(u, 0)
        print(f"  {NAMES[u]:13s} {census[u]:7,d} {k:7,.0f}  {(k / kt) / (census[u] / ct):.2f}")
    empty = [NAMES[u] for u in census if per.get(u, 0) <= 0]
    assert not empty, f"states with no populated hex: {empty}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    layer.to_file(OUT, layer="hexes", driver="GPKG")
    print(f"  wrote {OUT} ({len(layer)} hexes)")


if __name__ == "__main__":
    main()
