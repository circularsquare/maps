"""Mexico: the placement layer, INEGI's 2020 AGEBs less the islands nobody lives on.

Writes data/geo/mx/mx_place.gpkg, which `countries/mx.py` places on. Record: `sources/mx.md`
"Islands nobody lives on" (2026-10-04); boundaries: `sources/mx_geo.md`.

**THE ISLANDS ARE THEIR OWN AGEBs, AND AN AGEB TAKES AN EQUAL SHARE.** Mexico places each
municipio's dots in equal shares across its AGEBs (spec §8.2), which is a population weight only
because INEGI builds AGEBs to a population target. Offshore islands break that: INEGI gives
nearly every island and cay a rural AGEB of its own, so 48 of Progreso's 113 AGEBs are the cays
of Arrecife Alacranes, 49 of San Quintín's 165 are Gulf of California and Pacific islands, 27 of
La Huerta's 68 are islets off the Jalisco coast, and 5 of Celestún's 16 are Arrecife Triángulos
and its neighbours. Each took as many dots as a mainland AGEB of a thousand people: 171 dots
(171,000 people) were drawn on islands where the 2020 census counts fewer than 250, 16 at
1:10,000. Smaller islets that are pieces of a mainland AGEB took an area share.

So: Mexico's land is split into its connected pieces (the union of `00ent`), every ITER 2020
locality is put on the piece it lies on (nearest within 2 km, for points just offshore), and
every piece other than the mainland holding fewer than `KEEP_ISLAND_POP` people is cut out of the
AGEBs. An AGEB with nothing left is dropped. A municipio's counts do not move (no unit loses its
last AGEB; asserted), only where inside it they are drawn.

`KEEP_ISLAND_POP = 250`, a quarter of a 1:1,000 dot. It drops Isla Guadalupe (113 people at a
fishing camp, ITER `Tepeyac (Campamento Weste)`), Tiburón (16), Isla Grande de Ixtapa (16), San
Martín (15), El Pardito (13), San Benito Oeste (8) and four smaller camps, 193 people on 10
islands, which an equal AGEB share would otherwise draw as a whole dot each; the other 333 cut
islands have nobody. It keeps Natividad (268) and San Marcos (367), which are pieces of a
mainland AGEB and take an area share. Result: 81,451 AGEBs -> 81,226 (225 dropped, 8 trimmed).

Usage:
    python sources/mx_geo.py
"""
import os
import re
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from geo_checks import read_layer  # noqa: E402

MG = os.path.join(ROOT, "data", "geo", "mx", "mg2020", "conjunto_de_datos")
ITER = os.path.join(ROOT, "data", "raw", "mx", "iter_00_cpv2020_csv.zip")
ITER_CSV = "iter_00_cpv2020/conjunto_de_datos/conjunto_de_datos_iter_00CSV20.csv"
OUT = os.path.join(ROOT, "data", "geo", "mx", "mx_place.gpkg")

KEEP_ISLAND_POP = 250        # people (ITER 2020) an island needs to keep its land
SNAP_M = 2000                # a locality point this close to land is on it
EXPECTED_UNITS = 2469
EXPECTED_DROPPED = 225       # whole AGEBs on cut islands, 2026-10-04
ITER_TOTAL = 126_014_024

# (name, lon, lat, kept?) The point must fall on, or within 3 km of, an island piece.
WITNESS = [
    ("Arrecife Alacranes", -89.67, 22.48, False),
    ("Arrecife Triangulos", -92.23, 20.91, False),
    ("Isla Socorro", -110.98, 18.79, False),
    ("Isla Clarion", -114.72, 18.36, False),
    ("Isla Guadalupe", -118.27, 29.04, False),
    ("Isla Angel de la Guarda", -113.31, 29.27, False),
    ("Isla Tiburon", -112.35, 28.99, False),
    ("Isla Maria Madre", -106.58, 21.62, False),
    ("Isla Espiritu Santo", -110.33, 24.47, False),
    ("Cozumel", -86.92, 20.43, True),
    ("Isla Mujeres", -86.73, 21.23, True),
    ("Holbox", -87.33, 21.55, True),
    ("Isla de Cedros", -115.21, 28.20, True),
    ("Isla Natividad", -115.18, 27.87, True),
    ("Isla San Marcos", -112.07, 27.22, True),
]
# Printed, not asserted: cays INEGI may draw as a speck or not at all.
LOOK = [("Cayo Arenas", -91.40, 22.12), ("Cayos Arcas", -91.97, 20.21)]


def dms(s):
    m = re.match(r"\s*(\d+)°(\d+)'([\d.]+)\"\s*([NSEW])", str(s))
    if not m:
        return np.nan
    v = int(m[1]) + int(m[2]) / 60 + float(m[3]) / 3600
    return -v if m[4] in "SW" else v


def land_pieces():
    ent = read_layer(os.path.join(MG, "00ent.shp"))
    parts = shapely.get_parts(shapely.union_all(ent.geometry.values))
    g = gpd.GeoDataFrame(geometry=parts, crs=ent.crs)
    g["area_km2"] = g.area / 1e6
    g = g.sort_values("area_km2", ascending=False).reset_index(drop=True)
    g["piece"] = np.arange(len(g))
    return g


def localities(crs):
    with zipfile.ZipFile(ITER) as z, z.open(ITER_CSV) as f:
        it = pd.read_csv(f, dtype=str, usecols=["ENTIDAD", "MUN", "LOC", "NOM_LOC",
                                                "LONGITUD", "LATITUD", "POBTOT"])
    it = it[(it.LOC != "0000") & ~it.LOC.isin(["9998", "9999"]) & it.LONGITUD.notna()].copy()
    it["pop"] = pd.to_numeric(it.POBTOT, errors="raise")
    assert int(it["pop"].sum()) == ITER_TOTAL, int(it["pop"].sum())
    x, y = it.LONGITUD.map(dms), it.LATITUD.map(dms)
    assert x.notna().all() and y.notna().all()
    return gpd.GeoDataFrame(it, geometry=gpd.points_from_xy(x, y), crs=4326).to_crs(crs)


def main():
    pieces = land_pieces()
    crs = pieces.crs
    loc = localities(crs)
    j = gpd.sjoin_nearest(loc[["pop", "NOM_LOC", "geometry"]], pieces[["piece", "geometry"]],
                          how="left", max_distance=SNAP_M)
    j = j[~j.index.duplicated()]
    off = j["piece"].isna()
    assert not off.any(), f"{int(off.sum())} localities more than {SNAP_M} m from land"
    pop = j.groupby("piece")["pop"].sum()
    names = j.groupby("piece")["NOM_LOC"].agg(lambda s: "; ".join(list(s)[:3]))
    pieces["pop"] = pieces["piece"].map(pop).fillna(0).astype(int)
    pieces["names"] = pieces["piece"].map(names).fillna("")
    islands = pieces.iloc[1:]
    drop = islands[islands["pop"] < KEEP_ISLAND_POP]
    print(f"{len(pieces):,} land pieces; mainland {pieces.area_km2.iloc[0]:,.0f} km2 holds "
          f"{pieces['pop'].iloc[0]:,} people")
    print(f"{len(islands):,} islands; {len(drop):,} under {KEEP_ISLAND_POP} people, "
          f"{drop.area_km2.sum():,.1f} km2, {int(drop['pop'].sum()):,} people, cut out")
    shown = drop[drop["pop"] > 0]
    for r in shown.itertuples():
        print(f"    cut, {r.pop:>4} people  {r.area_km2:8.2f} km2  {r.names}")

    # Each AGEB part goes to the land piece its representative point is on.
    a = read_layer(os.path.join(MG, "00a.shp"))
    a["geometry"] = shapely.make_valid(a.geometry.values)   # 22 self-intersections, mx_geo.md §6
    n0 = len(a)
    # Only AGEBs touching a cut island are rebuilt; every other keeps its geometry untouched.
    near = gpd.sjoin(a[["geometry"]], drop[["geometry"]], predicate="intersects").index.unique()
    ex = a.loc[near, ["geometry"]].explode(index_parts=False)
    ex = ex[ex.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(names="row")
    rp = gpd.GeoDataFrame({"part": ex.index}, geometry=ex.geometry.representative_point(),
                          crs=crs)
    hit = gpd.sjoin(rp, pieces[["piece", "geometry"]], how="left",
                    predicate="intersects").drop_duplicates("part").set_index("part")
    # (a part on no piece is a sliver between INEGI's AGEB and state lines; it stays)
    ex["gone"] = hit["piece"].reindex(ex.index).isin(set(drop["piece"])).to_numpy()
    touched = sorted(set(ex.loc[ex["gone"], "row"]))
    geom = a.geometry.values.copy()
    lost = np.zeros(len(a), dtype=bool)
    kept ={row: shapely.union_all(g.geometry.values)
            for row, g in ex[~ex["gone"]].groupby("row")}
    for row in touched:
        if row in kept:
            geom[row] = kept[row]
        else:
            lost[row] = True
    a["geometry"] = geom
    trimmed = len(touched) - int(lost.sum())
    dropped = a[lost]
    a = a[~lost].reset_index(drop=True)
    print(f"AGEBs: {n0:,} -> {len(a):,}; {int(lost.sum())} dropped whole (island only), "
          f"{trimmed} trimmed of island pieces")
    print("    dropped:", ", ".join(dropped["CVEGEO"].tolist()))

    assert int(lost.sum()) == EXPECTED_DROPPED, f"{int(lost.sum())} dropped; was {EXPECTED_DROPPED}"
    unit = a["CVE_ENT"].astype(str) + a["CVE_MUN"].astype(str)
    assert unit.nunique() == EXPECTED_UNITS, unit.nunique()

    # Witnesses: named islands land on the expected side.
    for name, lon, lat, kept in WITNESS:
        pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(crs)
        d = pieces.distance(pt.iloc[0])
        k = int(d.idxmin())
        assert d[k] < 3000 and k != 0, (name, float(d[k]), k)
        is_kept = k not in set(drop["piece"])
        print(f"    witness {name}: piece {k}, {pieces.area_km2[k]:.2f} km2, "
              f"{pieces['pop'][k]} people, {'kept' if is_kept else 'cut'}")
        assert is_kept == kept, name
    for name, lon, lat in LOOK:
        pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(crs)
        d = pieces.distance(pt.iloc[0])
        k = int(d.idxmin())
        print(f"    look {name}: nearest piece {k} at {d[k] / 1000:.1f} km, "
              f"{pieces.area_km2[k]:.3f} km2, {pieces['pop'][k]} people, "
              f"{'kept' if k not in set(drop['piece']) else 'cut'}")

    a.to_file(OUT, driver="GPKG")
    print(f"wrote {len(a):,} AGEBs -> {OUT}")


if __name__ == "__main__":
    main()
