"""Switzerland: the placement layer. religiondots' 2,197 commune polygons plus Verzasca, on Kontur.

Writes:
    data/geo/ch/ch_units.gpkg    2,198 commune polygons, `unit` = BFS number (1.1.2020 state)
    data/geo/ch/ch_hexes.gpkg    religiondots' hexes + Verzasca's, `unit` and `pop`   (`place`)

Usage:
    python sources/ch_geo.py     (run again after sources/ch_vz2000.py for the census check)

WHY NOT religiondots' ch_grid_400m.gpkg AS IT STANDS. It is the right layer on the right units,
built from GISCO LAU 2021 (which is really the 1 January 2020 commune state) with the lakes and
comunanze dropped against BFS's register of 1.1.2020. That register predates the Verzasca merger
(18 October 2020), so the new commune Verzasca (BFS 5399), whose polygon IS in the LAU file, was
dropped with the lakes ("5399 holds 802 people" in religiondots/sources/ch_geo.py). The five 2000
communes that became it (Brione (Verzasca), Corippo, Frasco, Sonogno, Vogorno; 746 people in
2000) then have nowhere to go. So the units here are religiondots' 2,197 polygons, read
unchanged, plus LAU's CH5399; and the hexes are religiondots' 40,641, unchanged, plus Verzasca's
own Kontur hexes cut religiondots' way (representative point inside, then clipped to the
commune). Rebuilding the whole layer with sources/_grid.py was tried first and is worse here: by
plain centroid it leaves 176,700 people in lake-shore hexes outside every commune and six small
communes with no hex at all, which religiondots' builder had already solved (its tiny communes
carry their own polygon). Nothing is written into religiondots.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
RD = ROOT.parent / "religiondots"
RD_UNITS = RD / "data" / "geo" / "ch" / "ch_communes.gpkg"
RD_GRID = RD / "data" / "geo" / "ch" / "ch_grid_400m.gpkg"
LAU = RD / "data" / "geo" / "lau2021" / "shp4326" / "LAU_RG_01M_2021_4326.shp"
OUT_DIR = ROOT / "data" / "geo" / "ch"
UNITS_OUT = OUT_DIR / "ch_units.gpkg"
NORM = ROOT / "data" / "normalized" / "ch.csv"
EXPECTED_RD = 2_197
ADD = {"5399": "Verzasca"}


def units():
    import geopandas as gpd
    import pandas as pd

    rd = gpd.read_file(RD_UNITS)[["unit", "geo_name", "geometry"]]
    rd["unit"] = rd["unit"].astype(str)
    if len(rd) != EXPECTED_RD or rd["unit"].nunique() != EXPECTED_RD:
        raise SystemExit(f"religiondots' ch_communes.gpkg has {len(rd)} units, expected "
                         f"{EXPECTED_RD}")
    lau = gpd.read_file(LAU, columns=["CNTR_CODE", "LAU_ID", "LAU_NAME"])
    lau = lau[lau["CNTR_CODE"] == "CH"].copy()
    lau["unit"] = lau["LAU_ID"].astype(str).str.replace("CH", "").str.zfill(4)
    add = lau[lau["unit"].isin(ADD)].to_crs(rd.crs)
    if len(add) != len(ADD):
        raise SystemExit(f"LAU lacks {set(ADD) - set(add['unit'])}")
    if set(add["unit"]) & set(rd["unit"]):
        raise SystemExit("an added unit is already in religiondots' layer; nothing to add")
    add = add.assign(geo_name=add["unit"].map(ADD))[["unit", "geo_name", "geometry"]]
    # the added polygon must not overlap religiondots' communes (it was dropped, not merged)
    ov = gpd.overlay(add.to_crs(2056), rd.to_crs(2056), how="intersection")
    ov_km2 = ov.area.sum() / 1e6
    print(f"  Verzasca {add.to_crs(2056).area.sum() / 1e6:,.1f} km2; overlap with the other "
          f"communes {ov_km2:.3f} km2")
    if ov_km2 > 0.5:
        raise SystemExit("Verzasca overlaps religiondots' communes")
    u = gpd.GeoDataFrame(pd.concat([rd, add], ignore_index=True), crs=rd.crs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    u.to_file(UNITS_OUT, layer="units", driver="GPKG")
    print(f"  wrote {UNITS_OUT} ({len(u):,} units)")
    return u


def main():
    import geopandas as gpd
    import pandas as pd
    from _grid import kontur_path

    u = units()
    rd = gpd.read_file(RD_GRID)
    rd["unit"] = rd["unit"].astype(str)
    if rd["unit"].nunique() != EXPECTED_RD:
        raise SystemExit(f"religiondots' grid covers {rd['unit'].nunique()} units")
    add = u[u["unit"].isin(ADD)]

    # religiondots' hexes, unchanged, must not already cover the added commune
    inside = gpd.sjoin(gpd.GeoDataFrame(geometry=rd.geometry.representative_point(),
                                        crs=rd.crs), add.to_crs(rd.crs), predicate="within")
    if len(inside):
        raise SystemExit(f"{len(inside)} religiondots hexes already lie in {list(ADD)}")

    # the added commune's hexes, religiondots' way: representative point in, then clipped
    k = gpd.read_file(kontur_path("ch"))
    popcol = next(c for c in k.columns if c.lower() == "population")
    k = k[[popcol, "geometry"]].rename(columns={popcol: "pop"}).to_crs(rd.crs)
    pts = gpd.GeoDataFrame({"i": range(len(k))}, geometry=k.geometry.representative_point(),
                           crs=rd.crs)
    hit = gpd.sjoin(pts, add.to_crs(rd.crs)[["unit", "geometry"]], predicate="within")
    new = k.iloc[hit["i"].to_numpy()].copy()
    new["unit"] = hit["unit"].to_numpy()
    new = gpd.overlay(new, add.to_crs(rd.crs)[["unit", "geometry"]].rename(
        columns={"unit": "u2"}), how="intersection", keep_geom_type=True)
    new = new[["unit", "pop", "geometry"]]
    print(f"  added {len(new):,} Kontur hexes for {list(ADD.values())}, "
          f"{new['pop'].sum():,.0f} people (Kontur 2023)")
    if new.empty or new["pop"].sum() <= 0:
        raise SystemExit("the added commune got no populated hex")

    layer = gpd.GeoDataFrame(pd.concat([rd[["unit", "pop", "geometry"]], new],
                                       ignore_index=True), crs=rd.crs)
    have = set(layer["unit"])
    missing = set(u["unit"]) - have
    if missing:
        raise SystemExit(f"units without a placement polygon: {sorted(missing)[:10]}")
    out = OUT_DIR / "ch_hexes.gpkg"
    tmp = OUT_DIR / "ch_hexes.tmp.gpkg"
    if tmp.exists():
        tmp.unlink()
    layer.to_file(tmp, layer="hexes", driver="GPKG")
    os.replace(tmp, out)
    print(f"  wrote {out} ({len(layer):,} polygons, {layer['unit'].nunique():,} units, "
          f"{layer['pop'].sum():,.0f} people)")

    if NORM.exists():
        df = pd.read_csv(NORM, dtype={"geo_id": str})
        df = df[df["geo_level"] == "commune"]
        census = df.groupby("geo_id")["count"].sum()
        kont = layer.groupby("unit")["pop"].sum()
        nodata = sorted(set(census.index) - have)
        print(f"  census units without a polygon: {len(nodata)}")
        for unit in ADD:
            print(f"  {ADD[unit]}: census 2000 {census.get(unit, 0):,}, Kontur 2023 "
                  f"{kont.get(unit, 0):,.0f}")
        zero = sorted(set(kont.index) - set(census.index))
        print(f"  placement units the census puts nobody in: {zero}")


if __name__ == "__main__":
    main()
