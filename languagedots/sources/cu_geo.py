"""Cuba: re-key religiondots' province hexes to its 168 municipalities, and write the municipal
population as the count -> data/geo/cu/cu_hexes.gpkg, data/normalized/cu.csv.

    python sources/cu_geo.py

WHY MUNICIPALITIES. Kontur 2023 is badly off in eastern Cuba (religiondots' cu_grid.py: Granma's
raw Kontur is 0.17 of its share of ONEI's count, Santiago and Guantanamo about 0.4). Drawing on
municipal counts pins each municipality's people to it, and only the placement inside a
municipality is Kontur's. religiondots' layers (cu_hexes.gpkg, cu_municipalities.gpkg/.csv) are
read only; the re-keyed copy is languagedots' own.

THE COUNT. ONEI's 2023 municipal populations as religiondots carries them (cu_municipalities.csv
`pop2023`). No Cuban census asks language; everyone is drawn as Spanish (`derived`).

CHECKS: 168 municipalities in both the csv and the polygons, joined 1:1 on adm2_pcode; every hex
assigned (centroid in polygon, else nearest polygon within the same province); every
municipality gets at least one hex; each hex's municipality lies in the hex's own province.
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
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

SRC = RD_GEO / "cu"
OUT_GEO = HERE / "data" / "geo" / "cu" / "cu_hexes.gpkg"
OUT_CSV = HERE / "data" / "normalized" / "cu.csv"


def main():
    mc = pd.read_csv(SRC / "cu_municipalities.csv", dtype=str)
    mc["pop2023"] = mc["pop2023"].astype(int)
    mg = gpd.read_file(SRC / "cu_municipalities.gpkg")
    print("municipality layer columns:", list(mg.columns))
    key = "adm2_pcode" if "adm2_pcode" in mg.columns else "ADM2_PCODE"
    mg = mg.rename(columns={key: "adm2_pcode"})[["adm2_pcode", "geometry"]]
    assert len(mc) == 168 and mg["adm2_pcode"].nunique() == 168
    assert set(mc["adm2_pcode"]) == set(mg["adm2_pcode"])
    mg = mg.merge(mc[["adm2_pcode", "adm1_pcode"]], on="adm2_pcode", validate="1:1")

    hx = gpd.read_file(SRC / "cu_hexes.gpkg").to_crs(4326)
    mg = mg.to_crs(4326)
    cen = hx[["unit"]].copy()
    cen = gpd.GeoDataFrame(cen, geometry=hx.geometry.representative_point(), crs=4326)
    hit = gpd.sjoin(cen, mg, how="left", predicate="within")
    hit = hit[~hit.index.duplicated()]
    out = hit["adm2_pcode"].copy()
    miss = out.isna() | (hit["adm1_pcode"] != hit["unit"])
    print(f"hexes: {len(hx):,}; centroid outside a municipality of its own province: "
          f"{int(miss.sum()):,}")
    if miss.any():
        m3 = mg.to_crs(3857)
        c3 = cen.loc[miss].to_crs(3857)
        for i, row in c3.iterrows():
            cand = m3[m3["adm1_pcode"] == hx.at[i, "unit"]]
            out.at[i] = cand.loc[cand.distance(row.geometry).idxmin(), "adm2_pcode"]
    hx["prov"] = hx["unit"]
    hx["unit"] = out.values
    assert hx["unit"].notna().all()
    lost = set(mc["adm2_pcode"]) - set(hx["unit"])
    assert not lost, f"municipalities with no hex: {sorted(lost)}"
    prov_of = dict(zip(mc["adm2_pcode"], mc["adm1_pcode"]))
    assert (hx["unit"].map(prov_of) == hx["prov"]).all()
    OUT_GEO.parent.mkdir(parents=True, exist_ok=True)
    hx.to_file(OUT_GEO, driver="GPKG")
    print(f"wrote {OUT_GEO}: {len(hx):,} hexes over {hx['unit'].nunique()} municipalities")

    df = pd.DataFrame(dict(geo_id=mc["adm2_pcode"], geo_level="municipio", geo_name=mc["muni"],
                           source_category="Español", count=mc["pop2023"], tier="derived",
                           source_id="onei_pop_municipal_2023", year=2023,
                           note="no language question; everyone drawn as Spanish"))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"wrote {OUT_CSV}: {len(df)} municipalities, {df['count'].sum():,} people")


if __name__ == "__main__":
    main()
