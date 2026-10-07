"""Sum Kontur Population (AF, 2023-11-01, 400 m H3 hexes) into each COD district
by hex centroid. Hexes whose centroid falls outside every district (border
slivers) go to the nearest district if within 2 km, else are dropped.

Writes helper1m/data/afghanistan/kontur_adm2.csv (code, kontur). Used by
fetch.py to split Kaldar against Sharak-e-Hayratan, and by check.py.
Source is religiondots' copy of the gpkg (read only).
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import geopandas as gpd
import pandas as pd

HERE = Path(__file__).parent
HELPER = HERE.parents[1]
REPO = HELPER.parent
KONTUR = REPO / "religiondots" / "data" / "raw" / "af" / "kontur_population_AF_20231101.gpkg"
ADM2 = HELPER / "data" / "afghanistan" / "boundaries" / "adm2.gpkg"
OUT = HELPER / "data" / "afghanistan" / "kontur_adm2.csv"


def main():
    k = gpd.read_file(KONTUR)
    d = gpd.read_file(ADM2)[["code", "geometry"]]
    k = k.to_crs("EPSG:32642")
    d = d.to_crs("EPSG:32642")
    pts = gpd.GeoDataFrame({"population": k["population"]}, geometry=k.geometry.centroid, crs=k.crs)
    j = gpd.sjoin(pts, d, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out = j[j["code"].isna()].drop(columns=["code", "index_right"])
    near = gpd.sjoin_nearest(out, d, how="left", max_distance=2000)
    near = near[~near.index.duplicated()]
    dropped = near["code"].isna()
    print(f"{len(k):,} hexes, {k['population'].sum():,.0f} people; "
          f"{len(out)} centroids outside every district, {int((~dropped).sum())} snapped, "
          f"{int(dropped.sum())} dropped ({near.loc[dropped, 'population'].sum():,.0f} people)")
    s = pd.concat([j.dropna(subset=["code"])[["code", "population"]],
                   near.dropna(subset=["code"])[["code", "population"]]])
    tot = s.groupby("code")["population"].sum().round().astype(int)
    tot = tot.reindex(d["code"], fill_value=0)
    tot.rename("kontur").to_csv(OUT, index_label="code")
    print(f"wrote {len(tot)} districts, {tot.sum():,} people -> {OUT}")


if __name__ == "__main__":
    main()
