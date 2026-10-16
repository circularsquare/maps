"""T-456: OSM places against ONS UK Business Counts (local units) by English local authority."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:\Users\anita\projects\maps\trainworld\data\raw"
lad = gpd.read_file(os.path.join(RAW, "tier0", "lad_dec2023_uk_bgc.geojson"))
lad = lad[lad.LAD23CD.str.startswith("E")].to_crs("EPSG:4326")
lad["km2"] = lad.to_crs("EPSG:27700").area / 1e6
osm = pd.read_csv(os.path.join(RAW, "t456", "pois_europe_united-kingdom_england.csv.gz"))
op = gpd.GeoDataFrame(osm, geometry=gpd.points_from_xy(osm.lon, osm.lat), crs="EPSG:4326")
j = gpd.sjoin(op, lad[["LAD23CD", "geometry"]], how="inner", predicate="within")
j = j[~j.index.duplicated()]
oc = j.groupby(["LAD23CD", "cls"]).size().unstack(fill_value=0).add_prefix("o_")
bc = pd.read_csv(os.path.join(RAW, "t456", "ukbc_lad.csv"))
bc["k"] = bc.INDUSTRY_NAME.str.split(" ").str[0].map({"Total": "all", "45": "motor", "47": "ret", "55": "acc", "56": "food", "95": "repair", "96": "pers"})
bc = bc.pivot_table(index="GEOGRAPHY_CODE", columns="k", values="OBS_VALUE", aggfunc="sum").add_prefix("e_")
pop = pd.read_csv(os.path.join(RAW, "tier0", "census2021_ts001_lad_england.csv")).set_index("GEOGRAPHY_CODE").OBS_VALUE.rename("pop")
df = lad[["LAD23CD", "LAD23NM", "km2"]].set_index("LAD23CD").join(oc).join(bc).join(pop)
itl = gpd.read_file(os.path.join(RAW, "tier0", "itl3_jan2025_uk_bgc.geojson")).to_crs("EPSG:4326")
icol = [c for c in itl.columns if c.upper().startswith("ITL") and c.upper().endswith("CD")][0]
cen = gpd.GeoDataFrame(lad[["LAD23CD"]], geometry=lad.representative_point(), crs="EPSG:4326")
k = gpd.sjoin(cen, itl[[icol, "geometry"]], how="left", predicate="within").drop_duplicates("LAD23CD")
k["RGN"] = k[icol].str[:3]
k[["LAD23CD", "RGN"]].to_csv(os.path.join(HERE, "lad_region.csv"), index=False)
print("LADs", len(df), "no business counts", int(df.e_all.isna().sum()), "no pop", int(df["pop"].isna().sum()))
df.to_csv(os.path.join(HERE, "uk_lad.csv"), encoding="utf-8")
print(df.sum(numeric_only=True))
