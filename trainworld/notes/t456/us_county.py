"""T-456: OSM places against County Business Patterns 2022 (establishments) by county, Northeast."""
import os, sys, glob, zipfile
os.environ["OMP_NUM_THREADS"] = "2"
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:\Users\anita\projects\maps\trainworld\data\raw"
ST = {"09", "10", "11", "24", "25", "34", "36", "42", "44"}
co = gpd.read_file("zip://" + os.path.join(RAW, "cb_2020_us_county_500k.zip"))
co = co[co.STATEFP.isin(ST)].to_crs("EPSG:4326")
co["fips"] = co.STATEFP + co.COUNTYFP
co["km2"] = co.to_crs("EPSG:5070").area / 1e6
osm = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RAW, "t456", "us", "pois_*.csv.gz"))], ignore_index=True)
osm = osm.drop_duplicates(["cls", "lat", "lon"])
op = gpd.GeoDataFrame(osm, geometry=gpd.points_from_xy(osm.lon, osm.lat), crs="EPSG:4326")
j = gpd.sjoin(op, co[["fips", "geometry"]], how="inner", predicate="within")
j = j[~j.index.duplicated()]
oc = j.groupby(["fips", "cls"]).size().unstack(fill_value=0).add_prefix("o_")

with zipfile.ZipFile(os.path.join(RAW, "t456", "cbp22co.zip")) as z:
    cbp = pd.read_csv(z.open(z.namelist()[0]), dtype=str)
cbp.columns = [c.lower() for c in cbp.columns]
cbp["fips"] = cbp.fipstate + cbp.fipscty
cbp["est"] = pd.to_numeric(cbp.est, errors="coerce").fillna(0)
pick = {"44----": "ret", "811///": "repair", "812///": "pers", "722///": "food", "------": "all", "42----": "whl"}
c = cbp[cbp.naics.isin(pick)].assign(k=lambda x: x.naics.map(pick)).pivot_table(index="fips", columns="k", values="est", aggfunc="sum").add_prefix("e_")
print("CBP Connecticut county codes:", sorted(i for i in c.index if i.startswith("09"))[:10])

pe = pd.read_csv(os.path.join(RAW, "co-est2021-alldata.csv"), encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
pe = pe[pe.SUMLEV == 50]
pop = pd.Series(pe.POPESTIMATE2021.to_numpy(), index=(pe.STATE.str.zfill(2) + pe.COUNTY.str.zfill(3)), name="pop")
inc = pd.read_csv(os.path.join(RAW, "t456", "acsdt5y2023-b19013.dat"), sep="|", dtype=str)
inc = inc[inc.GEO_ID.str.startswith("0500000US")]
inc = pd.Series(pd.to_numeric(inc.B19013_E001, errors="coerce").to_numpy(), index=inc.GEO_ID.str[-5:], name="income")

df = co[["fips", "NAME", "km2"]].set_index("fips").join(oc).join(c).join(pop).join(inc).fillna({"o_shop": 0, "o_eam": 0, "o_food": 0})
# Connecticut: CBP 2022 uses planning regions, the 2020 polygons are the old counties; fold both to the state
ct = df.index.str.startswith("09")
if df.loc[ct, "e_ret"].isna().all():
    cc = c[c.index.str.startswith("09")].sum()
    row = df[ct].sum(numeric_only=True)
    for k in cc.index:
        row[k] = cc[k]
    row["income"] = np.nan
    df = pd.concat([df[~ct], pd.DataFrame([row], index=["09000"])])
    df.loc["09000", "NAME"] = "Connecticut"
df["off_err"] = df.e_ret + df.e_pers + df.e_repair
df["off_food"] = df.e_food
df["jobs"] = np.nan
df = df[df.off_err > 0]
df.index.name = "fips"
df.to_csv(os.path.join(HERE, "us_county.csv"))
print(len(df), "counties;", df[["pop", "o_shop", "o_food", "e_ret", "off_err", "off_food"]].sum().to_dict())
