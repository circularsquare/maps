"""T-456: OSM places against the 2021 economic census (9-1A establishments, 9-1B persons employed)
by municipality, all of Japan. Writes japan_muni.csv beside this script."""
import os, sys, glob, json
os.environ["OMP_NUM_THREADS"] = "2"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import numpy as np, pandas as pd, geopandas as gpd, openpyxl
import japan

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:\Users\anita\projects\maps\trainworld\data\raw"
JP = os.path.join(RAW, "tier0", "japan")


def census_table(path):
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = rows[5]
    num = lambda v: float(v) if isinstance(v, (int, float)) else 0.0

    def cols(pred):
        return [j for j, h in enumerate(head) if isinstance(h, str) and pred(h.split("_")[0])]
    two = lambda cs: cols(lambda c: c in cs)
    adm = lambda cs: cols(lambda c: len(c) == 3 and c.endswith("0") and c[:2] in cs)
    R = [str(c) for c in range(56, 62)]
    S = {"AS": two(["AS"]), "ret": two(R), "ret_adm": adm(R), "whl": two([str(c) for c in range(50, 56)]),
         "pers": two(["78", "79"]), "pers_adm": adm(["78", "79"]), "food": two(["76", "77"]), "food_adm": adm(["76", "77"]),
         "acc": two(["75"]), "amuse": two(["80"])}
    assert all(S.values()), [k for k, v in S.items() if not v]
    out = {}
    for r in rows[9:]:
        if not isinstance(r[1], str):
            continue
        code, lvl = r[1].split("_")[0], str(r[0])
        v = {k: sum(num(r[j]) for j in js) for k, js in S.items()}
        out[code] = {"lvl": lvl, "AS": v["AS"], "ret": v["ret"] - v["ret_adm"], "whl": v["whl"],
                     "pers": v["pers"] - v["pers_adm"], "food": v["food"] - v["food_adm"], "acc": v["acc"], "amuse": v["amuse"]}
    return pd.DataFrame(out).T


est = census_table(os.path.join(RAW, "t456", "ec2021_000040067884.xlsx")).add_prefix("e_")
emp = census_table(os.path.join(RAW, "t453", "ec2021_9-1B.xlsx")).add_prefix("p_")
ec = est.join(emp, how="outer")
nat = ec.loc["00000"]
print("national establishments: retail %.0f, personal services %.0f, eating %.0f, all %.0f" % (nat.e_ret, nat.e_pers, nat.e_food, nat.e_AS))

# municipalities: single-prefecture caches (Hokkaido by subprefecture)
files = [f for f in glob.glob(os.path.join(JP, "municipalities_*.geojson"))
         if (os.path.basename(f)[15:-8].isdigit() and len(os.path.basename(f)[15:-8]) == 2)
         or (os.path.basename(f).startswith("municipalities_01-") and "_" not in os.path.basename(f)[15:-8]
             and "~" not in f and "01203" not in f)]
mg = pd.concat([gpd.read_file(f) for f in files], ignore_index=True)
mg = mg.drop_duplicates("code")
mg = gpd.GeoDataFrame(mg, geometry="geometry", crs="EPSG:4326")
mg["km2"] = mg.to_crs("EPSG:6933").area / 1e6
print(f"{len(mg):,} municipalities from {len(files)} files")

# census meshes: population 2020 and persons employed 2021
codes = sorted({os.path.basename(f)[8:12] for f in glob.glob(os.path.join(JP, "mesh", "T001141_*.zip"))})
pop = japan.mesh_table("T001141", codes)["T001141001"]
jobs = japan.mesh_table("T001162", codes)["T001162023"]
m = pd.DataFrame({"pop": pop, "jobs": jobs}).fillna(0.0)
la, lo = japan.mesh_sw(m.index.to_numpy())
pts = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(lo + 1 / 320, la + 1 / 480), crs="EPSG:4326")
j = gpd.sjoin(pts, mg[["code", "geometry"]], how="left", predicate="within")
j = j[~j.index.duplicated()]
agg = j.groupby("code")[["pop", "jobs"]].sum()
print(f"meshes: {len(m):,}, pop {m['pop'].sum():,.0f} ({j['pop'][j.code.isna()].sum():,.0f} outside any polygon)")

# OSM places
osm = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RAW, "t456", "pois_asia_japan_*.csv.gz"))], ignore_index=True)
osm = osm.drop_duplicates(["cls", "lat", "lon"])  # regions overlap along borders
op = gpd.GeoDataFrame(osm, geometry=gpd.points_from_xy(osm.lon, osm.lat), crs="EPSG:4326")
jo = gpd.sjoin(op, mg[["code", "geometry"]], how="inner", predicate="within")
jo = jo[~jo.index.duplicated()]
oc = jo.groupby(["code", "cls"]).size().unstack(fill_value=0).add_prefix("o_")
print(f"OSM places: {len(osm):,}, in a municipality {len(jo):,}")

df = mg[["code", "name", "pref", "km2"]].set_index("code").join(agg).join(oc).join(ec).fillna(0.0)
df.to_csv(os.path.join(HERE, "japan_muni.csv"), encoding="utf-8")
print(df[["pop", "jobs", "o_shop", "o_food", "e_ret", "e_pers", "e_food", "p_ret"]].sum())
print("unmatched census codes in polygons:", int((df.e_AS == 0).sum()))
