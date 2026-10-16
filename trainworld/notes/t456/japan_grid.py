"""T-456: fallback D tested on a 2 km grid over all of Japan, within each prefecture, against the
economic census's establishments (500 m mesh, I x municipal retail share + N x municipal 78-79 share).
Also thins OSM at random to see where D stops beating jobs alone."""
import os, sys, glob
os.environ["OMP_NUM_THREADS"] = "2"
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\anita\projects\maps\trainworld\pipeline")
import numpy as np, pandas as pd, geopandas as gpd
import japan

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:\Users\anita\projects\maps\trainworld\data\raw"
JP = os.path.join(RAW, "tier0", "japan")
rng = np.random.default_rng(456)

mu = pd.read_csv(os.path.join(HERE, "japan_muni.csv"), dtype={"code": str}).set_index("code")
mu = mu[mu.e_lvl.astype(str).isin(["0", "2", "3"])]
r_ret = (mu.e_ret / (mu.e_ret + mu.e_whl)).fillna(0.5)
nat_n = None

files = [f for f in glob.glob(os.path.join(JP, "municipalities_*.geojson"))
         if (os.path.basename(f)[15:-8].isdigit() and len(os.path.basename(f)[15:-8]) == 2)
         or (os.path.basename(f).startswith("municipalities_01-") and "_" not in os.path.basename(f)[15:-8]
             and "~" not in f and "01203" not in f)]
mg = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in files], ignore_index=True).drop_duplicates("code"), geometry="geometry", crs="EPSG:4326")

codes = sorted({os.path.basename(f)[8:12] for f in glob.glob(os.path.join(JP, "mesh", "T001141_*.zip"))})
pop = japan.mesh_table("T001141", codes)["T001141001"]
ec = japan.mesh_table("T001162", codes)
m = pd.DataFrame({"pop": pop, "jobs": ec["T001162022"], "I": ec["T001162011"], "N": ec["T001162016"], "M": ec["T001162015"]}).fillna(0.0)
la, lo = japan.mesh_sw(m.index.to_numpy())
pts = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(lo + 1 / 320, la + 1 / 480), crs="EPSG:4326")
j = gpd.sjoin(pts, mg[["code", "geometry"]], how="inner", predicate="within")
j = j[~j.index.duplicated()]
# N's 78-79 share by municipality: 9-1A personal services over N (78-80); approximated as pers / (pers + amuse)
r_pers = (mu.e_pers / (mu.e_pers + mu.e_amuse)).fillna(0.7)
j["off"] = j.I * j.code.map(r_ret).fillna(0.5) + j.N * j.code.map(r_pers).fillna(0.7)
j["pref"] = j.code.str[:2]
la, lo = japan.mesh_sw(j.index.to_numpy())
j["gx"] = np.floor((lo - 100) * 40 + 1e-9).astype(np.int64)  # 4 meshes of 1/160 degree
j["gy"] = np.floor(la * 60 + 1e-9).astype(np.int64)            # 4 meshes of 1/240 degree

osm = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RAW, "t456", "pois_asia_japan_*.csv.gz"))], ignore_index=True)
osm = osm.drop_duplicates(["cls", "lat", "lon"])
osm = osm[osm.cls.isin(["shop", "eam"])]
osm["mesh"] = japan.mesh_of(osm.lat.to_numpy(), osm.lon.to_numpy())
osm = osm[osm.mesh.isin(j.index)]
osm["gx"] = j.gx.reindex(osm.mesh).to_numpy()
osm["gy"] = j.gy.reindex(osm.mesh).to_numpy()
osm["pref"] = j.pref.reindex(osm.mesh).to_numpy()

g = j.groupby(["pref", "gx", "gy"])[["pop", "jobs", "off"]].sum()
print(f"2 km cells: {len(g):,}; official errand establishments {g.off.sum():,.0f}; OSM errand places {len(osm):,}")


def osm_counts(frac):
    o = osm if frac >= 1 else osm.sample(frac=frac, random_state=int(frac * 1e6))
    return o.groupby(["pref", "gx", "gy"]).size().reindex(g.index, fill_value=0).astype(float)


def misplaced_by_pref(w):
    out = {}
    for p, y in g.assign(w=w).groupby(level=0):
        a = y.off / y.off.sum()
        b = y.w / y.w.sum() if y.w.sum() > 0 else a * 0
        out[p] = 50 * np.abs(a - b).sum()
    return pd.Series(out)


share = lambda v: v / v.groupby(level=0).transform("sum")
pw = g["pop"].groupby(level=0).sum()
wavg = lambda s: (s * pw).sum() / pw.sum()
compl = pd.Series(1000 * osm.groupby("pref").size() / pw)
rows = []
for frac in (1.0, 0.3, 0.1, 0.03, 0.01):
    O = osm_counts(frac)
    sj, sp, so = share(g.jobs), share(g["pop"]), share(O)
    so1 = share(O + 1.0)  # one pseudo-place per cell
    nO = O.groupby(level=0).sum()
    c = (nO / pw * 1000 / 5.0).clip(0, 1).reindex(g.index, level=0)  # completeness against 5 OSM errand places per 1k
    res = {"jobs": misplaced_by_pref(sj), "population": misplaced_by_pref(sp), "OSM": misplaced_by_pref(so),
           "D": misplaced_by_pref(np.sqrt(sj * so)), "D, +1 place a cell": misplaced_by_pref(np.sqrt(sj * so1)),
           "D', a = c/2": misplaced_by_pref(sj ** (1 - c / 2) * so ** (c / 2)),
           "D', a = c/2, +1 place": misplaced_by_pref(sj ** (1 - c / 2) * so1 ** (c / 2))}
    rows.append((frac, 1000 * len(osm) * frac / pw.sum(), {k: wavg(v) for k, v in res.items()}))
    if frac == 1.0:
        full = res
keys = list(rows[0][2])
print("\n| OSM kept | OSM errand places /1k res | " + " | ".join(keys) + " |")
print("|---" * (len(keys) + 2) + "|")
for frac, per, r in rows:
    print(f"| {frac:.0%} | {per:.2f} | " + " | ".join(f"{r[k]:.1f}" for k in keys) + " |")

# per prefecture, the real (uneven) completeness: does D beat jobs where OSM is thin?
t = pd.DataFrame({"osm_per_1k": compl, "shops/official": osm.groupby("pref").size() / g.off.groupby(level=0).sum(),
                  "jobs": full["jobs"], "D": full["D"], "D+1": full["D, +1 place a cell"], "OSM": full["OSM"]}).sort_values("osm_per_1k")
t["D gain"] = t.jobs - t.D
print("\nper prefecture, sorted by OSM errand places per 1k residents (misplaced %, 2 km cells):")
print(t.round(2).to_string())
print("\ncorrelation of D's gain over jobs with OSM per 1k: %.2f" % np.corrcoef(t.osm_per_1k, t["D gain"])[0, 1])
