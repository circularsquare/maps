"""T-453 stage 1, Tokyo: everyday attraction from economic census sector jobs against the OSM counts.
Sector jobs per cell = the pack's jobs x the 500 m mesh's sector share (T001162, persons employed by
major industry), retail and customer-facing shares of each major industry from the municipality's
table 9-1B (2021 economic census, persons employed by minor industry)."""
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import openpyxl
from shapely import points
from t453_common import pack, signals_on, circle_shares, zone_misplaced, social, table
import japan

p = pack("tyo")
n = p["n"]
print(f"tyo: {n:,} cells, jobs {p['jobs'].sum():,.0f}")

# --- mesh sector shares
bounds = (p["lon"].min(), p["lat"].min(), p["lon"].max(), p["lat"].max())
t = japan.mesh_table("T001162", japan.first_meshes(bounds))
col = lambda k: t[f"T001162{k:03d}"].astype(float)
AS, I, M, N, R = col(23), col(33), col(37), col(38), col(42)
cm = pd.Series(japan.mesh_of(p["lat"], p["lon"]))
def share(x):
    s = (x / AS.where(AS > 0)).fillna(0.0)
    return cm.map(s).fillna(0.0).to_numpy()
rI, rM, rN = share(I), share(M), share(N)
print(f"  mesh: {len(t):,} meshes; cells with a mesh: {np.mean(cm.isin(t.index)):.1%}")

# --- municipality splits from 9-1B
wb = openpyxl.load_workbook(r"C:\Users\anita\projects\maps\trainworld\data\raw\t453\ec2021_9-1B.xlsx", read_only=True)
ws = wb.worksheets[0]
rows = list(ws.iter_rows(values_only=True))
head = rows[5]
num = lambda v: float(v) if isinstance(v, (int, float)) else 0.0
def cols_where(pred):
    return [j for j, h in enumerate(head) if isinstance(h, str) and pred(h.split("_")[0])]
two = lambda codes: cols_where(lambda c: c in codes)
admin = lambda codes: cols_where(lambda c: len(c) == 3 and c.endswith("0") and c[:2] in codes)
RET = [str(c) for c in range(56, 62)]
SETS = {"I": cols_where(lambda c: c == "I"), "M": cols_where(lambda c: c == "M"), "N": cols_where(lambda c: c == "N"),
        "ret": two(RET), "ret_adm": admin(RET), "whl": two([str(c) for c in range(50, 56)]),
        "M2": two(["75", "76", "77"]), "M_adm": admin(["75", "76", "77"]),
        "p78": two(["78", "79"]), "p78_adm": admin(["78", "79"]), "p80": two(["80"]), "p80_adm": admin(["80"])}
assert all(SETS.values()), {k: v for k, v in SETS.items() if not v}
mun = {}
for r in rows[9:]:
    if not r[1] or not isinstance(r[1], str):
        continue
    code = r[1].split("_")[0]
    mun[code] = {k: sum(num(r[j]) for j in js) for k, js in SETS.items()}
def ratio(code, a, b, sub=None):
    m = mun.get(code)
    if not m or m[b] <= 0:
        return np.nan
    return max(m[a] - (m[sub] if sub else 0.0), 0.0) / m[b]
nat = "00000"
print("  national: retail customer-facing / I = %.3f, M customer / M = %.3f, 78-79 / N = %.3f, 80 / N = %.3f" % (
    ratio(nat, "ret", "I", "ret_adm"), ratio(nat, "M2", "M", "M_adm"), ratio(nat, "p78", "N", "p78_adm"), ratio(nat, "p80", "N", "p80_adm")))
for c in ("13101", "13102", "13103", "13104", "13113", "13115", "13203"):
    print(f"  {c}: retail/I {ratio(c, 'ret', 'I', 'ret_adm'):.3f}  wholesale/I {ratio(c, 'whl', 'I'):.3f}  M cust {ratio(c, 'M2', 'M', 'M_adm'):.3f}")

frames = [gpd.read_file(rf"C:\Users\anita\projects\maps\trainworld\data\raw\tier0\japan\municipalities_{q}.geojson") for q in ("08", "11", "12", "13", "14")]
mg = pd.concat(frames, ignore_index=True)
mg = gpd.GeoDataFrame(mg, geometry="geometry", crs=frames[0].crs)
j = gpd.sjoin(gpd.GeoDataFrame(geometry=points(p["lon"], p["lat"]), crs=mg.crs), mg, how="left", predicate="within")
j = j[~j.index.duplicated()]
code = j["code"].fillna("?").to_numpy()
name = j["name"].fillna("?").to_numpy()
pref = j["pref"].fillna("?").astype(str).to_numpy()
def per_cell(a, b, sub=None):
    cache = {}
    v = np.array([cache.setdefault(c, ratio(c, a, b, sub)) for c in code])
    miss = np.isnan(v)
    if miss.any():
        print(f"    {a}/{b}: {miss.mean():.2%} of cells with no municipal row, national used")
    v[miss] = ratio(nat, a, b, sub)
    return v
s_ret, s_m = per_cell("ret", "I", "ret_adm"), per_cell("M2", "M", "M_adm")
s78, s80 = per_cell("p78", "N", "p78_adm"), per_cell("p80", "N", "p80_adm")

jobs = p["jobs"]
E = {"before (OSM)": p["errand_jobs"],
     "I whole": jobs * (rI + rN * s78),
     "sector": jobs * (rI * s_ret + rN * s78)}
L = {"before (OSM)": p["leisure_jobs"],
     "sector": jobs * (rM * s_m + rN * s80)}
sig = signals_on(p, "tyo")
osm_all = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
zsum = lambda w: np.bincount(p["zone"], weights=w, minlength=len(p["gx"]))[p["zone"]]
# fallback A: each 2 km zone's total jobs, spread over its cells by OSM places of the purpose
E["fallback A"] = zsum(jobs) * np.divide(sig["osm_errand"], zsum(sig["osm_errand"]), out=np.zeros(n), where=zsum(sig["osm_errand"]) > 0)
L["fallback A"] = zsum(jobs) * np.divide(sig["osm_leisure"], zsum(sig["osm_leisure"]), out=np.zeros(n), where=zsum(sig["osm_leisure"]) > 0)
# fallback B: the cell's jobs x the purpose's share of the cell's OSM places (zone-level shares)
E["fallback B"] = jobs * np.divide(zsum(sig["osm_errand"]), zsum(osm_all), out=np.zeros(n), where=zsum(osm_all) > 0)
L["fallback B"] = jobs * np.divide(zsum(sig["osm_leisure"]), zsum(osm_all), out=np.zeros(n), where=zsum(osm_all) > 0)
# fallback C: fixed national sector share of each cell's jobs (no OSM)
E["fallback C (jobs only)"] = jobs.copy()
L["fallback C (jobs only)"] = jobs.copy()
# fallback D: each zone's weight the geometric mean of its jobs share and its OSM share, spread by OSM
for D, k in ((E, "osm_errand"), (L, "osm_leisure")):
    zj, zo = zsum(jobs), zsum(sig[k])
    D["fallback D"] = np.sqrt(zj / zj.sum() * zo / zo.sum()) * np.divide(sig[k], zo, out=np.zeros(n), where=zo > 0)
np.savez_compressed(r"C:\Users\anita\projects\maps\artifacts\trainworld\t453_tyo_variants.npz",
                    **{"E_" + k: v for k, v in E.items()}, **{"L_" + k: v for k, v in L.items()})
print("totals: errands sector %.0f, leisure sector %.0f, all jobs %.0f" % (E["sector"].sum(), L["sector"].sum(), jobs.sum()))

CENTRES = {"Shinjuku": (35.6896, 139.7006), "Shibuya": (35.6580, 139.7016), "Ikebukuro": (35.7295, 139.7109),
           "Ginza": (35.6717, 139.7649)}
print("\nerrands, % within 1.5 km")
table(circle_shares(p, E, CENTRES), list(E))
print("\nleisure jobs part, % within 1.5 km")
table(circle_shares(p, L, CENTRES), list(L))
print("\nmisplaced between 2 km zones against the sector figures: errands",
      {k: round(zone_misplaced(p, v, E["sector"]), 1) for k, v in E.items()},
      "leisure", {k: round(zone_misplaced(p, v, L["sector"]), 1) for k, v in L.items()})

# --- T-451's centre share
CORE = ["千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "墨田区", "江東区", "品川区", "目黒区", "渋谷区", "中野区", "豊島区", "北区", "荒川区"]
NEWURB = ["大田区", "世田谷区", "杉並区", "練馬区", "板橋区", "足立区", "葛飾区", "江戸川区", "武蔵野市", "三鷹市", "調布市", "狛江市"]
region = np.full(n, 3)
region[np.array([q.startswith("13") for q in pref])] = 2
region[np.isin(name, NEWURB)] = 1
region[np.isin(name, CORE)] = 0
gx, gy, zm, zone, nz = p["gx"], p["gy"], p["zm"], p["zone"], len(p["gx"])
pop = p["pop"]
def region_od(w, scale, rate):
    mass = np.bincount(zone, weights=w, minlength=nz)
    Rm = np.stack([np.bincount(zone, weights=w * (region == r), minlength=nz) for r in range(4)], 1)
    Rm = np.divide(Rm, mass[:, None], out=np.zeros_like(Rm), where=mass[:, None] > 0)
    zp = np.stack([np.bincount(zone, weights=pop * (region == r), minlength=nz) for r in range(4)], 1)
    od = np.zeros((4, 4))
    for oz in np.nonzero(zp[:, :3].sum(1) > 0)[0]:
        d = np.hypot(gx - gx[oz], gy - gy[oz]) * zm
        d[oz] = 1040.0
        pz = mass * np.exp(-d / scale)
        pz /= pz.sum()
        od += np.outer(zp[oz] * rate, pz @ Rm)
    return od
med = p["medical_visits"]
print("\ncentre share of everyday outings (errands + social + medical), PT 2018: core 84.4, new urban 15.6, Tama 3.5")
for k in ("before (OSM)", "sector", "fallback A", "fallback C (jobs only)", "fallback D"):
    od = region_od(E[k], 3000.0, 0.299) + region_od(social(p, L[k]), 6000.0, 0.200) + region_od(med, 5000.0, 0.04)
    sh = [100 * od[r, 0] / od[r].sum() for r in range(3)]
    own = [100 * od[r, r] / od[r].sum() for r in range(3)]
    out = [100 * od[r, 3] / od[r].sum() for r in range(3)]
    print(f"  {k}: to centre from core/new urban/Tama {sh[0]:.1f} / {sh[1]:.1f} / {sh[2]:.1f}; own area {own[0]:.1f} / {own[1]:.1f} / {own[2]:.1f}; outside Tokyo-to {out[0]:.1f} / {out[1]:.1f} / {out[2]:.1f}")
