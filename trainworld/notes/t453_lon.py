"""T-453 stage 1, London: everyday attraction from BRES 2024 employment by LSOA x SIC division
against the OSM counts. Sector jobs per cell = the pack's jobs x the LSOA's sector share."""
import json
import numpy as np
import pandas as pd
from t453_common import pack, signals_on, circle_shares, zone_misplaced, table
import england

p = pack("lon")
n = p["n"]
print(f"lon: {n:,} cells, jobs {p['jobs'].sum():,.0f}")
with open(r"C:\Users\anita\projects\maps\trainworld\data\raw\tier0\lsoa2021_bgc_lon.json", encoding="utf-8") as f:
    feats = json.load(f)
from shapely.geometry import shape
from shapely import make_valid
units = [(ft["properties"]["LSOA21CD"], make_valid(shape(ft["geometry"]))) for ft in feats]
idx, _ = england.cell_units(p["lat"], p["lon"], units)
code = np.array([units[i][0] if i >= 0 else "" for i in idx])
print(f"  cells in an LSOA: {np.mean(idx >= 0):.1%}")

b = pd.read_csv(r"C:\Users\anita\projects\maps\trainworld\data\raw\t453\bres_lsoa_sic2_england.csv", dtype={"INDUSTRY_CODE": str})
w = b.pivot_table(index="GEOGRAPHY_CODE", columns="INDUSTRY_CODE", values="OBS_VALUE", aggfunc="sum").fillna(0.0)
tot, year = england.bres_lsoa()
print(f"  BRES all industries {year}; sectors 2024")
tot = pd.Series(tot)
def share(cols):
    s = (w[cols].sum(1) / tot.reindex(w.index)).clip(upper=1.0).fillna(0.0)
    return pd.Series(code).map(s).fillna(0.0).to_numpy()
jobs = p["jobs"]
sig = signals_on(p, "lon")
osm_all = sig["osm_errand"] + sig["osm_leisure"] + sig["osm_job"] + sig["osm_land"]
zsum = lambda v: np.bincount(p["zone"], weights=v, minlength=len(p["gx"]))[p["zone"]]
def spread_in_zone(zone_total, osm):
    return zone_total * np.divide(osm, zsum(osm), out=np.zeros(n), where=zsum(osm) > 0)
def fallbacks(osm):
    zj, zo = zsum(jobs), zsum(osm)
    return {"fallback A": spread_in_zone(zj, osm),
            "fallback B": jobs * np.divide(zo, zsum(osm_all), out=np.zeros(n), where=zsum(osm_all) > 0),
            "fallback C (jobs only)": jobs.copy(),
            "fallback D": spread_in_zone(np.sqrt(zj / zj.sum() * zo / zo.sum()), osm)}
E = {"before (OSM)": p["errand_jobs"], "sector": jobs * share(["45", "47", "95", "96"]),
     "sector +94": jobs * share(["45", "47", "94", "95", "96"]), **fallbacks(sig["osm_errand"])}
L = {"before (OSM)": p["leisure_jobs"], "sector": jobs * share(["55", "56", "90", "91", "92", "93"]), **fallbacks(sig["osm_leisure"])}
print("totals: errands sector %.0f, leisure sector %.0f" % (E["sector"].sum(), L["sector"].sum()))
CENTRES = {"West End": (51.5152, -0.1419), "City": (51.5133, -0.0889), "Canary Wharf": (51.5054, -0.0235)}
print("\nerrands, % within 1.5 km")
table(circle_shares(p, E, CENTRES), list(E))
print("\nleisure jobs part, % within 1.5 km")
table(circle_shares(p, L, CENTRES), list(L))
print("\nmisplaced between 2 km zones against the sector figures: errands",
      {k: round(zone_misplaced(p, v, E["sector"]), 1) for k, v in E.items()},
      "leisure", {k: round(zone_misplaced(p, v, L["sector"]), 1) for k, v in L.items()})
