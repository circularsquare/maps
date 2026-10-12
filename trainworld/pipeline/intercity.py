"""T-117: reproducible coarse Northeast non-commute travel endpoints.

New York, Philadelphia, Boston and Washington city cells plus Mercer County
Census block/LODES job internal points.
600 m sites retain neighbourhood access; 6 km zones form the gravity market.
No observed OD counts are implied by this first tunable prototype.
"""
import json
import math
from pathlib import Path
import numpy as np
from read_pack import load_pack
from build_city import blocks_for_state, jobs_for_state, CITIES, R_EARTH

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = CITIES["nyc"]["origin"]
# Stable IDs: saved city selections and registry entries use these indices.
REGISTRY = json.loads((ROOT / "app/src/game/registry.json").read_text(encoding="utf-8"))
CITY_MARKETS = tuple((c["market"], c["id"]) for c in REGISTRY["cities"])
MARKETS = [p["name"] for p in REGISTRY["market"]["places"]]

def sites():
    rows = []
    for market, key in CITY_MARKETS:
        h, a = load_pack(key)
        o = h["origin"]
        lat = o["lat"] + np.degrees(a["y_m"] / R_EARTH)
        lon = o["lon"] + np.degrees(a["x_m"] / (R_EARTH * math.cos(math.radians(o["lat"]))))
        rows.append((market, lon, lat, a["pop"], a["jobs"]))
    b, _ = blocks_for_state("34", {"34021"})
    j, _, _ = jobs_for_state("nj", (2023, 2022), {"34021"})
    b = b.merge(j[["geoid", "jobs"]], on="geoid", how="left").fillna({"jobs": 0})
    rows.append((2, b.lon.to_numpy(), b.lat.to_numpy(), b["pop"].to_numpy(), b.jobs.to_numpy()))
    out = []
    for market, lon, lat, pop, jobs in rows:
        x = R_EARTH * np.radians(lon - ORIGIN[0]) * math.cos(math.radians(ORIGIN[1]))
        y = R_EARTH * np.radians(lat - ORIGIN[1])
        groups = {}
        for xx, yy, pp, jj in zip(x, y, pop, jobs):
            if pp + jj <= 0: continue
            k = (int(xx // 600), int(yy // 600))
            r = groups.setdefault(k, [0., 0., 0., 0.])
            r[0] += xx * (pp + jj); r[1] += yy * (pp + jj)
            r[2] += pp; r[3] += jj
        for r in groups.values():
            w = r[2] + r[3]
            out.append([round(float(r[0]/w), 1), round(float(r[1]/w), 1), round(float(r[2]), 3), round(float(r[3]), 3), market])
    return out

if __name__ == "__main__":
    data = {"format": 1, "origin": {"lon": ORIGIN[0], "lat": ORIGIN[1]},
            "markets": MARKETS,
            "zone_m": 6000, "trip_rate": 0.015,
            "description": "Judgement prototype: 0.015 daily non-commute long-distance departures per resident, distributed over represented destinations. Not fitted OD data.",
            "sources": ["2020 Census block population", "LODES 2023/2022 jobs", "registered city-pack cells"],
            "sites": sites()}
    path = ROOT / "data/packs/intercity.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(path, len(data["sites"]), "sites", path.stat().st_size, "bytes")

