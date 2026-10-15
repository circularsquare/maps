"""T-117, T-276: the Northeast's sites and commutes between zones, data/packs/intercity.json.

Endpoint sites from the Northeast's zone packs (the registry's cities with a `market`: 600 m squares
in the New York equirectangular frame, `origin`; 6 km zones), each site's parking for an arriving
car (the local layer's rule), then the joint commute gravity, which reads these sites and writes
the commutes between zones into the same file. Mercer County is in New York's pack since T-276;
its market place (id 2) keeps its name and has no sites.

    python pipeline/intercity.py            the sites, the joint commute gravity (rewrites every
                                            Northeast pack's gravity, T-276), then every zone's
                                            market pack (pipeline/market_zones.py, T-377);
                                            --no-joint skips the gravity

The game no longer reads this file (T-377): pipeline/market_zones.py takes the Northeast's sites
and commutes from it into each zone's own market pack, `data/packs/<id>.market.json`.
"""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from read_pack import load_pack
from build_city import CITIES, R_EARTH

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = CITIES["nyc"]["origin"]
# Stable IDs: saved city selections and registry entries use these indices.
REGISTRY = json.loads((ROOT / "app/src/game/registry.json").read_text(encoding="utf-8"))
# a city with market -1 is not in the Northeast market (Toronto, T-277, until T-276 places it)
CITY_MARKETS = tuple((c["market"], c["id"]) for c in REGISTRY["cities"] if c["market"] >= 0)
MARKETS = [p["name"] for p in REGISTRY["market"]["places"]]

# T-276: the market's trips a weekday per resident (fitted with its curve, research/intercity/market_fit.py)
TRIP_RATE = 0.107
# the local layer's parking (sim kernel `Params`: park_min, park_jobs_per_km2_per_min, park_cost_cap_min)
PARK_MIN, PARK_JOBS_PER_KM2_PER_MIN, PARK_CAP_MIN = 3.0, 2500.0, 40.0


def lonlat_to_market(lon, lat):
    return (R_EARTH * np.radians(lon - ORIGIN[0]) * math.cos(math.radians(ORIGIN[1])),
            R_EARTH * np.radians(lat - ORIGIN[1]))


def sites():
    """Sites ([x, y, people, jobs, market]) and each one's parking for an arriving car."""
    rows = []
    for market, key in CITY_MARKETS:
        h, a = load_pack(key)
        o = h["origin"]
        lat = o["lat"] + np.degrees(a["y_m"] / R_EARTH)
        lon = o["lon"] + np.degrees(a["x_m"] / (R_EARTH * math.cos(math.radians(o["lat"]))))
        # T-276: parking as the local layer charges it at a destination: its 2 km gravity zone's
        # jobs per km² (kernel `mode_choice`)
        zm = h["gravity"]["zone_m"] / 1000.0
        zj = np.bincount(a["zone"], a["jobs"].astype(float), h["zones"])
        park = PARK_MIN + np.minimum(zj[a["zone"]] / zm ** 2 / PARK_JOBS_PER_KM2_PER_MIN, PARK_CAP_MIN)
        rows.append((market, lon, lat, a["pop"], a["jobs"], park))
    out, parks = [], []
    for market, lon, lat, pop, jobs, park in rows:
        x, y = lonlat_to_market(lon, lat)
        groups = {}
        for xx, yy, pp, jj, pk in zip(x, y, pop, jobs, park):
            if pp + jj <= 0: continue
            k = (int(xx // 600), int(yy // 600))
            r = groups.setdefault(k, [0., 0., 0., 0., 0., 0.])
            r[0] += xx * (pp + jj); r[1] += yy * (pp + jj)
            r[2] += pp; r[3] += jj
            r[4] += pk * (jj + 1e-6); r[5] += jj + 1e-6
        for r in groups.values():
            w = r[2] + r[3]
            out.append([round(float(r[0]/w), 1), round(float(r[1]/w), 1), round(float(r[2]), 3), round(float(r[3]), 3), market])
            parks.append(round(float(r[4] / r[5]), 2))
    return out, parks


def joint():
    """T-276: one commute gravity over every zone pack (`joint_gravity`): rewrites each pack's
    gravity factors and the market pack's commutes between zones. About a minute (T-297)."""
    cargo_bin("joint_gravity", "--root", str(ROOT), "--threads", os.environ.get("OMP_NUM_THREADS", "3"),
              "--report", str(ROOT / "data/work/joint_report.json"))


def cargo_bin(name, *args):
    env = dict(os.environ)
    env["PATH"] = os.path.join(os.path.expanduser("~"), ".cargo", "bin") + os.pathsep + env.get("PATH", "")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    env["CARGO_TARGET_DIR"] = str(ROOT / "data/work/target")
    cargo = shutil.which("cargo", path=env["PATH"])
    if not cargo:
        sys.exit("cargo not found: install Rust (rustup) or put ~/.cargo/bin on PATH")
    subprocess.run([cargo, "run", "--quiet", "--release", "--features", "demand", "--bin", name,
                    "--manifest-path", str(ROOT / "sim/Cargo.toml"), "--", *args], env=env, check=True)


def frame():
    """The frame the sites are in (T-128), as the registry writes frames."""
    return {"kind": "equirect", "lon": ORIGIN[0], "lat": ORIGIN[1]}


if __name__ == "__main__":
    path = ROOT / "data/packs/intercity.json"
    site_rows, site_park = sites()
    data = {"format": 2, "origin": {"lon": ORIGIN[0], "lat": ORIGIN[1]}, "frame": frame(),
            "markets": MARKETS, "zone_m": 6000, "trip_rate": TRIP_RATE,
            "description": "The Northeast's endpoint sites and commutes between zones; the game reads each zone's market pack instead (T-377, pipeline/market_zones.py).",
            "sources": ["2020 Census block population less institutional group quarters (T-234)", "LODES 2023/2022 jobs", "registered zone-pack cells"],
            "sites": site_rows, "site_park": site_park}
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(path, len(site_rows), "sites", path.stat().st_size, "bytes")
    if "--no-joint" not in sys.argv[1:]:
        joint()
    import market_zones
    market_zones.main([])

