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
    python pipeline/intercity.py --group jp  a country's packs as their own joint gravity group
                                            (T-463, Japan), then their market packs; --no-market
                                            stops after the gravity

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
# the local layer's parking (sim kernel `Params` defaults: park_min, park_paid_min, park_paid_d0,
# park_paid_cap_min, park_free_d; `CELL_KM2`)
PARK_MIN, PARK_PAID_MIN, PARK_PAID_D0, PARK_PAID_CAP_MIN, PARK_FREE_D = 3.0, 15.0, 3000.0, 50.0, 8000.0
CELL_KM2 = 0.1053
# kernel `CORDONS` (congestion charges, perceived minutes a one-way trip): lon/lat rings, charge
CORDONS = [
    (13.5, [(-74.020, 40.700), (-74.016, 40.715), (-74.012, 40.730), (-74.009, 40.745), (-74.002, 40.760), (-73.994, 40.774), (-73.959, 40.759), (-73.968, 40.748), (-73.972, 40.738), (-73.974, 40.726), (-73.978, 40.711), (-73.999, 40.707), (-74.012, 40.701)]),
    (35.1, [(-0.159, 51.513), (-0.163, 51.520), (-0.126, 51.528), (-0.105, 51.530), (-0.088, 51.526), (-0.077, 51.524), (-0.072, 51.515), (-0.075, 51.5055), (-0.100, 51.495), (-0.125, 51.487), (-0.144, 51.495), (-0.152, 51.503)]),
]


def in_ring(ring, x, y):
    inside = np.zeros(len(x), bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        (xi, yi), (xj, yj) = ring[i], ring[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            cross = (yi > y) != (yj > y)
            inside ^= cross & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        j = i
    return inside


def park_by_cell(h, a, lon, lat):
    """Each cell's parking for an arriving car, perceived minutes: the mean over drivers in its
    2 km zone. Mirrors the sim kernel's `Params::park_mean` over `park_parts` (T-380, T-440) with
    `cordon_by_zone`: 3 to park and the zone's congestion charge for everyone, a paid space
    15 ln(d / 3,000) more (at most 50) for the share 1 - 1 / (1 + d / 8,000), d the zone's jobs
    per km² of land (res-9 cells, a merged res-8 cell counts 7, at most the square)."""
    nz, zone = h["zones"], a["zone"].astype(np.int64)
    jobs = a["jobs"].astype(np.float64)
    res8 = ((a["h3"] >> np.uint64(52)) & np.uint64(0xF)) == 8
    ncell = np.bincount(zone, np.where(res8, 7.0, 1.0), nz)
    land = np.minimum(np.maximum(ncell, 1) * CELL_KM2, (h["gravity"]["zone_m"] / 1000.0) ** 2)
    zj = np.bincount(zone, jobs, nz)
    d = zj / land
    paid = np.minimum(PARK_PAID_MIN * np.log(np.maximum(d, PARK_PAID_D0) / PARK_PAID_D0), PARK_PAID_CAP_MIN)
    free = 1.0 / (1.0 + d / PARK_FREE_D)
    charged = np.zeros(nz)
    for charge, ring in CORDONS:
        inside = in_ring(ring, lon, lat) & (jobs > 0)
        charged += np.bincount(zone[inside], jobs[inside] * charge, nz)
    cordon = np.where(zj > 0, charged / np.maximum(zj, 1e-12), 0.0)
    return (PARK_MIN + cordon + (1.0 - free) * paid)[zone]


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
        park = park_by_cell(h, a, lon, lat)
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


def group(iso2):
    """T-463: one joint commute gravity over a country's registered packs (Japan's), apart from the
    Northeast: each pack's factors, data/work/<iso2>.joint_commutes.json, then their market packs."""
    ids = [c["id"] for c in REGISTRY["cities"] if CITIES.get(c["id"], {}).get("iso2") == iso2.upper()]
    work = ROOT / "data/work"
    # Japan's columns stall at about 8e-4 from step ~90, all in Okinawa's outlying islands (weakly
    # joined to anything; notes/T-463.md): the best step is kept, so 150 steps are as good as 500
    cargo_bin("joint_gravity", "--root", str(ROOT), "--threads", os.environ.get("OMP_NUM_THREADS", "3"), "--max-iters", "150",
              "--packs", ",".join(ids), "--out", str(work / f"{iso2.lower()}.joint_commutes.json"),
              "--report", str(work / f"{iso2.lower()}.joint_report.json"))
    return ids


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
    if "--group" in sys.argv:
        ids = group(sys.argv[sys.argv.index("--group") + 1])
        if "--no-market" not in sys.argv:
            import market_zones
            market_zones.main(ids)
        sys.exit()
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

