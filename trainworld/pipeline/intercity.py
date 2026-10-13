"""T-117, T-276: the regional market's pack, data/packs/intercity.json.

Endpoint sites from every registered zone pack's cells (600 m squares; 6 km zones form the market's
gravity), each site's parking for an arriving car (the local layer's rule), and the outside world:
everyone living in no zone (pipeline/outside.py, Kontur), so a share of each zone's longer trips
leaves the map. The trip rate and the distance curve are fitted to NHTS 2017
(research/intercity/market_fit.py, docs/zones.md 5). Mercer County is in New York's pack since
T-276; its market place (id 2) keeps its name and has no sites.

    python pipeline/outside.py              first, when a zone is added (the outside world)
    python pipeline/intercity.py            sites from the city packs, the joint commute gravity
                                            (rewrites every zone pack's gravity, T-276), then
                                            the derived block; --no-joint skips the gravity
    python pipeline/intercity.py --derive   only the derived block, on the pack as it is

The derived block (T-174) holds the figures that need every zone pair (each zone's gravity
normaliser, the day's trips, each corridor's), computed by the sim crate's own market code
(`market_pack`, seconds), so the game never visits all pairs when it opens the market. Run
`--derive` after changing the market's zone-pair formula in sim/src/demand/intercity.rs (its
`FORMULA`); a pack whose block is stale still opens, deriving it in every worker (slow).

The sites' x and y are metres in the New York equirectangular frame (`origin`). The pack names
that frame (`frame`, T-128), and the game takes zone distances and drive times through it on the
ground (great circles) instead of straight in the plane. `--derive` adds `frame` to a pack written
before T-128 (its sites are in the same frame); without it the game keeps the plane distances.
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


def outside_points(site_rows):
    """The outside world (pipeline/outside.py) in the sites' frame, each point's people times the
    represented places' attraction per person (people + 2 x jobs, over people)."""
    src = json.loads((ROOT / "data/work/outside.json").read_text(encoding="utf-8"))
    zones = {c["id"] for c in REGISTRY["cities"] if c["market"] >= 0}
    if set(src["zones"]) != zones:
        sys.exit(f"data/work/outside.json was made for zones {sorted(src['zones'])}, the registry has {sorted(zones)}: run pipeline/outside.py")
    s = np.array(site_rows, float)
    alpha = (s[:, 2] + 2 * s[:, 3]).sum() / s[:, 2].sum()
    p = np.array(src["points"], float)
    x, y = lonlat_to_market(p[:, 0], p[:, 1])
    return [[round(float(a), 0), round(float(b), 0), round(float(alpha * w), 1)] for a, b, w in zip(x, y, p[:, 2])], alpha

def joint():
    """T-276: one commute gravity over every zone pack (`joint_gravity`): rewrites each pack's
    gravity factors and the market pack's commutes between zones. About two minutes."""
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


def derive(path):
    """Write the pack's derived block with the sim crate's `market_pack` (T-174)."""
    env = dict(os.environ)
    env["PATH"] = os.path.join(os.path.expanduser("~"), ".cargo", "bin") + os.pathsep + env.get("PATH", "")
    env.setdefault("CARGO_BUILD_JOBS", "2")
    # build_city.py's target dir: the pipeline never waits on (or blocks) the app's wasm builds
    env["CARGO_TARGET_DIR"] = str(ROOT / "data/work/target")
    cargo = shutil.which("cargo", path=env["PATH"])
    if not cargo:
        sys.exit("cargo not found: install Rust (rustup) or put ~/.cargo/bin on PATH")
    subprocess.run([cargo, "run", "--quiet", "--release", "--features", "demand", "--bin", "market_pack",
                    "--manifest-path", str(ROOT / "sim/Cargo.toml"), "--", str(path)], env=env, check=True)


def frame():
    """The frame the sites are in (T-128), as the registry writes frames."""
    return {"kind": "equirect", "lon": ORIGIN[0], "lat": ORIGIN[1]}


def name_frame(path):
    """Give a pack written before T-128 its `frame` (and format 2); the block is rewritten after."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "frame" in data:
        return
    o = data.get("origin", {"lon": ORIGIN[0], "lat": ORIGIN[1]})
    if (o["lon"], o["lat"]) != tuple(ORIGIN):
        sys.exit(f"{path}: sites at origin {o}, not {ORIGIN}: rebuild with pipeline/intercity.py")
    out = {}
    for k, v in data.items():
        if k == "derived":
            continue
        out[k] = 2 if k == "format" else v
        if k == "origin":
            out["frame"] = frame()
    out.setdefault("frame", frame())
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(path, "now names its frame", out["frame"])


if __name__ == "__main__":
    path = ROOT / "data/packs/intercity.json"
    if "--derive" in sys.argv[1:]:
        name_frame(path)
    else:
        site_rows, site_park = sites()
        outside, alpha = outside_points(site_rows)
        data = {"format": 2, "origin": {"lon": ORIGIN[0], "lat": ORIGIN[1]}, "frame": frame(),
                "markets": MARKETS,
                "zone_m": 6000, "trip_rate": TRIP_RATE,
                "description": f"{TRIP_RATE} weekday non-commute departures per resident (NHTS 2017, Northeast, fitted with the distance curve: research/intercity/market_fit.py), shared among every zone and the outside world (attraction {alpha:.3f} per outside person).",
                "sources": ["2020 Census block population less institutional group quarters (T-234)", "LODES 2023/2022 jobs", "registered zone-pack cells",
                            "NHTS 2017 trips by distance", "Kontur Population 2023 (the outside world)"],
                "sites": site_rows, "site_park": site_park, "outside": outside}
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(path, len(site_rows), "sites", len(outside), "outside points", path.stat().st_size, "bytes")
        if "--no-joint" not in sys.argv[1:]:
            joint()
    derive(path)

