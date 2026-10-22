"""T-938: the ceiling tests A and B for each setting and city; skips outputs that exist.

    python run_ceil.py <settings,..> [cities]
"""
import concurrent.futures as cf
import glob
import json
import os
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
EXE = sorted(glob.glob(f"{D}/copy/sim/target/release/deps/trainworld_sim-*.exe"), key=os.path.getmtime)[-1]
SPEED = "road_city_kmh=16.5,road_town_kmh=44,road_far_kmh=88,road_open_kmh=110"
OWN = "car_own_usd_trip=2.0,car_own_usd_km=0.13"
SETS = {
    "before": "",
    "speed": SPEED,
    "own": OWN,
    "after": SPEED + "," + OWN,
}
for k in list(SETS):
    pass
OUT = f"{D}/ceil"
os.makedirs(OUT, exist_ok=True)
names = sys.argv[1].split(",")
cities = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nyc", "lon", "par", "tyo", "hou"]


def one(c, v):
    out = f"{OUT}/{c}_{v}.json"
    if os.path.exists(out):
        return c, v, "cached"
    env = dict(os.environ, T938_PACK=f"{TW}/data/packs/{c}.json", T938_OUT=out, T938_SET=SETS.get(v, v), RAYON_NUM_THREADS="1")
    t = time.time()
    r = subprocess.run([EXE, "t938_ceiling::t938", "--ignored", "--test-threads", "1"], env=env, capture_output=True, text=True)
    return c, v, f"{time.time() - t:.0f}s rc {r.returncode} {r.stderr.strip()[-300:] if r.returncode else ''}"


def share(path, scen):
    j = json.load(open(path))
    s = j["scens"].index(scen)
    a = j["acc"][s * 5 + 4]
    return 100 * a[1] / (a[1] + a[3])


with cf.ThreadPoolExecutor(2) as ex:
    for res in ex.map(lambda cv: one(*cv), [(c, v) for c in cities for v in names]):
        print(*res, flush=True)
print("city setting A B T")
for c in cities:
    for v in names:
        p = f"{OUT}/{c}_{v}.json"
        if os.path.exists(p):
            print(c, v, *(f"{share(p, s):.1f}" for s in ("A", "B", "T")))
