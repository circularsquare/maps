"""T-938: the real networks (calib, native) before and after; skips outputs that exist."""
import json
import os
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
EXE = f"{D}/copy/sim/target/release/calib.exe"
TW = "C:/Users/anita/projects/maps/trainworld"
OUT = f"{D}/real"
SPEED = "road_city_kmh=16.5,road_town_kmh=44,road_far_kmh=88,road_open_kmh=110"
OWN = "car_own_usd_trip=2.0,car_own_usd_km=0.13"
VARS = {"before": None, "speed": SPEED, "own": OWN, "after": SPEED + "," + OWN}
os.makedirs(OUT, exist_ok=True)
cities = sys.argv[1].split(",")
names = sys.argv[2].split(",")
env = dict(os.environ, TW_ROOT=TW)
for c in cities:
    for v in names:
        out = f"{OUT}/{c}_{v}.json"
        if not os.path.exists(out):
            cmd = [EXE, "--net", f"{TW}/data/work/realnet/{c}_real.json", "--out", out]
            if VARS.get(v, v):
                cmd += ["--set", VARS.get(v, v)]
            t = time.time()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            print(c, v, f"{time.time() - t:.0f}s", r.returncode, r.stderr.strip().splitlines()[-3:] if r.returncode else "", flush=True)
        j = json.load(open(out))
        s = j.get("summary", j)
        print(c, v, {k: s[k] for k in s if isinstance(s[k], (int, float))} if isinstance(s, dict) else "", flush=True)
