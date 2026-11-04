"""T-1147: calib on the real networks with speed variants, outputs in this folder (never data/).
python -I run_real.py <variant,...> <cities>; variants in run_real_vars.py."""
import json
import os
import subprocess
import sys
import time

TREE = "C:/Users/anita/projects/maps/.tw-trees/t1147/trainworld"
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = f"{TREE}/sim/target/release/calib.exe"
sys.path.insert(0, f"{TREE}/pipeline")
sys.path.insert(0, HERE)
os.environ["OMP_NUM_THREADS"] = "1"
import calib  # noqa: E402
from run_real_vars import VARS  # noqa: E402

THREADS = os.environ.get("T1147_THREADS", "2")
for var in sys.argv[1].split(","):
    s = ",".join(f"{k}={v:.4f}" for k, v in VARS[var].items())
    tagdir = os.path.join(HERE, "real", var.replace(" ", "_").replace("+", "_"))
    os.makedirs(tagdir, exist_ok=True)
    for city in sys.argv[2].split(","):
        tg = json.load(open(os.path.join(calib.CALIB, city, "targets.json")))
        out = os.path.join(tagdir, f"run_{city}.json")
        if not os.path.exists(out):
            cmd = [BIN, "--net", tg["net"], "--out", out, "--rounds", "3"]
            if tg.get("pack"):
                cmd += ["--pack", tg["pack"]]
            # New York's cell_county.u8 predates its pack's rebuild (61,427 against 54,441 cells)
            if tg.get("areas_file") and city not in ("nyc", "tyo"):
                cmd += ["--areas", tg["areas_file"]]
            if s:
                cmd += ["--set", s]
            t = time.time()
            env = dict(os.environ, RAYON_NUM_THREADS=THREADS)
            with open(os.path.join(tagdir, f"run_{city}.log"), "w") as log:
                subprocess.run(cmd, check=True, stderr=log, stdout=log, env=env)
            print(var, city, f"{time.time() - t:.0f}s", flush=True)
        if city in ("nyc", "tyo"):
            tg = dict(tg, areas=[])
        with open(os.path.join(tagdir, f"score_{city}.txt"), "w", encoding="utf-8") as f:
            sc = calib.score(json.load(open(out)), tg, say=lambda x: f.write(x + "\n"))
        json.dump({k: sc.get(k) for k in ("total", "parts", "station_level", "station_spread", "n_stations", "operators", "day", "everyday_rows", "rail_share_model", "rail_share_measured", "walk_areas")},
                  open(os.path.join(tagdir, f"score_{city}.json"), "w"))
