"""T-1092 runs: python run.py meas|ceil|real [cities] [variants]. Before = the central draw off and
T-637's scales; after = the defaults. Outputs in out/, skipped if present."""
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

D = os.path.dirname(os.path.abspath(__file__))
TREE = "C:/Users/anita/projects/maps/.tw-trees/t1092/trainworld"
SIM = f"{TREE}/sim/target/release"
OUT = f"{D}/out"
SETS = {"before": "central_share=0,decay_errands=1300,decay_social=1400", "after": ""}
what = sys.argv[1]
cities = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nyc", "lon", "tyo"]
variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["before", "after"]
tests = [p for p in glob.glob(f"{SIM}/deps/trainworld_sim-*.exe")]
TEST = max(tests, key=os.path.getmtime)


def one(job):
    c, v = job
    out = f"{OUT}/{what}_{c}_{v}.json"
    if os.path.exists(out):
        return f"{c} {v} cached"
    pack = f"{TREE}/data/packs/{c}.json"
    t = time.time()
    if what == "meas":
        env = dict(os.environ, T1092_PACK=pack, T1092_OUT=out, T1092_SET=SETS[v], RAYON_NUM_THREADS="1")
        cmd = [TEST, "t1092_meas::t1092_meas", "--ignored", "--test-threads", "1"]
    elif what == "ceil":
        env = dict(os.environ, T938_PACK=pack, T938_OUT=out, T938_SET=SETS[v], RAYON_NUM_THREADS="1")
        cmd = [TEST, "t1056_ceiling::t1056", "--ignored", "--test-threads", "1"]
    else:
        env = dict(os.environ, TW_ROOT=TREE, RAYON_NUM_THREADS="3")
        cmd = [f"{SIM}/calib.exe", "--net", f"{TREE}/data/work/realnet/{c}_real.json", "--out", out]
        if SETS[v]:
            cmd += ["--set", SETS[v]]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = (r.stderr.strip().splitlines() or [""])[-1] if what == "real" else r.stdout.strip()[-400:] if r.returncode else ""
    return f"{c} {v} {time.time() - t:.0f}s rc {r.returncode} {tail}"


os.makedirs(OUT, exist_ok=True)
jobs = [(c, v) for c in cities for v in variants]
workers = 1 if what == "real" else 3
with ThreadPoolExecutor(workers) as ex:
    for line in ex.map(one, jobs):
        print(line, flush=True)
