"""T-1277 runs: python run.py ceil|real cities variants [mults]. Outputs in out/, skipped if present.
Variants: base (the game), fwN (fares weigh N times), evN (everyday fares weigh N times),
powN (ratio curve power N; harness only)."""
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

D = os.path.dirname(os.path.abspath(__file__))
TREE = "C:/Users/anita/projects/maps/trainworld"
BIN = f"{D}/target/release"
OUT = f"{D}/out"
MULTS = [0.333, 0.667, 1, 1.5, 2, 3, 4, 6, 8, 10, 13, 20]
what = sys.argv[1]
cities = sys.argv[2].split(",")
variants = sys.argv[3].split(",")
mults = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else MULTS


def weights(v):
    if v.startswith("fw"):
        w = float(v[2:])
        return w, w, 3.0
    if v.startswith("ev"):
        return 1.0, float(v[2:]), 3.0
    if v.startswith("pow"):
        return 1.0, 1.0, float(v[3:])
    return 1.0, 1.0, 3.0


def one(job):
    c, v, m = job
    fw, fwe, pw = weights(v)
    t = time.time()
    if what == "ceil":
        out = f"{OUT}/ceil_{c}_{v}.json"
        if os.path.exists(out):
            return f"{c} {v} cached"
        test = max(glob.glob(f"{BIN}/deps/trainworld_sim-*.exe"), key=os.path.getmtime)
        env = dict(os.environ, T1277_PACK=f"{TREE}/data/packs/{c}.json", T1277_OUT=out, T1277_FW=str(fw), T1277_FWE=str(fwe), T1277_POW=str(pw), RAYON_NUM_THREADS="1")
        cmd = [test, "t1277_ceiling::t1277", "--ignored", "--test-threads", "1"]
    else:
        out = f"{OUT}/real_{c}_{v}_{m:g}.json"
        if os.path.exists(out):
            return f"{c} {v} {m} cached"
        env = dict(os.environ, TW_ROOT=TREE, RAYON_NUM_THREADS="2")
        b, k = 1.5 * m, 0.10 * m
        cmd = [f"{BIN}/calib.exe", "--net", f"{TREE}/data/work/realnet/{c}_real.json", "--out", out, "--fares", f"{b * fw:.4f},{k * fw:.5f}"]
        if fwe != fw:
            cmd += ["--fares-ev", f"{b * fwe:.4f},{k * fwe:.5f}"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = (r.stderr.strip().splitlines() or [""])[-2] if what == "real" and r.returncode == 0 else (r.stdout + r.stderr).strip()[-600:] if r.returncode else ""
    return f"{c} {v} {m} {time.time() - t:.0f}s rc {r.returncode} {tail}"


os.makedirs(OUT, exist_ok=True)
if what == "ceil":
    jobs = [(c, v, 0) for c in cities for v in variants]
    workers = 2
else:
    jobs = [(c, v, m) for v in variants for c in cities for m in mults]
    workers = 1
with ThreadPoolExecutor(workers) as ex:
    for line in ex.map(one, jobs):
        print(line, flush=True)
