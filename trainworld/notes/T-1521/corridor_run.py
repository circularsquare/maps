"""T-1521 round 3: run the corridor cases (corridor_test.rs in the crate copy) for each variant, `jobs`
at a time, and print the table.

    python trainworld/notes/T-1521/corridor_run.py <crate folder> <jobs> [variant,...] [case,...]
"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.abspath(os.path.join(TW, "..", "artifacts", "trainworld", "T-1521", "corridor"))
CRATE, JOBS = sys.argv[1], int(sys.argv[2])
EXE = max(glob.glob(os.path.join(CRATE, "sim", "target", "release", "deps", "trainworld_sim-*.exe")), key=os.path.getmtime)
fw = json.load(open(os.path.join(OUT, "fare_w.json")))
NEW = f"{fw['built']},0.1293,0.008621,0,0.1293,20,0.3017,100,2.0259"
VARIANTS = {
    "before": dict(rows="before", air="30,0.05", fare=f"{fw['before']},0.5357,0.03571"),
    "built": dict(rows="built", air="100,0.05", fare=NEW),
    "plr": dict(rows="built", air="162,0.045", fare=NEW, af=True),
    "upkeep": dict(rows="upkeep", air="100,0.05", fare=NEW),
}
CASES = ["china", "eu", "nec", "jpn"]
vs = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else list(VARIANTS)
cs = sys.argv[4].split(",") if len(sys.argv) > 4 else CASES


def one(job):
    v, case = job
    cfg = VARIANTS[v]
    out = os.path.join(OUT, f"{case}_{v}.json")
    env = dict(os.environ, T1521_CASE=case, T1521_ROOT=TW, T1521_OUT=out, T1521_FARE=cfg["fare"], T1521_AIR=cfg["air"],
               TW_ROWS=os.path.join(OUT, f"rows_{cfg['rows']}.json"), OMP_NUM_THREADS="1")
    if cfg.get("af"):
        env["T1521_AF"] = open(os.path.join(OUT, "af_plr.json")).read()
    try:
        with open(os.path.join(OUT, f"{case}_{v}.log"), "w") as log:
            subprocess.run([EXE, "t1521_corridors", "--ignored", "--nocapture", "--test-threads=1"], check=True, stdout=log, stderr=log, env=env)
        return f"{case} {v}: solve {json.load(open(out))['solve_ms'] / 1e3:.0f} s"
    except Exception as e:
        return f"FAILED {case} {v}: {e}"


with ThreadPoolExecutor(JOBS) as ex:
    for r in ex.map(one, [(v, c) for c in cs for v in vs]):
        print(r, flush=True)
print("ALL DONE", flush=True)
