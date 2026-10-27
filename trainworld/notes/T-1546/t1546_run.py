"""T-1546: market rows as built (the main tree's market-rows.json plus T-1521's `fw`, `fg`, `cu` per
country, as t1541_rows.py), then the dump test (t1546_test.rs in the crate copy) for each case, `jobs`
at a time. Outputs in ../artifacts/trainworld/T-1546/<case>.json and .log.

    python trainworld/notes/T-1546/t1546_run.py <crate folder> <jobs> case ...
"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.abspath(os.path.join(HERE, "..", ".."))
MAPS = os.path.dirname(TW)
OUT = os.path.join(MAPS, "artifacts", "trainworld", "T-1546")
R3 = os.path.join(MAPS, "artifacts", "trainworld", "T-1521", "corridor")
CRATE, JOBS = sys.argv[1], int(sys.argv[2])
EXE = max(glob.glob(os.path.join(CRATE, "sim", "target", "release", "deps", "trainworld_sim-*.exe")), key=os.path.getmtime)
os.makedirs(OUT, exist_ok=True)

rows = json.load(open(os.path.join(TW, "app", "src", "game", "market-rows.json"), encoding="utf-8"))
built = json.load(open(os.path.join(R3, "rows_built.json"), encoding="utf-8"))
by_c = {}
for r in built.values():
    by_c.setdefault(r["country"], {k: r[k] for k in ("fw", "fg", "cu") if k in r})
missing = sorted({r["country"] for r in rows.values() if r["country"] not in by_c})
for r in rows.values():
    r.update(by_c.get(r["country"], {}))
ROWS = os.path.join(OUT, "rows_built.json")
json.dump(rows, open(ROWS, "w"))
print(len(rows), "rows; countries with no built fields:", missing, flush=True)

fw = json.load(open(os.path.join(R3, "fare_w.json")))
FARE = f"{fw['built']},0.1293,0.008621,0,0.1293,20,0.3017,100,2.0259"


def one(case):
    out = os.path.join(OUT, f"{case}.json")
    env = dict(os.environ, T1546_CASE=case, T1521_ROOT=TW, T1546_OUT=out, T1521_FARE=FARE, TW_ROWS=ROWS,
               TW_REGISTRY=os.path.join(TW, "app", "src", "game", "registry.json"), OMP_NUM_THREADS="1")
    try:
        with open(os.path.join(OUT, f"{case}.log"), "w") as log:
            subprocess.run([EXE, "t1546_dump", "--ignored", "--nocapture", "--test-threads=1"], check=True, stdout=log, stderr=log, env=env)
        return f"{case}: ok"
    except Exception as e:
        return f"FAILED {case}: {e}"


with ThreadPoolExecutor(JOBS) as ex:
    for r in ex.map(one, sys.argv[3:]):
        print(r, flush=True)
print("ALL DONE", flush=True)
