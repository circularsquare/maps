"""T-1541: run the access diagnosis (t1541_test.rs in the crate copy) for case:variant jobs, `jobs` at
a time. Fare weight from T-1521 round 3 (../artifacts/trainworld/T-1521/corridor/); rows from
t1541_rows.py and the main tree's registry, since the packs moved on after round 3.

    python trainworld/notes/T-1541/t1541_run.py <crate folder> <t1521 trainworld> <jobs> case:variant ...
"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(MAPS, "artifacts", "trainworld", "T-1541")
R3 = os.path.join(MAPS, "artifacts", "trainworld", "T-1521", "corridor")
CRATE, TW, JOBS = sys.argv[1], sys.argv[2], int(sys.argv[3])
EXE = max(glob.glob(os.path.join(CRATE, "sim", "target", "release", "deps", "trainworld_sim-*.exe")), key=os.path.getmtime)
fw = json.load(open(os.path.join(R3, "fare_w.json")))
FARE = f"{fw['built']},0.1293,0.008621,0,0.1293,20,0.3017,100,2.0259"
# the drive to a station as a ride: 15 + 2 x 1.33 x the drive + a 10 minute drop-off, each end
VARIANTS = {
    "built": {},
    "ride40": {"T1541_RIDE_MAX": "40000"},
    "ridew": {"T1541_RIDE_F": "1"},
    "ride40w": {"T1541_RIDE_MAX": "40000", "T1541_RIDE_F": "1"},
    "airride": {"T1541_AIRACC": "2.66,50"},
    "lines40": {"T1541_RIDE_LINES": "40"},
    "air133": {"T1541_AIRACC": "1.33,0"},
    "air2": {"T1541_AIRACC": "2,20"},
    "zair1": {"T1541_ZAIR": "1,0"},
    "zair133": {"T1541_ZAIR": "1.33,0"},
    "zride": {"T1541_ZAIR": "2.66,25"},
    "r40w_zair133": {"T1541_RIDE_MAX": "40000", "T1541_RIDE_F": "1", "T1541_ZAIR": "1.33,0"},
    "r40w_air133":{"T1541_RIDE_MAX": "40000", "T1541_RIDE_F": "1", "T1541_AIRACC": "1.33,0"},
}
os.makedirs(OUT, exist_ok=True)


def one(job):
    case, v = job.split(":")
    out = os.path.join(OUT, f"{case}_{v}.json")
    env = dict(os.environ, T1541_CASE=case, T1521_ROOT=TW, T1541_OUT=out, T1521_FARE=FARE, T1521_AIR="100,0.05",
               TW_ROWS=os.path.join(OUT, "rows_built.json"), TW_REGISTRY=os.path.join(MAPS, "trainworld", "app", "src", "game", "registry.json"),
               OMP_NUM_THREADS="1", **VARIANTS[v])
    try:
        with open(os.path.join(OUT, f"{case}_{v}.log"), "w") as log:
            subprocess.run([EXE, "t1541_access", "--ignored", "--nocapture", "--test-threads=1"], check=True, stdout=log, stderr=log, env=env)
        return f"{case} {v}: ok"
    except Exception as e:
        return f"FAILED {case} {v}: {e}"


with ThreadPoolExecutor(JOBS) as ex:
    for r in ex.map(one, sys.argv[4:]):
        print(r, flush=True)
print("ALL DONE", flush=True)
