"""Runs the t636 test and calib.py run for every (city, setting); 5 at a time."""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

S = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
PY = "C:/Users/anita/projects/maps/venv/Scripts/python.exe"
EXE = [f for f in os.listdir(f"{S}/target/release/deps") if f.startswith("trainworld_sim-") and f.endswith(".exe")]
EXE = max((f"{S}/target/release/deps/{f}" for f in EXE), key=os.path.getmtime)
SIMPLE = "car_own_usd_trip=1.5,car_own_usd_km=0.1,asc_rail=0,everyday_asc_rail=0,country_asc_rail=0,everyday_asc_walk=0"
SETS = {
    "today": "",
    "simple": SIMPLE,
    "b075": SIMPLE + ",beta=0.075",
    "b100": SIMPLE + ",beta=0.1",
    "d3a": SIMPLE + ",beta=0.075,damp_min=26.67",
    "d3b": SIMPLE + ",beta=0.1,damp_min=15",
}
CITIES = ["nyc", "lon"]


def job(args):
    kind, city, name = args
    s = SETS[name]
    if kind == "test":
        env = dict(os.environ, T636_CITY=city, T636_OUT=f"{S}/out/{city}_{name}.json", T636_SET=s, T636_DATA=f"{TW}/data")
        r = subprocess.run([EXE, "--ignored", "--nocapture", "t636"], env=env, capture_output=True, text=True)
    else:
        env = dict(os.environ, CALIB_BIN=f"{S}/target/release/calib.exe", TW_ROOT=TW, OMP_NUM_THREADS="2")
        cmd = [PY, f"{TW}/pipeline/calib.py", "run", city, "--tag", f"t636_{name}"]
        if s:
            cmd += ["--set", s]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=f"{TW}/..")
        open(f"{S}/out/{city}_{name}.calib.txt", "w").write(r.stdout + r.stderr)
    print(kind, city, name, r.returncode, flush=True)
    return r.returncode


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["test", "calib"]
    jobs = [(k, c, n) for k in kinds for c in CITIES for n in SETS]
    with ThreadPoolExecutor(5) as ex:
        list(ex.map(job, jobs))
