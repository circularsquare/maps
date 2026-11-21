"""T-3246: calib runs in parallel. python runs.py <name>=<exe>|<args> ... (args split on spaces)."""
import os
import subprocess
import sys
import time

S = os.path.dirname(os.path.abspath(__file__))
MAPS = "C:/Users/anita/projects/maps"
RN = MAPS + "/trainworld/data/work/realnet"
EXE = {"base": S + "/target-base/release/calib.exe", "proto": S + "/target-proto/release/calib.exe"}
CITY = {
    "nyc": ["--net", RN + "/nyc_real.json", "--areas", RN + "/cell_county.u8"],
    "tyo": ["--net", RN + "/tyo_real.json", "--areas", RN + "/tyo_cell_muni.u16"],
}
GAME = {
    "@game": ["--set", "fare_w=4.4", "--curve", "0,1.5,20,3.5,1000,248.5"],
    "@x2": ["--set", "fare_w=4.4", "--fares", "3,0.2", "--curve", "0,3,20,7,1000,497"],
    "@jp": ["--set", "fare_w=6.0", "--fares", "0.9,0.06", "--curve", "0,0.9,20,2.1,1000,149.1", "--set", "t3246_lo_f=1.62,t3246_hi_f=0.67"],
}

jobs = []
for spec in sys.argv[1:]:
    name, rest = spec.split("=", 1)
    exe, city, extra = rest.split("|")
    jobs.append((name, EXE[exe], CITY[city], extra.split() if extra else []))
n = len(jobs)
r = subprocess.run([sys.executable, MAPS + "/cores.py", "claim", str(n), "incomes", "--pid", str(os.getpid()), "T-3246 calib runs"],
                   capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1], flush=True)
env = dict(os.environ, TW_ROOT=MAPS + "/.tw-trees/incomes/trainworld", RAYON_NUM_THREADS="1")
procs = []
for name, exe, city, extra in jobs:
    # `@game` expands to the game's fares for that city
    ex = []
    for a in extra:
        ex += GAME[a] if a in GAME else [a]
    sets = [a for a in ex]
    # merge several --set into one
    merged, setv = [], []
    i = 0
    while i < len(sets):
        if sets[i] == "--set":
            setv.append(sets[i + 1])
            i += 2
        else:
            merged.append(sets[i])
            i += 1
    if setv:
        merged += ["--set", ",".join(setv)]
    cmd = [exe] + city + ["--out", f"{S}/{name}.json", "--rounds", "3", "--cells", "--exact"] + merged
    log = open(f"{S}/{name}.log", "w")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=log, env=env, creationflags=subprocess.CREATE_NO_WINDOW), time.time()))
for name, p, t in procs:
    p.wait()
    print(name, "rc", p.returncode, f"{time.time() - t:.0f}s", flush=True)
subprocess.run([sys.executable, MAPS + "/cores.py", "release", "incomes"], capture_output=True)
print("done", flush=True)
