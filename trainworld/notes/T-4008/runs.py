"""T-4008: calib runs, in parallel up to the claim. python runs.py OUTDIR name=exe|extra ... (extra split on spaces;
@game expands to New York's starting fares as T-3246 ran them). Each run's summary lines go to OUTDIR/<name>.log."""
import os
import subprocess
import sys
import time

MAPS = "C:/Users/anita/projects/maps"
TREE = MAPS + "/.tw-trees/timegroups/trainworld"
SCR = "C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/d15915ac-6dd5-4586-bd98-843cbc616824/scratchpad/timegroups"
EXE = {"base": SCR + "/target-base/release/calib.exe", "proto": SCR + "/target-proto/release/calib.exe"}
RN = MAPS + "/trainworld/data/work/realnet"
NYC = ["--net", RN + "/nyc_real.json"]
GAME = ["--curve", "0,1.5,20,3.5,1000,248.5"]
SETS = {"@game": "fare_w=4.4"}

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
jobs = []
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    exe, extra = rest.split("|")
    args, sets = [], []
    for a in extra.split():
        if a == "@game":
            args += GAME
            sets.append(SETS["@game"])
        else:
            args.append(a)
    if sets:
        args += ["--set", ",".join(sets)]
    jobs.append((name, EXE[exe], args))
r = subprocess.run([sys.executable, MAPS + "/cores.py", "claim", str(len(jobs)), "timegroups", "--pid", str(os.getpid()), "T-4008 calib runs"],
                   capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1], flush=True)
env = dict(os.environ, TW_ROOT=TREE, RAYON_NUM_THREADS="1")
procs = []
for name, exe, args in jobs:
    cmd = [exe] + NYC + ["--out", f"{out}/{name}.json", "--rounds", "3", "--exact", "--cells"] + args
    log = open(f"{out}/{name}.log", "w")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=log, env=env, creationflags=subprocess.CREATE_NO_WINDOW), time.time()))
for name, p, t in procs:
    p.wait()
    print(name, "rc", p.returncode, f"{time.time() - t:.0f}s", flush=True)
subprocess.run([sys.executable, MAPS + "/cores.py", "release", "timegroups"], capture_output=True)
print("done", flush=True)
