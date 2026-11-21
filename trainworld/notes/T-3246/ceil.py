"""T-3246: Anita's ceilings A and B, one income group and with groups. python ceil.py [cities]
Runs the scratch test everyday::t3246_ceiling::t1056 (a copy of checks/ceiling/t1056_ceiling.rs with groups)."""
import glob
import json
import os
import subprocess
import sys

S = os.path.dirname(os.path.abspath(__file__))
MAPS = "C:/Users/anita/projects/maps"
TW = MAPS + "/trainworld"
EXE = sorted(glob.glob(f"{S}/target-proto/release/deps/trainworld_sim-*.exe"), key=os.path.getmtime)[-1]
cities = sys.argv[1].split(",") if len(sys.argv) > 1 else ["nyc", "lon", "par", "tyo", "hou"]
# low share, low factor, high share, high factor (on today's weight), from each country's quintiles
G = {"nyc": "0.2,1.76,0.2,0.58", "hou": "0.2,1.76,0.2,0.58", "lon": "0.2,1.52,0.2,0.66", "par": "0.2,1.51,0.2,0.66", "tyo": "0.2,1.62,0.2,0.67"}
variants = {"one": None, "groups": "G"}
r = subprocess.run([sys.executable, MAPS + "/cores.py", "claim", "3", "incomes-ceil", "--pid", str(os.getpid()), "T-3246 ceilings"], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1], flush=True)
procs = []
for c in cities:
    for v, g in variants.items():
        out = f"{S}/ceil_{c}_{v}.json"
        if os.path.exists(out):
            continue
        env = dict(os.environ, T938_PACK=f"{TW}/data/packs/{c}.json", T938_OUT=out, RAYON_NUM_THREADS="1")
        if g:
            env["T3246_G"] = G[c]
        procs.append((c, v, subprocess.Popen([EXE, "demand::everyday::t3246_ceiling::t1056", "--ignored", "--exact", "--test-threads", "1"], env=env,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)))
        if len(procs) >= 3:
            c0, v0, p0 = procs.pop(0)
            err = p0.communicate()[1]
            print(c0, v0, "rc", p0.returncode, err[-300:] if p0.returncode else "", flush=True)
for c0, v0, p0 in procs:
    err = p0.communicate()[1]
    print(c0, v0, "rc", p0.returncode, err[-300:] if p0.returncode else "", flush=True)
subprocess.run([sys.executable, MAPS + "/cores.py", "release", "incomes-ceil"], capture_output=True)
for c in cities:
    row = [c]
    for v in variants:
        j = json.load(open(f"{S}/ceil_{c}_{v}.json"))
        for s in ("A1056", "B1056"):
            a = j["acc"][j["scens"].index(s) * 5 + 4]
            row.append(f"{v} {s[0]} {100 * a[1] / (a[1] + a[3]):.1f}")
    print(*row, flush=True)
print("done", flush=True)
