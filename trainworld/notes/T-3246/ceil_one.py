"""T-3246: one city's ceilings one after the other, on one core. python ceil_one.py <city>"""
import glob
import os
import subprocess
import sys

S = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
EXE = sorted(glob.glob(f"{S}/target-proto/release/deps/trainworld_sim-*.exe"), key=os.path.getmtime)[-1]
G = {"nyc": "0.2,1.76,0.2,0.58", "hou": "0.2,1.76,0.2,0.58", "lon": "0.2,1.52,0.2,0.66", "par": "0.2,1.51,0.2,0.66", "tyo": "0.2,1.62,0.2,0.67"}
c = sys.argv[1]
for v in ("one", "groups"):
    out = f"{S}/ceil_{c}_{v}.json"
    if os.path.exists(out):
        continue
    env = dict(os.environ, T938_PACK=f"{TW}/data/packs/{c}.json", T938_OUT=out, RAYON_NUM_THREADS="1")
    if v == "groups":
        env["T3246_G"] = G[c]
    r = subprocess.run([EXE, "demand::everyday::t3246_ceiling::t1056", "--ignored", "--exact", "--test-threads", "1"], env=env,
                       capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    print(c, v, "rc", r.returncode, r.stderr[-300:] if r.returncode else "", flush=True)
print("done", flush=True)
