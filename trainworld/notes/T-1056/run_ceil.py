"""T-1056: Anita's ceilings A and B with and without the walking rule. python run_ceil.py [cities]"""
import glob
import json
import os
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
EXE = sorted(glob.glob(f"{D}/meas/sim/target/release/deps/trainworld_sim-*.exe"), key=os.path.getmtime)[-1]
OUT = f"{D}/ceil"
os.makedirs(OUT, exist_ok=True)
cities = sys.argv[1].split(",") if len(sys.argv) > 1 else ["nyc", "lon", "par", "tyo", "hou"]
for c in cities:
    out = f"{OUT}/{c}.json"
    if not os.path.exists(out):
        env = dict(os.environ, T938_PACK=f"{TW}/data/packs/{c}.json", T938_OUT=out, RAYON_NUM_THREADS="1")
        t = time.time()
        r = subprocess.run([EXE, "t1056_ceiling::t1056", "--ignored", "--test-threads", "1"], env=env, capture_output=True, text=True)
        print(c, f"{time.time() - t:.0f}s rc {r.returncode}", r.stderr.strip()[-300:] if r.returncode else "", flush=True)
    j = json.load(open(out))
    row = []
    for s in ("A", "A1056", "B", "B1056"):
        a = j["acc"][j["scens"].index(s) * 5 + 4]
        row.append(f"{s} {100 * a[1] / (a[1] + a[3]):.1f}")
    print(c, *row, flush=True)
