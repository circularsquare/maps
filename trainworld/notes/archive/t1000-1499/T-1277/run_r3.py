"""T-1277 round 3: the built calib (worktree) at each city's fare weight, fares as charged there."""
import json
import os
import subprocess
import sys

TREE = "C:/Users/anita/projects/maps/.tw-trees/t1277/trainworld"
CALIB = f"{TREE}/sim/target/release/calib.exe"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out3")
os.makedirs(OUT, exist_ok=True)
W = {"nyc": 4.4216, "lon": 5.2724}
mults = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 3, 4, 5, 6]
for c in ("nyc", "lon"):
    for m in mults:
        out = f"{OUT}/{c}_{m:g}.json"
        if not os.path.exists(out):
            env = dict(os.environ, TW_ROOT=TREE, RAYON_NUM_THREADS="2")
            r = subprocess.run([CALIB, "--net", f"{TREE}/data/work/realnet/{c}_real.json", "--out", out, "--fares", f"{1.5 * m},{0.1 * m}", "--set", f"fare_w={W[c]}"], env=env, capture_output=True, text=True)
            if r.returncode:
                print(c, m, r.stderr[-400:])
                continue
        j = json.load(open(out))
        cc, e = j["day"]["commute"], j["day"]["everyday"]
        rail = cc[1] + e[1]
        pkm = sum(j["lines"]["pkm"])
        rev = 1.5 * m * rail + 0.1 * m * pkm
        print(f"{c} x{m:g}: rail {rail / 1e6:.2f}M, rail of rail+car {100 * rail / (cc[0] + e[0] - cc[2] - e[2]):.1f}%, mean fare ${rev / rail:.2f}, revenue ${rev / 1e6:.1f}M a real day", flush=True)
