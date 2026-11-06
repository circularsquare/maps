"""T-2668 farescale: calib on test lines at a sweep of fare multiples; claims cores first.

python run.py <tag> <net json> <fare_w> <base> <per_km> <mults comma list>
Writes out/<tag>_<m>.json and prints riders, pkm, fare income a real day.
"""
import json
import os
import subprocess
import sys

MAPS = "C:/Users/anita/projects/maps"
TW = f"{MAPS}/trainworld"
CALIB = f"{TW}/sim/target/release/calib.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

tag, net, fw, base, km, mults = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), [float(x) for x in sys.argv[6].split(",")]
name = f"farescale-{tag}"
r = subprocess.run([sys.executable, f"{MAPS}/cores.py", "claim", "1", name, "--pid", str(os.getpid()), "T-2668 calib fare sweep"], capture_output=True, text=True)
grant = (r.stdout.strip().splitlines() or ["1"])[-1]
print("cores granted:", grant, flush=True)
threads = str(max(1, int(grant)) if grant.isdigit() else 1)
try:
    for m in mults:
        out = f"{OUT}/{tag}_{m:g}.json"
        if not os.path.exists(out):
            env = dict(os.environ, TW_ROOT=TW, RAYON_NUM_THREADS=threads, OMP_NUM_THREADS=threads)
            p = subprocess.run([CALIB, "--net", net, "--out", out, "--fares", f"{base * m},{km * m}", "--set", f"fare_w={fw}"], env=env, capture_output=True, text=True)
            if p.returncode:
                print(tag, m, p.stderr[-800:], flush=True)
                continue
        j = json.load(open(out))
        cc, e = j["day"]["commute"], j["day"]["everyday"]
        rail = cc[1] + e[1]
        pkm = sum(j["lines"]["pkm"])
        board = sum(j["lines"]["board"])
        rev = base * m * rail + km * m * pkm
        print(f"{tag} x{m:g}: rail trips {rail:.0f}, boardings {board:.0f}, pkm {pkm:.0f} (mean {pkm / max(rail, 1):.1f} km), fare income ${rev:.0f} a real day, mean fare ${rev / max(rail, 1):.2f}", flush=True)
finally:
    subprocess.run([sys.executable, f"{MAPS}/cores.py", "release", name], capture_output=True, text=True)
