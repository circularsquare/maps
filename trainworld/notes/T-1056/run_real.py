"""T-1055/T-1056: calib on the real networks. python run_real.py <cities> <runs>
runs: before (HEAD), after (T-1055), measure (T-1055, T1056=1 counts), rule (T1056=2)."""
import os
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
EXE = {"before": f"{D}/before/trainworld/sim/target/release/calib.exe"}
for r in ("after", "measure", "rule"):
    EXE[r] = f"{D}/meas/sim/target/release/calib.exe"
MODE = {"before": "0", "after": "0", "measure": "1", "rule": "2"}
OUT = f"{D}/real"
os.makedirs(OUT, exist_ok=True)
for c in sys.argv[1].split(","):
    for r in sys.argv[2].split(","):
        out = f"{OUT}/{c}_{r}.json"
        if os.path.exists(out):
            continue
        env = dict(os.environ, TW_ROOT=TW, T1056=MODE[r], RAYON_NUM_THREADS="3")
        t = time.time()
        p = subprocess.run([EXE[r], "--net", f"{TW}/data/work/realnet/{c}_real.json", "--out", out], env=env, capture_output=True, text=True)
        print(c, r, f"{time.time() - t:.0f}s rc {p.returncode}", p.stderr.strip().splitlines()[-3:] if p.returncode else p.stderr.strip().splitlines()[0], flush=True)
