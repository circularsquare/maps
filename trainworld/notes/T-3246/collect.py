"""T-3246: each calib run's summary lines into results.txt beside this file. python collect.py <run folder>"""
import glob
import os
import sys

d = sys.argv[1]
out = []
for f in sorted(glob.glob(f"{d}/*.log")):
    keep = [l.strip() for l in open(f, encoding="utf-8-sig", errors="replace") if l.startswith(("commutes", "operators", "t3246 stages", "calib.exe : commutes"))]
    cmd = open(f, encoding="utf-8-sig", errors="replace").readline().strip()
    if keep:
        out.append(f"== {os.path.basename(f)[:-4]}\n{cmd.split('--rounds')[-1].strip()}\n" + "\n".join(keep))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt"), "w", encoding="utf-8").write("\n\n".join(out) + "\n")
print(len(out), "runs")
