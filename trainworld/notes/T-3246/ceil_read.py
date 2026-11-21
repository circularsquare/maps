"""T-3246: the ceilings read off. python ceil_read.py"""
import glob
import json
import os

S = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(f"{S}/ceil_*_*.json")):
    j = json.load(open(f))
    row = [os.path.basename(f)[5:-5]]
    for s in ("A", "B", "A1056", "B1056"):
        a = j["acc"][j["scens"].index(s) * 5 + 4]
        row.append(f"{s} {100 * a[1] / (a[1] + a[3]):.1f}")
    print(*row)
