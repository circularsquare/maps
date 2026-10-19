"""T-700: walks from a Middle Henan Rd cell to the cells around it, as the game weighs them.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t700_cells.py [pack dir]

Shares are the plain page's walk shares against a dense-district drive with free parking
(docs/demand-plain.md), interpolated; over 10 km a walk is not offered.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "checks"))
from read_pack import PACKS, load_pack  # noqa: E402
from t700_walk_banks import Walks  # noqa: E402

h, a = load_pack("sha", sys.argv[1] if len(sys.argv) > 1 else PACKS)
w = Walks(h, a)
lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
R = 6371008.8
kc = math.cos(math.radians(lat0))
x, y = a["x_m"], a["y_m"]
KM = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0]
SH = [0.85, 0.82, 0.72, 0.43, 0.18, 0.06, 0.0]


def share(km):
    return 0.0 if km > 10 else float(np.interp(km, KM, SH))


ox, oy = R * math.radians(121.4824 - lon0) * kc, R * math.radians(31.2363 - lat0)
o = int(np.argmin((x - ox) ** 2 + (y - oy) ** 2))
print(f"from cell {o}, zone {w.zone[o]} part {w.part[o]}")
d = np.hypot(x - x[o], y - y[o])
near = np.where((d < 1300) & (np.abs(y - y[o]) < 400) & (x >= x[o]))[0]
near = near[np.argsort(x[near])]
print(f"{'cell':>6} {'lon':>9} {'zone':>5} {'part':>4} {'straight m':>10} {'factor':>7} {'walk km':>8} {'share':>6}")
for c in near:
    f = w.cells(o, c)
    km = d[c] * f / 1000
    lon = lon0 + math.degrees(x[c] / (R * kc))
    print(f"{c:6d} {lon:9.4f} {w.zone[c]:5d} {w.part[c]:4d} {d[c]:10.0f} {f:7.2f} {km:8.2f} {share(km):6.0%}")
