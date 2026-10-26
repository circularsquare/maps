"""T-047: how many tiles each candidate rule picks, at 1 degree (world.csv) and, for towns, at
0.25 degree blocks (lc_class.tif).

    python isochrone/notes/T-047/counts.py <world.csv>
"""
import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(wc):
    T = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(wc))]
    print("land tiles", len(T))
    for u in (0.005, 0.01, 0.02, 0.05, 0.1):
        print("urban >= %.3f: %d" % (u, sum(t["urban"] >= u for t in T)))
    for g in (0.25, 0.4, 0.5, 0.7):
        print("rugged >= %.2f: %d" % (g, sum(t["rugged"] >= g for t in T)))
    for m in (0.1, 0.2):
        print("mixed >= %.2f: %d; and urban >= 0.01: %d" % (m, sum(t["mixed"] >= m for t in T),
              sum(t["mixed"] >= m and t["urban"] >= 0.01 for t in T)))
    for u, g in ((0.02, 0.5), (0.02, 0.4), (0.05, 0.5), (0.01, 0.5)):
        print("urban >= %.2f or rugged >= %.2f: %d" % (u, g, sum(t["urban"] >= u or t["rugged"] >= g for t in T)))
    # quarter-degree blocks with towns in them
    src = rasterio.open(os.path.join(HERE, "data", "surface", "lc_class.tif"))
    counts = {0.02: 0, 0.05: 0, 0.1: 0, 0.2: 0}
    anyurb = 0
    for t in range(145):
        a = src.read(1, window=Window(0, t * 120, 43200, 120)) == 50
        b = a.reshape(4, 30, 1440, 30).sum((1, 3)) / 900
        for k in counts:
            counts[k] += int((b >= k).sum())
        anyurb += int((b > 0).sum())
    print("quarter-degree blocks with any town %d; urban share >= %s" % (anyurb, counts))


if __name__ == "__main__":
    main(sys.argv[1])
