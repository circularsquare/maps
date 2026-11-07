"""A = MERIT km / RiverATLAS km outside the forest tiles, by Q class x built gradient class."""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
p = np.load(os.path.join(HERE, "pairs.npz"))
box, q, ra, mk, grad, lon, lat = p["box"], p["q"], p["ra_km"], p["merit_km"], p["grad"], p["lon"], p["lat"]
ft = set(map(tuple, (t[:2] for t in json.load(open(os.path.join(HERE, "forest_tiles.json")))["0.2"])))
tr = np.floor((85 - lat) / 5).astype(int)
tc = np.floor((lon + 180) / 5).astype(int)
inf = np.array([(a, b) in ft for a, b in zip(tr, tc)])
print("pairs in forest tiles by box:", {b: int((inf & (box == b)).sum()) for b in dict.fromkeys(box.tolist()) if (inf & (box == b)).any()})
ratio = mk / ra
ok = (ratio > 0.5) & (ratio < 2.5)
g = np.nan_to_num(grad, nan=0)
CL = [(1, 10), (10, 100), (100, 1000), (1000, 1e7)]
GB = [(-99, 0.5), (0.5, 2), (2, 1e9)]
for lab, sel in (("outside forest tiles", ~inf), ("inside", inf), ("outside, south of 60N", ~inf & (lat < 60)), ("north of 60N", lat >= 60)):
    print("\n" + lab)
    for lo, hi in CL:
        s = "Q %-10s" % ("%g-%g" % (lo, hi))
        for a, b in GB:
            m = ok & sel & (q >= lo) & (q < hi) & (g >= a) & (g < b)
            nb = len(set(box[m].tolist()))
            s += "  %18s" % ("%.3f [%d, %d boxes]" % (mk[m].sum() / ra[m].sum(), m.sum(), nb) if m.sum() >= 10 else "-")
        m = ok & sel & (q >= lo) & (q < hi)
        s += "   all %.3f [%d]" % (mk[m].sum() / ra[m].sum(), m.sum()) if m.sum() else ""
        print(s)
