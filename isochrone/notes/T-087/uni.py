"""T-087: truth grids with the land cover mosaic taken out, so only the roads differ from the 1 km grid.

    python isochrone/notes/T-087/uni.py <tiles dir>

Per tile writes f_uni.bin (every off-road 100 m cell walks at its 1 km cell's off-road speed, road cells
at the L1 road speed, water cells as in f_cap) and f_uniplain.bin (the same without roads).
"""
import gzip
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "tools", "fine"))
import build as B

d = sys.argv[1]
up = lambda a: np.repeat(np.repeat(a, 10, 0), 10, 1)
for name in sorted(os.listdir(d)):
    p = os.path.join(d, name, "feat.npz")
    if not os.path.exists(p):
        continue
    f = np.load(p)
    lat = int(name[1:3]) * (1 if name[0] == "n" else -1)
    lon = int(name[4:7]) * (1 if name[3] == "e" else -1)
    cls = B.land_classes((lon, lat, lon + 1, lat + 1))
    bits = np.frombuffer(gzip.open(os.path.join(B.BITS, name + ".bits.gz")).read(), np.uint8).reshape(1200, 1200)
    water = (cls == 80) | (cls == 200)
    road = (bits & 7) > 0
    walk100, rk100 = up(f["walk"]), up(f["road_kmh"])
    off = np.where(water | (walk100 <= 0), 0, walk100)
    # a water or no-land-cover 100 m cell inside a land 1 km cell stays blocked, as in the other truths
    uni = np.where(road & ~water, np.maximum(off, rk100), off)
    uni = np.where(water & ((bits & 16) > 0), rk100, uni)
    q = lambda k: np.where(k > 0, np.clip(np.rint(k * 40), 1, 255), 0).astype(np.uint8)
    q(uni).tofile(os.path.join(d, name, "f_uni.bin"))
    q(off).tofile(os.path.join(d, name, "f_uniplain.bin"))
    print(name, flush=True)
