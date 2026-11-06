"""Which tiles and quarter-degree blocks get the fine level (docs/fine.md, "Where the fine level
goes"), from the 1 km layers of data/surface/.

    python isochrone/tools/fine/select.py        (~2 min, one core) -> data/fine/selection.json

selection.json:
  towns      {tile: mask} 0.25 degree blocks with LC100 urban >= 2% of the block's 1 km pixels;
             mask bit 4 * i + j is block row i from the north, column j from the west (0-3)
  water      [tile] >= 10% of pixels part water (2-98%) and urban >= 1% of land
  mountains  [tile] >= 70% of land sloping over 5 degrees (waits for stage 2 elevation, T-039)
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.join(HERE, "data", "surface")
OUT = os.path.join(HERE, "data", "fine", "selection.json")

TOWN_URBAN = 0.02       # of a block's pixels
WATER_MIXED = 0.10      # of a tile's pixels
WATER_URBAN = 0.01      # of a tile's land
MOUNTAIN_RUGGED = 0.70  # of a tile's land
URBAN = 50              # LC100 class

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import tile_name  # noqa: E402


def tiles_of(a):
    """(120, 43200) -> (360, 14400): the 120 x 120 pixels of each 1 degree tile."""
    return a.reshape(120, 360, 120).transpose(1, 0, 2).reshape(360, -1)


def main():
    src = {n: rasterio.open(os.path.join(S, n + ".tif")) for n in ("lc_class", "lc_water", "slope", "walk_today")}
    towns, water, mountains = {}, [], []
    weights = 1 << np.arange(16).reshape(4, 4)
    for t in range(145):                       # 1 degree rows, 85 N to 60 S
        win = Window(0, t * 120, 43200, 120)
        cls = src["lc_class"].read(1, window=win)
        wat = src["lc_water"].read(window=win).astype(np.float32).sum(0) / 250
        sl = src["slope"].read(1, window=win).astype(np.float32) / 100
        land = src["walk_today"].read(1, window=win) > 0
        urb = cls == URBAN
        blk = urb.reshape(4, 30, 1440, 30).sum((1, 3)) / 900          # (4, 1440) blocks
        blk = blk.reshape(4, 360, 4).transpose(1, 0, 2) >= TOWN_URBAN  # (360 tiles, 4, 4)
        mask = (blk * weights).sum((1, 2))
        L = tiles_of(land).sum(1)
        U = tiles_of(urb & land).sum(1)
        M = tiles_of((wat > 0.02) & (wat < 0.98)).sum(1)
        R = tiles_of(land & (sl > 5)).sum(1)
        lat = 84 - t
        for c in range(360):
            name = tile_name(lat, c - 180)
            if mask[c]:
                towns[name] = int(mask[c])
            if L[c] == 0:
                continue
            if M[c] >= WATER_MIXED * 14400 and U[c] >= WATER_URBAN * L[c]:
                water.append(name)
            if R[c] >= MOUNTAIN_RUGGED * L[c]:
                mountains.append(name)
    nblocks = sum(bin(m).count("1") for m in towns.values())
    full = sum(1 for m in towns.values() if m == 0xFFFF)
    out = {"rule": {"townUrban": TOWN_URBAN, "waterMixed": WATER_MIXED, "waterUrban": WATER_URBAN,
                    "mountainRugged": MOUNTAIN_RUGGED},
           "towns": dict(sorted(towns.items())), "water": sorted(water), "mountains": sorted(mountains)}
    tmp = OUT + ".tmp"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(out, f, indent=0)
    os.replace(tmp, OUT)
    both = set(towns) | set(water)
    print("towns: %d blocks in %d tiles (%d tiles with all 16)" % (nblocks, len(towns), full))
    print("water: %d tiles (%d without a town block)" % (len(water), len(set(water) - set(towns))))
    print("mountains: %d tiles (not built yet)" % len(mountains))
    print("towns or water: %d tiles" % len(both))


if __name__ == "__main__":
    main()
