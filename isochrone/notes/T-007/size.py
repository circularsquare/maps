"""T-007: gzipped size of proto.py's patch in a plain binary layout, per split and listed cell.

    python isochrone/notes/T-007/size.py <proto out dir> <f: 10 or 3>

Layout tried (all tiles of the dir together, roads on and off files kept apart):
  split cells: cell delta (uint32), slots (uint8); per slot 4 speeds (uint8 km/h x 40, one a direction)
  and 8 link masks (uint8); the slot map, 4 bits a pixel (f x f);
  listed cells: cell delta (uint32), 8 link masks.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

d, f = sys.argv[1], int(sys.argv[2])
for roads in ("on", "off"):
    files = sorted(glob.glob(os.path.join(d, "p_%s_*.json" % roads)))
    if not files:
        continue
    sp, li = [], []
    for fn in files:
        r = json.load(open(fn))
        sp += r["split"]
        li += r["listed"]
    sp.sort(key=lambda x: x[0])
    li.sort(key=lambda x: x[0])
    a = bytearray()
    prev = 0
    for cell, sds, masks, smap in sp:
        a += np.uint32(cell - prev).tobytes()
        prev = cell
        a += bytes([len(sds)])
        for s, sd in enumerate(sds):
            a += bytes(int(min(255, max(1, round(40 / h)))) for h in sd)
            a += bytes(masks[s])
        m = np.array(smap, np.uint8)
        m[m == 255] = 15
        if len(m) % 2:
            m = np.append(m, 15)
        a += bytes((m[0::2] << 4 | m[1::2]).tolist())
    b = bytearray()
    prev = 0
    for cell, masks in li:
        b += np.uint32(cell - prev).tobytes()
        prev = cell
        b += bytes(masks)
    zs, zl = len(gzip.compress(bytes(a), 9)), len(gzip.compress(bytes(b), 9))
    ns = sum(len(x[1]) for x in sp)
    print("roads %s: %d tiles, split %d (%d slots) %.1f KB gz = %.1f B a split cell; listed %d %.1f KB gz = %.1f B a cell"
          % (roads, len(files), len(sp), ns, zs / 1e3, zs / max(1, len(sp)), len(li), zl / 1e3, zl / max(1, len(li))))
