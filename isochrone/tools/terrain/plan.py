"""The world list for the terrain run (notes/T-011.md): every 1 degree tile from 85 N to 60 S with
land on the page's L1 and a GLO-90 tile in the bucket, north to south and west to east, with what
the run needs to know about it.

    python isochrone/tools/terrain/plan.py        # writes data/terrain/plan.json, prints totals

Per tile: land (L1 land cells), roadcells (land cells with a road in their 3 x 3, which need the
grid with roads solved too), keep (the fine level will want its elevation: a town tile's block
mask, 0xFFFF for water country, mountains and the test regions; docs/fine.md).
"""
import gzip
import json
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glo90 as G                 # noqa: E402

HERE = G.HERE
TILES = os.path.join(HERE, "data", "tiles")
ROADS = os.path.join(HERE, "data", "roads", "l1")
FINE = os.path.join(HERE, "data", "fine")
TEST = ["n40w075", "n40w074", "n41w075", "n41w074", "n46e023", "n46e024", "n47e023", "n47e024"]


def main():
    ix = json.load(open(os.path.join(TILES, "index.json")))
    have = set(ix["levels"]["1"]["tiles"])
    tx = ix["tilesX"]
    sel = json.load(open(os.path.join(FINE, "selection.json")))
    keep = {k: int(v) for k, v in sel["towns"].items()}
    for k in sel["water"] + sel["mountains"] + TEST:
        keep[k] = 0xFFFF
    bucket = G.bucket_tiles()
    road_have = set(f[:-7] for f in os.listdir(ROADS) if f.endswith(".bin.gz"))
    out = []
    tot = {"tiles": 0, "land": 0, "roadcells": 0, "keep": 0, "no_glo90": 0}
    for tr in range(29):                       # 5 degree rows from 85 N
        strip = np.zeros((600, 43200), np.uint8)
        for tc in range(tx):
            if tr * tx + tc in have:
                t = np.frombuffer(gzip.open(os.path.join(TILES, "l1", "%d_%d.bin.gz" % (tr, tc))).read(), np.uint8)
                strip[:, tc * 600:(tc + 1) * 600] = t[:360000].reshape(600, 600)
        for j in range(5):
            lat = 84 - 5 * tr - j
            band = strip[j * 120:(j + 1) * 120]
            # L1 cells with a road in their 3 x 3 (within this row of tiles: an estimate at its edges)
            cnt = np.zeros((120, 43200), np.uint8)
            for lon in range(-180, 180):
                nm = G.tile_name(lat, lon)
                if nm in road_have:
                    b = np.frombuffer(gzip.open(os.path.join(ROADS, nm + ".bin.gz")).read(), np.uint8)
                    cnt[:, (lon + 180) * 120:(lon + 181) * 120] = b[:14400].reshape(120, 120) > 0
            near = maximum_filter(cnt, size=3, mode="wrap")
            for lon in range(-180, 180):
                cols = slice((lon + 180) * 120, (lon + 181) * 120)
                land = band[:, cols] > 0
                n = int(land.sum())
                if not n:
                    continue
                nm = G.tile_name(lat, lon)
                if nm not in bucket:
                    tot["no_glo90"] += 1
                    continue
                rc = int((land & (near[:, cols] > 0)).sum())
                out.append({"name": nm, "land": n, "roadcells": rc, "keep": keep.get(nm, 0)})
                tot["tiles"] += 1
                tot["land"] += n
                tot["roadcells"] += rc
                tot["keep"] += nm in keep
        print("row %d (%d N): %d tiles so far" % (tr, 85 - 5 * tr, tot["tiles"]), flush=True)
    os.makedirs(G.TERRAIN, exist_ok=True)
    json.dump({"tiles": out, "totals": tot}, open(os.path.join(G.TERRAIN, "plan.json"), "w"))
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
