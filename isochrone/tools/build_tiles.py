"""Cut the walking-only friction surface into tiles for the view refine.

Input: MAP 2020 walking-only friction surface, 43200 x 17400 pixels of ~1 km,
EPSG:4326, 85N to 60S, minutes per metre. Exactly 0.06 (1 km/h) is water.

Output, under data/tiles/ (see docs/resolution.md):
  l1/<tr>_<tc>.bin.gz   L1, native 1/120 degree, 600 x 600 cells per tile
  l3/<tr>_<tc>.bin.gz   L3, 1/40 degree (3 x 3 pixels), 200 x 200 cells per tile
  index.json            grid sizes and the tiles that exist, per level

Both levels use 5 degree tiles: 72 x 29 of them. Values are uint8, row-major:
  0       water, or a cell with too little land
  1..255  walking speed in km/h times SCALE

Same rule as build_grid.py: a cell's speed comes from the mean of its land
pixels' friction (time per metre), and it is land when at least MIN_LAND of its
pixels are land. For L1 a cell is one pixel. All-water tiles are not written.
"""
import gzip
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import rasterio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data", "raw", "unzipped", "2020_walking_only_friction_surface.geotiff")
OUT = os.path.join(HERE, "data", "tiles")

SCALE = 40
MIN_LAND = 0.25
WATER = np.float32(0.06)
TILE_DEG = 5
LEVELS = {1: 600, 3: 200}   # factor -> cells per tile side


def encode(f, land, factor):
    """uint8 speeds for one strip, aggregated factor x factor."""
    rows, w = f.shape
    if factor == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            kmh = 0.06 / f
        val = np.clip(np.rint(kmh * SCALE), 1, 255)
        val[~land] = 0
        return np.nan_to_num(val).astype(np.uint8)
    fl = np.where(land, f, 0).astype(np.float64)
    gh, gw = rows // factor, w // factor
    fs = fl.reshape(gh, factor, gw, factor).sum(axis=(1, 3))
    ls = land.reshape(gh, factor, gw, factor).sum(axis=(1, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        kmh = 0.06 / (fs / ls)
    val = np.clip(np.rint(kmh * SCALE), 1, 255)
    val[ls < MIN_LAND * factor * factor] = 0
    return np.nan_to_num(val).astype(np.uint8)


def main():
    t0 = time.time()
    with rasterio.open(SRC) as ds:
        w, h = ds.width, ds.height
        strip = LEVELS[1]                      # one row of 5 degree tiles
        assert w % strip == 0 and h % strip == 0
        tx, ty = w // strip, h // strip
        found = {f: [] for f in LEVELS}
        nbytes = {f: 0 for f in LEVELS}
        for f in LEVELS:
            os.makedirs(os.path.join(OUT, "l%d" % f), exist_ok=True)
        for tr in range(ty):
            a = ds.read(1, window=((tr * strip, (tr + 1) * strip), (0, w)))
            land = (a != WATER) & (a > 0)
            for factor, size in LEVELS.items():
                g = encode(a, land, factor)
                assert g.shape == (size, tx * size)
                for tc in range(tx):
                    t = g[:, tc * size:(tc + 1) * size]
                    if not t.any():
                        continue
                    path = os.path.join(OUT, "l%d" % factor, "%d_%d.bin.gz" % (tr, tc))
                    data = gzip.compress(np.ascontiguousarray(t).tobytes(), compresslevel=6)
                    with open(path + ".tmp", "wb") as fh:
                        fh.write(data)
                    os.replace(path + ".tmp", path)
                    found[factor].append(tr * tx + tc)
                    nbytes[factor] += len(data)
            print("tile row %d / %d  (%.0f s)  L1 tiles %d  L3 tiles %d" %
                  (tr + 1, ty, time.time() - t0, len(found[1]), len(found[3])), flush=True)

    index = {"west": -180.0, "east": 180.0, "north": 85.0, "south": -60.0,
             "scale": SCALE, "tileDeg": TILE_DEG, "tilesX": tx, "tilesY": ty, "levels": {}}
    for factor, size in LEVELS.items():
        index["levels"][str(factor)] = {
            "factor": factor, "width": w // factor, "height": h // factor, "tile": size,
            "path": "l%d" % factor, "tiles": found[factor]}
    with open(os.path.join(OUT, "index.json"), "w") as fh:
        json.dump(index, fh, separators=(",", ":"))
    for f in LEVELS:
        print("L%d: %d tiles, %.1f MB gzipped" % (f, len(found[f]), nbytes[f] / 1e6))
    print("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
