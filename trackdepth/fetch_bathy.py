"""Fetch riverbed / seafloor elevation under the river tunnels: NOAA NCEI CUDEM topobathy,
1/9 arc-second (about 3.4 m x 2.6 m here), NAVD88 metres, via the NCEI DEM_mosaics image
service. The tiles ncei19_n40x75_w074x00 / w074x25 and n41x00_w074x00 / w074x25 (2015v1 and
2018v2 vintages of the NCEI "Continuously Updated DEM") cover New York Harbor and the rivers.

Looks at data/tracks.json (build.py output) for tunnel points the build treats as wet (ground
missing or below 1 m, the same rule as build.py) and downloads native-grid GeoTIFFs (EPSG:4326,
float32) for the 0.04 degree blocks that contain them, with a small margin.

Writes raw/bathy/cudem_<i>_<j>.tif. Then run make_riverbed.py (evidence/riverbed.csv).

    python fetch_bathy.py --fill   # raw/bathy_fill/: the rest of the water in the terrain extent
"""
import json
import math
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "raw" / "bathy"
URL = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage"
UA = "trackdepth/0.1"
BLOCK = 0.04            # degrees per tile (1296 px at 1/9 arc-second)
PX = 1296
MARGIN = 0.004          # degrees around a wet point that a block must reach
# CUDEM catalogue items (OBJECTID in DEM_all) covering NYC, found by /query on 2026-11-02
LOCK_IDS = [1620, 1621, 1629, 1630]


def wet_points():
    d = json.loads((ROOT / "data" / "tracks.json").read_text(encoding="utf-8"))
    pts = []
    for t in d["tracks"]:
        if t["cls"] != "tunnel":
            continue
        for lon, lat, g in zip(t["lon"], t["lat"], t["ground"]):
            if g is None or g < 1.0:
                pts.append((lon, lat))
    return pts


def needed_blocks(pts):
    blocks = set()
    for lon, lat in pts:
        for dx in (-MARGIN, MARGIN):
            for dy in (-MARGIN, MARGIN):
                blocks.add((math.floor((lon + dx) / BLOCK), math.floor((lat + dy) / BLOCK)))
    return sorted(blocks)


def fetch(i, j):
    out = OUT / f"cudem_{i}_{j}.tif"
    if out.exists() and out.stat().st_size > 1e5:
        return
    x0, y0 = i * BLOCK, j * BLOCK
    params = {
        "bbox": f"{x0},{y0},{x0 + BLOCK},{y0 + BLOCK}",
        "bboxSR": 4326, "imageSR": 4326,
        "size": f"{PX},{PX}",
        "format": "tiff", "pixelType": "F32", "noData": -9999,
        "interpolation": "RSP_BilinearInterpolation",
        "compression": "LZ77",
        "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": LOCK_IDS,
                                  "ascending": True, "mosaicOperatorType": "MT_FIRST"}),
        "f": "image",
    }
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, headers={"User-Agent": UA}, timeout=300)
            if r.ok and r.content.startswith((b"II*", b"MM\x00*")):
                break
            print(f"{out.name}: {r.status_code} {r.content[:200]!r}, retrying", flush=True)
        except requests.RequestException as ex:
            print(f"{out.name}: {ex}, retrying", flush=True)
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"gave up on {out.name}")
    tmp = out.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(out)
    print(f"{out.name} {len(r.content) / 1e6:.1f} MB", flush=True)


# --- terrain fill (make_terrain.py) ----------------------------------------------------------
# The terrain tiles need the bed under all the water inside their extent, not just under the
# river tunnels. `--fill` fetches every block of the terrain extent that has 3DEP ground below
# 1 m and is not in raw/bathy, at half the pixel count (about 6 m cells: the terrain grid is
# 14.5 m), into raw/bathy_fill/ (a separate folder: make_riverbed.py reads only raw/bathy). The
# catalogue items added to LOCK_IDS cover the east side of the extent (Jamaica Bay, the Sound,
# the Atlantic off the Rockaways) and the Hudson above 41 N.
FILL_OUT = ROOT / "raw" / "bathy_fill"
FILL_IDS = LOCK_IDS + [1619, 1628, 1638, 1639]
FILL_PX = 648


def fill_blocks():
    import make_terrain as mt
    import numpy as np
    dem, _ = mt.sources()
    w, s, e, n = mt.extent(dem)     # the terrain extent
    tx0, ty0, nx, ny = mt.tile_block(w, s, e, n)
    f = 2 ** (mt.ZMAX - mt.ZMIN)
    g = mt.mosaic(dem, tx0 * f, ty0 * f, nx * f, ny * f)
    coarse = mt.mosaic(mt.fetch_coarse(w, s, e, n), tx0 * f, ty0 * f, nx * f, ny * f)
    g[~np.isfinite(g)] = coarse[~np.isfinite(g)]
    res = mt.TILE_M(mt.ZMAX) / 256
    cx = (-mt.HALF + (tx0 * f * 256 + np.arange(g.shape[1]) + 0.5) * res) / mt.HALF * 180
    my = mt.HALF - (ty0 * f * 256 + np.arange(g.shape[0]) + 0.5) * res
    cy = np.degrees(2 * np.arctan(np.exp(my / mt.HALF * math.pi)) - math.pi / 2)
    wet = ~np.isfinite(g) | (g < 1.0)
    cols = (cx >= w) & (cx <= e)
    rows = (cy >= s) & (cy <= n)
    wet = wet[np.ix_(rows, cols)]
    bi = np.floor(cx[cols] / BLOCK).astype(int)
    bj = np.floor(cy[rows] / BLOCK).astype(int)
    blocks = []
    for i in np.unique(bi):
        for j in np.unique(bj):
            if (OUT / f"cudem_{i}_{j}.tif").exists():
                continue
            if wet[np.ix_(bj == j, bi == i)].any():
                blocks.append((int(i), int(j)))
    return blocks


def fetch_fill(i, j):
    out = FILL_OUT / f"cudem_{i}_{j}.tif"
    if out.exists() and out.stat().st_size > 1e4:
        return
    x0, y0 = i * BLOCK, j * BLOCK
    params = {
        "bbox": f"{x0},{y0},{x0 + BLOCK},{y0 + BLOCK}",
        "bboxSR": 4326, "imageSR": 4326, "size": f"{FILL_PX},{FILL_PX}",
        "format": "tiff", "pixelType": "F32", "noData": -9999,
        "interpolation": "RSP_BilinearInterpolation", "compression": "LZ77",
        "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": FILL_IDS,
                                  "ascending": True, "mosaicOperatorType": "MT_FIRST"}),
        "f": "image",
    }
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, headers={"User-Agent": UA}, timeout=300)
            if r.ok and r.content.startswith((b"II*", b"MM\x00*")):
                break
            print(f"{out.name}: {r.status_code} {r.content[:200]!r}, retrying", flush=True)
        except requests.RequestException as ex:
            print(f"{out.name}: {ex}, retrying", flush=True)
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"gave up on {out.name}")
    tmp = out.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(out)
    print(f"{out.name} {len(r.content) / 1e6:.1f} MB", flush=True)
    time.sleep(1)


if __name__ == "__main__" and "--fill" in sys.argv:
    FILL_OUT.mkdir(parents=True, exist_ok=True)
    blocks = fill_blocks()
    print(len(blocks), "fill blocks", flush=True)
    for i, j in blocks:
        fetch_fill(i, j)
    raise SystemExit

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pts = wet_points()
    blocks = needed_blocks(pts)
    print(len(pts), "wet tunnel points;", len(blocks), "blocks:", blocks, flush=True)
    for i, j in blocks:
        fetch(i, j)
