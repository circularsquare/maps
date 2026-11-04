"""Fetch bare-earth ground elevation (USGS 3DEP, mostly the 2017 NYC lidar at 1 m)
resampled to 3 m in UTM 18N, as 4.5 km GeoTIFF tiles covering the subway.

Writes raw/dem/dem_<col>_<row>.tif (metres, NAVD88). Skips tiles with no track.
"""
import json
import math
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from pyproj import Transformer

ROOT = pathlib.Path(__file__).parent
RAW = ROOT / "raw"
OUT = RAW / "dem"
URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
UA = "trackdepth/0.1"
RES = 3.0
PX = 1500  # 4000 px times out at the server (504)
TILE = RES * PX
X0, Y0 = 560000.0, 4480000.0  # UTM 18N grid origin, south-west of Staten Island

to_utm = Transformer.from_crs(4326, 32618, always_xy=True)


def needed_tiles():
    import build  # the same track selection the model uses
    tiles = set()
    for e in build.load_tracks():
        for p in e["geometry"]:
            x, y = to_utm.transform(p["lon"], p["lat"])
            tiles.add((math.floor((x - X0) / TILE), math.floor((y - Y0) / TILE)))
    return sorted(tiles)


def fetch(c, r):
    out = OUT / f"dem_{c}_{r}.tif"
    if out.exists() and out.stat().st_size > 1e6:
        return
    x0, y0 = X0 + c * TILE, Y0 + r * TILE
    params = {
        "bbox": f"{x0},{y0},{x0 + TILE},{y0 + TILE}",
        "bboxSR": 32618, "imageSR": 32618,
        "size": f"{PX},{PX}",
        "format": "tiff", "pixelType": "F32", "noData": -9999,
        "interpolation": "RSP_BilinearInterpolation",
        "compression": "LZ77",
        "f": "image",
    }
    for attempt in range(4):
        try:
            r_ = requests.get(URL, params=params, headers={"User-Agent": UA}, timeout=600)
            if r_.ok:
                break
            print(f"{out.name}: {r_.status_code}, retrying", flush=True)
        except requests.RequestException as ex:
            print(f"{out.name}: {ex}, retrying", flush=True)
        time.sleep(10 * (attempt + 1))
    r_.raise_for_status()
    if not r_.content.startswith((b"II*", b"MM\x00*")):
        raise RuntimeError(f"not a tiff for {c},{r}: {r_.content[:300]!r}")
    tmp = out.with_suffix(".part")
    tmp.write_bytes(r_.content)
    tmp.replace(out)
    print(f"{out.name} {len(r_.content)/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = needed_tiles()
    print(len(tiles), "tiles:", tiles, flush=True)
    with ThreadPoolExecutor(4) as pool:
        for f in [pool.submit(fetch, c, r) for c, r in tiles]:
            f.result()
