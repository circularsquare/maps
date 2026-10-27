"""Check that 10x10 blocks of the 100 m grid line up with GHSL's own 1 km grid.

    python -I check_1km.py <100 m tif> <1 km tif>

Sums 10x10 blocks of 100 m cells over a few test areas (window origins are multiples of 10
cells from the grid origin, as in build.py) and compares them with the 1 km product cell by cell.
A shifted alignment shows up as a much worse fit than the aligned one.
"""
import sys

import numpy as np
import rasterio
from rasterio.windows import Window

AREAS = {"Netherlands": (4.3, 52.0), "Lagos": (3.4, 6.5), "Dhaka": (90.4, 23.8),
         "Iowa": (-93.6, 41.6)}
N = 200  # km


def main():
    f100, f1k = sys.argv[1], sys.argv[2]
    from pyproj import Transformer
    with rasterio.open(f100) as a, rasterio.open(f1k) as b:
        tr = Transformer.from_crs("EPSG:4326", a.crs.to_wkt(), always_xy=True)
        assert a.transform.c == b.transform.c and a.transform.f == b.transform.f
        for name, (lon, lat) in AREAS.items():
            x, y = tr.transform(lon, lat)
            r1, c1 = b.index(x, y)
            r1, c1 = r1 - N // 2, c1 - N // 2
            ref = b.read(1, window=Window(c1, r1, N, N))
            ref[ref < 0] = 0
            for shift in (0, 5):
                raw = a.read(1, window=Window(c1 * 10 + shift, r1 * 10 + shift, N * 10, N * 10))
                raw[raw < 0] = 0
                blk = raw.reshape(N, 10, N, 10).sum(axis=(1, 3))
                err = np.abs(blk - ref).sum() / ref.sum()
                print(f"{name:12} shift {shift}: 100 m total {raw.sum():14,.0f}  1 km total "
                      f"{ref.sum():14,.0f}  cellwise abs diff {err * 100:6.3f}%")


if __name__ == "__main__":
    main()
