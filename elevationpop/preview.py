"""Render the whole country to one PNG, to judge the ramp without tiling.

Composited over the same dark base the map uses, so what you see here is what
the population layer contributes before the hillshade goes on top.
"""

import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import numpy as np
import rasterio
from PIL import Image

import ramp

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "tur_pop_3857.tif")
OUT = os.path.join(HERE, "data", "preview.png")

WIDTH = 2400
BASE = (34, 28, 46)  # keep in step with the background layer in index.html


def main():
    with rasterio.open(SRC) as d:
        height = int(round(WIDTH * d.height / d.width))
        arr = d.read(1, out_shape=(height, WIDTH))

    lit = arr[arr > ramp.LO]
    print(f"{WIDTH} x {height}")
    print(f"cells above the ramp floor: {lit.size:,} of {arr.size:,} ({100 * lit.size / arr.size:.1f}%)")
    if lit.size:
        for p in (50, 90, 99, 100):
            print(f"  p{p:<4} {np.percentile(lit, p):9.3f}")

    rgba = ramp.colorize(arr).astype("float32")
    a = rgba[..., 3:4] / 255.0
    flat = rgba[..., :3] * a + np.array(BASE, dtype="float32") * (1.0 - a)

    Image.fromarray(flat.round().astype("uint8"), "RGB").save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
