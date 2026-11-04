"""T-2301: the seed tar a wave's machine unpacks under trainworld/ (launch.py --file seed=<out>, wave.sh GET_seed):
home's shared inputs a build reads that the code bundle lacks. Makes no AWS call.

    seed.py <out.tgz> [w,s,e,n]       (the box defaults to the US with Alaska, Hawaii and Puerto Rico)

- data/work/world_field.npz, .countries.json, .links.json and outside.json (routes' land parts read the
  world field, T-2213);
- the coastline water polygons (osm_water's SEA_FGB) clipped to the box, `zip_fid` kept so osm_water.sea
  reads the same polygons in the same order, and an empty zip a day older beside it, so sea_fgb takes the
  fgb as fresh and downloads nothing (osmdata.openstreetmap.de did not answer on 2026-11-04).
"""
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import pyogrio

TW = Path(__file__).resolve().parents[2]
SEA = "data/raw/osmdata/water-polygons-split-4326"
SHARED = ["data/work/world_field.npz", "data/work/world_field.countries.json", "data/work/world_field.links.json",
          "data/work/outside.json"]
US_BOX = (-180.0, 15.0, -60.0, 72.0)


def main(out, box=US_BOX):
    t = time.time()
    df = pyogrio.read_dataframe(str(TW / f"{SEA}.fgb"), bbox=box)
    with tempfile.TemporaryDirectory() as tmp:
        fgb, zp = os.path.join(tmp, "sea.fgb"), os.path.join(tmp, "sea.zip")
        pyogrio.write_dataframe(df, fgb, driver="FlatGeobuf", promote_to_multi=False, SPATIAL_INDEX="YES")
        open(zp, "wb").close()
        day_ago = time.time() - 86400
        os.utime(zp, (day_ago, day_ago))
        with tarfile.open(out, "w:gz") as tar:
            tar.add(zp, f"{SEA}.zip")
            tar.add(fgb, f"{SEA}.fgb")
            for f in SHARED:
                tar.add(str(TW / f), f)
    print(f"{out}: {len(df):,} sea polygons in {box}, {os.path.getsize(out) / 1e6:.0f} MB, {time.time() - t:.0f} s")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    main(sys.argv[1], tuple(float(x) for x in sys.argv[2].split(",")) if len(sys.argv) == 3 else US_BOX)
