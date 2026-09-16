"""Build a PMTiles archive from the dots, merging rather than dropping. religiondots' method.

At low zoom a tile cannot hold every dot. Tippecanoe's answer is to DROP dots until it fits,
so small groups blink in and out as you pan. Here nothing is dropped: dots of one
nationality inside one merge cell (1/32 of a tile per side) become one mark at their mean
position carrying `k`, and the viewer draws its area proportional to k. Every group is
present at every zoom, in proportion.

Features within a tile are shuffled (seeded), because MVT paints in file order and an
unshuffled tile would let whichever group was emitted last paint over the others.

The MVT bytes come from mvt.py, a points-only encoder copied from religiondots, which is
~26x faster than mapbox_vector_tile for this shape of data.

Usage:
    python tiles.py                 # z0-12
    python tiles.py --max-zoom 11 --jobs 4
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "6")

import argparse
import contextlib
import gzip
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer

import mvt
from common import KEYS, PROC

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DOTS = os.path.join(PROC, "dots_2020.npz")
OUT = os.path.join(PROC, "chinaethnicity.pmtiles")
CELL_BITS = 6
EXTENT = 4096
SHUFFLE_SEED = 20260914


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def merge_at_zoom(g, wx, wy, z, cell_bits=None):
    CELL_BITS = cell_bits if cell_bits is not None else globals()["CELL_BITS"]
    n = 1 << (z + CELL_BITS)
    cx = np.minimum((wx * n).astype(np.int64), n - 1)
    cy = np.minimum((wy * n).astype(np.int64), n - 1)
    ng = len(KEYS)
    key = (cx * n + cy) * ng + g
    codes, uniq = pd.factorize(key)
    k = np.bincount(codes, minlength=len(uniq))
    mx = np.bincount(codes, weights=wx, minlength=len(uniq)) / k
    my = np.bincount(codes, weights=wy, minlength=len(uniq)) / k
    u = uniq.astype(np.int64)
    u, o_g = np.divmod(u, ng)
    return dict(tx=(u // n) >> CELL_BITS, ty=(u % n) >> CELL_BITS, wx=mx, wy=my,
                k=k.astype(np.int64), g=o_g)


def tile_pixels(wx, wy, tx, ty, z):
    nt = 1 << z
    px = np.clip((wx * nt - tx) * EXTENT, 0, EXTENT - 1).astype(np.int64)
    py = np.clip((wy * nt - ty) * EXTENT, 0, EXTENT - 1).astype(np.int64)
    return px, py


def bucket(tx, ty, z, rng):
    """Sort features into tiles, in a shuffled order within each tile."""
    key = tx.astype(np.int64) * (1 << z) + ty.astype(np.int64)
    perm = rng.permutation(len(key))
    order = perm[np.argsort(key[perm], kind="stable")]
    return key[order], order


def _encode_chunk(tasks):
    return [(tid, gzip.compress(mvt.encode(payload, EXTENT), 6, mtime=0))
            for tid, payload in tasks]


def _chunk(tasks, jobs):
    n = max(1, min(len(tasks), jobs * 8))
    size = -(-len(tasks) // n)
    return [tasks[i:i + size] for i in range(0, len(tasks), size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-zoom", type=int, default=0)
    ap.add_argument("--max-zoom", type=int, default=12)
    ap.add_argument("--jobs", type=int, default=6, help="encoding processes (default 6)")
    ap.add_argument("--cell-bits", type=int, default=CELL_BITS,
                    help="merge cells per tile side = 2^bits (default %(default)s: 8 px on a "
                         "512 px tile). religiondots uses 5; at 1:200 that reads as a lattice")
    args = ap.parse_args()

    D = np.load(DOTS)
    lon, lat, g = D["lon"], D["lat"], D["g"].astype(np.int64)
    wx = (lon + 180.0) / 360.0
    s = np.sin(np.radians(np.clip(lat, -85.05112878, 85.05112878)))
    wy = 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)
    log(f"{len(lon):,} dots at 1:{int(D['dot_value']):,}")

    NODE_T = [mvt.value_str(k) for k in KEYS]
    rng = np.random.default_rng(SHUFFLE_SEED)
    tmp = OUT + ".tmp"
    n_tiles = 0
    pool_ctx = (ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1
                else contextlib.nullcontext())
    with pool_ctx as pool, open(tmp, "wb") as fh:
        w = Writer(fh)
        for z in range(args.min_zoom, args.max_zoom + 1):
            m = merge_at_zoom(g, wx, wy, z, args.cell_bits)
            px, py = tile_pixels(m["wx"], m["wy"], m["tx"], m["ty"], z)
            kc, k_table = mvt.intern(m["k"], kind="int")
            key, o = bucket(m["tx"], m["ty"], z, rng)
            lpx, lpy = px[o], py[o]
            codes = np.stack([m["g"], kc], axis=1)[o]
            touched = np.unique(key)
            nt = 1 << z
            tids = np.fromiter((zxy_to_tileid(z, int(t // nt), int(t % nt)) for t in touched),
                               dtype=np.int64, count=len(touched))
            tasks = []
            for i in np.argsort(tids):
                t = touched[i]
                a0 = np.searchsorted(key, t, "left")
                b0 = np.searchsorted(key, t, "right")
                tasks.append((int(tids[i]), [("dots", lpx[a0:b0], lpy[a0:b0], ["n", "k"],
                                              codes[a0:b0], [NODE_T, k_table])]))
            chunks = _chunk(tasks, args.jobs)
            for blobs in (map(_encode_chunk, chunks) if pool is None
                          else pool.map(_encode_chunk, chunks)):
                for tid, blob in blobs:
                    w.write_tile(tid, blob)
            n_tiles += len(touched)
            log(f"  z{z:<2} {len(touched):>7,} tiles  {len(m['k']):>9,} marks "
                f"(largest merge {int(m['k'].max()):,})")
        w.finalize(
            {"tile_type": TileType.MVT, "tile_compression": Compression.GZIP,
             "min_zoom": args.min_zoom, "max_zoom": args.max_zoom,
             "min_lon_e7": int(lon.min() * 1e7), "min_lat_e7": int(lat.min() * 1e7),
             "max_lon_e7": int(lon.max() * 1e7), "max_lat_e7": int(lat.max() * 1e7),
             "center_zoom": 4, "center_lon_e7": int(105 * 1e7), "center_lat_e7": int(35 * 1e7)},
            {"name": "chinaethnicity",
             "vector_layers": [{"id": "dots", "fields": {"n": "String", "k": "Number"}}]})
    mb = os.path.getsize(tmp) / 1e6
    try:
        os.replace(tmp, OUT)
    except OSError as e:
        raise SystemExit(f"built {tmp} ({mb:.1f} MB) but could not move it into place: {e}\n"
                         f"something (usually `npx serve`) has {OUT} open")
    log(f"wrote {OUT} ({mb:.1f} MB, {n_tiles:,} tiles)")


if __name__ == "__main__":
    main()
