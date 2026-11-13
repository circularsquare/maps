"""T-2901: terrain classes for all land, one file per one degree block, from the fine intermediate.

The world lattice and the classes are zone_terrain.py's (`classes_of`, `encode`); each block is made with one cell of its
neighbours so the 3 x 3 filter agrees across block edges. Out, in <out>: `<la>_<lo>.terrain` (the south-west corner, TWT1
gzip as `ClassGrid::decode` reads: 960 x 960 cells) for every block with data, flat blocks included (the hillshade reads
them), and `index.bin`: which blocks have a file. A block with no data has no file (open sea). Blocks already written are skipped.

index.bin: `TWI1`, i16 south (-65), u16 rows (150), i16 west (-180), u16 columns (360), u32 blocks listed, then one bit a
block row-major from the south-west, the first in the low bit of each byte.

    world_terrain.py build <fine dir> <out dir> [--jobs n]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import zone_terrain as zt

SOUTH, ROWS, WEST, COLS = -65, 150, -180, 360
PER_DEG = zt.PER_DEG

_FINE = None


def _fine(root: str) -> zt.Fine:
    global _FINE
    if _FINE is None:
        _FINE = zt.Fine(Path(root))
    return _FINE


def block_classes(fine: zt.Fine, la: int, lo: int) -> "tuple[np.ndarray, int, int] | None":
    """A block's classes with its lattice corner; None when the block has no data."""
    c0, r0 = (lo + 180) * PER_DEG, (la + 90) * PER_DEG
    q = fine.slopes(c0 - 1, r0 - 1, PER_DEG + 2, PER_DEG + 2)
    if (q[1:-1, 1:-1] == 255).all():
        return None
    cls = zt.classes_of(q, fine.coarse(c0 - 1, r0 - 1, PER_DEG + 2, PER_DEG + 2))
    return cls[1:-1, 1:-1], c0, r0


def one(args) -> "tuple[str, int]":
    root, out, key = args
    la, lo = map(int, key.split("_"))
    path = Path(out) / f"{key}.terrain"
    if path.exists():
        return key, path.stat().st_size
    got = block_classes(_fine(root), la, lo)
    if got is None:
        return key, 0
    cls, c0, r0 = got
    data = zt.encode(cls, c0, r0)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return key, len(data)


def build(root: str, out: str, jobs: int) -> None:
    man = json.loads((Path(root) / "manifest.json").read_text())
    keys = sorted((k for k, v in man["blocks"].items() if v), key=lambda k: tuple(map(int, k.split("_"))))
    Path(out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bits = np.zeros((ROWS, COLS), bool)
    size = n = none = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for k, (key, m) in enumerate(ex.map(one, [(root, out, x) for x in keys], chunksize=16)):
            if m:
                la, lo = map(int, key.split("_"))
                bits[la - SOUTH, lo - WEST] = True
                size += m
                n += 1
            else:
                none += 1
            if (k + 1) % 1000 == 0 or k == len(keys) - 1:
                print(f"{k + 1}/{len(keys)} blocks, {n} files, {size / 1e6:.1f} MB, {time.time() - t0:.0f} s", flush=True)
    head = b"TWI1" + struct.pack("<hHhHI", SOUTH, ROWS, WEST, COLS, n)
    # replaced, never rewritten in place: builds hold it as a hard link (docs/world-run.md)
    tmp = Path(out) / "index.bin.part"
    tmp.write_bytes(head + np.packbits(bits.reshape(-1), bitorder="little").tobytes())
    os.replace(tmp, Path(out) / "index.bin")
    print(f"done: {n} files, {none} with no data, {size / 1e9:.3f} GB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "build":
        build(a[1], a[2], int(a[a.index("--jobs") + 1]) if "--jobs" in a else 4)
    else:
        print(__doc__)
