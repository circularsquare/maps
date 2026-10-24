"""T-852: 4 km blocks in thin country. Re-solves a res-9 pack's gravity with every aligned 2x2 block
of 2 km squares as one zone where at least two of its zones of one region lie there and each makes
under UNDER trip ends a weekday (merge_cells.trip_ends, summed per square); towns stay at 2 km.

    python pipeline/coarse_zones.py <id> [<id> ...] [--under 1000] [--packs DIR]

Runs on the res-9 pack after the build and before routes.py (the routes and the merge are made for
the zones it leaves; world_run.py --coarse puts it there). The pack's own curve and border factor
are kept; the gravity is the pack's alone until the joint solve redoes it. A pack already merged
is refused: `merge_cells.py --undo` first.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import sys

import numpy as np

import merge_cells
import read_pack
import simbin
from read_pack import load_pack

UNDER = 1000.0  # Anita, 2026-10-21 (notes/T-852.md)


def coarsen(city, under=UNDER, packs=None):
    packs = packs or read_pack.PACKS
    header, a = load_pack(city, packs)
    if "merged" in header:
        sys.exit(f"{city}: merged; coarse_zones runs on the res-9 pack (merge_cells.py --undo first)")
    g = header.get("gravity")
    if not g:
        sys.exit(f"{city}: no gravity to re-solve")
    n = header["cells"]
    ends = merge_cells.trip_ends(a, n).astype("<f4")
    work = os.path.join(read_pack.ROOT, "data", "work") if packs == read_pack.PACKS else packs
    os.makedirs(work, exist_ok=True)
    ends_path = os.path.join(work, f"{city}.ends.f32")
    ends.tofile(ends_path)
    pack_json = os.path.join(packs, f"{city}.json")
    decay_pow = g.get("decay_pow", 0.0) if g["decay"] == "pow_exp" else 0.0
    try:
        simbin.run(read_pack.ROOT, "pack_gravity", [
            pack_json, pack_json, "--zone-m", str(g["zone_m"]), "--decay-km", str(g["decay_km"]), "--decay-pow", str(decay_pow),
            "--border-factor", str(g.get("border_factor", 1.0)), "--coarse", str(under), "--ends", ends_path])
    finally:
        os.remove(ends_path)
    h2, a2 = load_pack(city, packs)
    read_pack.check_gravity(h2, a2)
    side = read_pack.zone_side(a2, h2["zones"])
    print(f"{city}: {header['zones']:,} zones -> {h2['zones']:,} ({int((side == 2).sum()):,} blocks of 4 km)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    under, packs = UNDER, None
    if "--under" in args:
        k = args.index("--under")
        under = float(args[k + 1])
        del args[k:k + 2]
    if "--packs" in args:
        k = args.index("--packs")
        packs = args[k + 1]
        del args[k:k + 2]
    if not args or any(x.startswith("--") for x in args):
        sys.exit(__doc__)
    for cid in args:
        coarsen(cid, under, packs)
