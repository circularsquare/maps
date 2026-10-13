"""T-181: which arrays of a pack changed against its copy in data/work/pre-T181/.

    ..\\venv\\Scripts\\python.exe trainworld/notes/t181_compare.py nyc wdc

Compares each array's bytes (offsets may move when arrays are compacted) and the header's
top-level blocks; prints the changed ones and, for changed f32 cell arrays, their totals.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = {"f32": 4, "f64": 8, "u8": 1, "u16": 2, "u32": 4, "u64": 8, "i32": 4}


def arrays(stem):
    header = json.load(open(stem + ".json"))
    buf = open(stem + ".bin", "rb").read()
    out = {}
    for a in header["arrays"]:
        n = a["count"] * SIZES[a["dtype"]]
        out[a["name"]] = (a["dtype"], buf[a["offset"]:a["offset"] + n])
    return header, out


def main(cities):
    for city in cities:
        ho, ao = arrays(os.path.join(ROOT, "data", "work", "pre-T181", city))
        hn, an = arrays(os.path.join(ROOT, "data", "packs", city))
        same = [k for k in ao if k in an and ao[k] == an[k]]
        changed = [k for k in ao if k in an and ao[k] != an[k]]
        print(f"{city}: {len(same)} arrays bit-identical, changed {changed}, "
              f"removed {[k for k in ao if k not in an]}, added {[k for k in an if k not in ao]}")
        for k in changed:
            dt, b0 = ao[k]
            if dt == "f32":
                v0 = np.frombuffer(b0, "<f4").astype(np.float64)
                v1 = np.frombuffer(an[k][1], "<f4").astype(np.float64)
                print(f"    {k}: sum {v0.sum():,.1f} -> {v1.sum():,.1f}; cells nonzero {int((v0 > 0).sum()):,} -> {int((v1 > 0).sum()):,}")
        blocks = sorted(set(ho) | set(hn))
        print(f"    header blocks changed: {[k for k in blocks if k != 'arrays' and ho.get(k) != hn.get(k)]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nyc", "wdc"])
