"""Writing packs (format: notes/T-004.md), T-1114. read_pack.py reads them with code of its own, so
it tests the format rather than this writer.
"""
import json
import os

import numpy as np

SIZES = {"f32": 4, "f64": 8, "u8": 1, "u16": 2, "u32": 4, "u64": 8, "i32": 4}  # bytes per value


def _dump(path, header):
    # text mode on purpose: every pack header so far has its platform's line ends
    with open(path, "w", encoding="utf-8") as f:
        json.dump(header, f, indent=2)


def write(stem, header, buf):
    """<stem>.bin and <stem>.json, each through a .part file, the .bin replaced first."""
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    _dump(stem + ".json.part", header)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")


def write_header(path, header):
    _dump(path + ".part", header)
    os.replace(path + ".part", path)


def replace_cell_arrays(header, buf, replacements, per="cell"):
    """The .bin with the arrays named in `replacements` ({name: per-cell f32 values, or per `per`})
    dropped and written again at its end, every other array's bytes kept exactly; header["arrays"]
    to match."""
    out, arrays = bytearray(), []
    for a in header["arrays"]:
        if a["name"] in replacements:
            continue
        out += b"\0" * (-len(out) % 8)
        data = buf[a["offset"]:a["offset"] + a["count"] * SIZES[a["dtype"]]]
        if len(data) != a["count"] * SIZES[a["dtype"]]:
            raise ValueError(f"truncated array {a['name']}")
        arrays.append(dict(a, offset=len(out)))
        out += data
    for name, weights in replacements.items():
        out += b"\0" * (-len(out) % 8)
        arrays.append(dict(name=name, dtype="f32", per=per, offset=len(out), count=len(weights)))
        out += np.asarray(weights, dtype="<f4").tobytes()
    header["arrays"] = arrays
    return out
