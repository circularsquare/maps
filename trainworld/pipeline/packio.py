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


def put_arrays(stem, header, arrays, block_key=None, block=None):
    """Write `arrays` {name: (values, dtype, per)} into the pack at `stem`, .bin first: over an
    array of that name and byte size in place, else appended (a replaced array's old bytes stay
    until the next rewrite); with `block_key`, header[block_key] = block. `header` is not changed."""
    with open(stem + ".bin", "rb") as f:
        buf = bytearray(f.read())
    header = json.loads(json.dumps(header))
    np_dt = {"f32": "<f4", "f64": "<f8", "u8": "u1", "u16": "<u2", "u32": "<u4", "u64": "<u8", "i32": "<i4"}
    for name, (values, dtype, per) in arrays.items():
        v = np.ascontiguousarray(values, dtype=np_dt[dtype]).tobytes()
        old = [e for e in header["arrays"] if e["name"] == name]
        if old and old[0]["dtype"] == dtype and old[0]["count"] * SIZES[dtype] == len(v):
            buf[old[0]["offset"]:old[0]["offset"] + len(v)] = v
            continue
        header["arrays"] = [e for e in header["arrays"] if e["name"] != name]
        buf += b"\0" * (-len(buf) % 8)
        header["arrays"].append({"name": name, "dtype": dtype, "per": per, "offset": len(buf), "count": len(values)})
        buf += v
    if block_key is not None:
        header[block_key] = block
    write(stem, header, buf)
    return header


def replace_cell_arrays(header, buf, replacements, per="cell"):
    """The .bin with the arrays named in `replacements` ({name: per-cell f32 values, or per `per`})
    dropped and written again at its end, every other array's bytes kept exactly; header["arrays"]
    to match. Per-cell arrays of a merged pack are refused: the step runs on the res-9 pack (T-1261)."""
    if per == "cell" and "merged" in header:
        raise SystemExit(f"{header.get('city')}: merged; this step runs on the res-9 pack: merge_cells.py "
                         f"<id> --through \"<module> {{id}} <args>\" (T-1261)")
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
