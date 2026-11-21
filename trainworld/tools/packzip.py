"""Compressed copies of the .bin packs for publishing (T-2486).

    python trainworld/tools/packzip.py pack <src packs> <out dir> [--all] [--jobs n]   write the copies
    python trainworld/tools/packzip.py check <src packs> <out dir> [--all]             inflate them and compare

A zone's `.bin` becomes one gzip member per array, back to back, and its header (`<id>.json`) gains
`bin_gz: {size, parts}`: the raw file's length and, for each member, [raw offset, offset in the file, length].
Members start at each array's offset (the first at 0) and run to the next, padding included, so they inflate
in order to the raw file byte for byte. `offset` and `count` in `arrays` stay the raw layout: the game inflates a
whole read back to it (demandClient's downloadPack) and maps a Range read onto the members covering it
(workers/rangeFetch.ts). A header without `bin_gz` is read as today, so raw packs keep loading.

`market-drives.bin` is only ever read whole: it is gzipped as one file, and the game inflates it on seeing the gzip
magic. No Content-Encoding on either: a range into a stream the browser inflates is not a range into the file.

The pipeline's writers and readers (nocar, enrich_*, merge_cells, pack_gravity) keep working on raw packs in
data/packs; this runs on a copy at publish time (tools/deploy.py). A pack whose arrays leave non-zero bytes
uncovered, or whose header points into the .bin from outside `arrays`, is refused.

Without --all only the registry's zones are written (what a build ships); other files are left out.
"""

import argparse
import gzip
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
REGISTRY = os.path.join(TW, "app", "src", "game", "registry.json")
DT = {"u8": 1, "i8": 1, "u16": 2, "i16": 2, "u32": 4, "i32": 4, "f32": 4, "u64": 8, "i64": 8, "f64": 8}
DRIVES = "market-drives.bin"
GZ_MAGIC = b"\x1f\x8b"
LEVEL = 9


def gz(b):
    """mtime 0: the same bytes give the same file, so rclone --checksum skips it next time."""
    return gzip.compress(b, LEVEL, mtime=0)


def stray_offsets(h, path=""):
    """Keys named offset outside `arrays` (anything else pointing into the .bin)."""
    out = []
    if isinstance(h, dict):
        for k, v in h.items():
            if k in ("arrays", "bin_gz") and not path:
                continue
            if k == "offset" and isinstance(v, (int, float)):
                out.append(path + "." + k)
            out += stray_offsets(v, path + "." + k)
    elif isinstance(h, list):
        for i, v in enumerate(h):
            out += stray_offsets(v, f"{path}[{i}]")
    return out


def members(h, raw):
    """The member starts: 0 and every array's offset. Raises if the arrays leave non-zero bytes uncovered."""
    spans = sorted((a["offset"], a["offset"] + a["count"] * DT[a["dtype"]]) for a in h["arrays"])
    at = 0
    for s, e in spans:
        if e > len(raw):
            raise ValueError(f"array at {s} runs past the end ({e} > {len(raw)})")
        if s > at and raw[at:s].strip(b"\0"):
            raise ValueError(f"non-zero bytes {at}-{s} outside every array")
        at = max(at, e)
    if raw[at:].strip(b"\0"):
        raise ValueError(f"non-zero bytes {at}-{len(raw)} after the last array")
    stray = stray_offsets(h)
    if stray:
        raise ValueError(f"offsets outside arrays: {', '.join(stray[:4])}")
    return sorted({0} | {s for s, _ in spans if s < len(raw)})


def write(path, data):
    """Through a .part renamed over the file: never written in place, so a hard link to a data file stays intact."""
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def pack_zone(src, out, zid):
    hp, bp = os.path.join(src, zid + ".json"), os.path.join(src, zid + ".bin")
    h = json.load(open(hp, encoding="utf-8"))
    raw = open(bp, "rb").read()
    if "bin_gz" in h:
        raise ValueError("already packed")
    starts = members(h, raw)
    parts, body = [], bytearray()
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(raw)
        m = gz(raw[s:e])
        parts.append([s, len(body), len(m)])
        body += m
    h["bin_gz"] = {"size": len(raw), "parts": parts}
    write(os.path.join(out, zid + ".bin"), bytes(body))
    write(os.path.join(out, zid + ".json"), json.dumps(h, separators=(",", ":")).encode("utf-8"))
    return zid, len(raw), len(body)


def unpack(h, body):
    z = h["bin_gz"]
    out = bytearray(z["size"])
    parts = z["parts"]
    for k, (s, zo, zl) in enumerate(parts):
        e = parts[k + 1][0] if k + 1 < len(parts) else z["size"]
        d = zlib.decompress(body[zo:zo + zl], 31)
        if len(d) != e - s:
            raise ValueError(f"member {k} inflates to {len(d)}, not {e - s}")
        out[s:e] = d
    return bytes(out)


def check_zone(src, out, zid):
    raw = open(os.path.join(src, zid + ".bin"), "rb").read()
    h0 = json.load(open(os.path.join(src, zid + ".json"), encoding="utf-8"))
    h = json.load(open(os.path.join(out, zid + ".json"), encoding="utf-8"))
    body = open(os.path.join(out, zid + ".bin"), "rb").read()
    back = unpack(h, body)
    if back != raw:
        raise ValueError("inflated file differs from the source")
    del h["bin_gz"]
    if h != h0:
        raise ValueError("header differs from the source beyond bin_gz")
    return zid, len(raw), len(body)


def zones(src, every):
    if every:
        return sorted(f[:-4] for f in os.listdir(src) if f.endswith(".bin") and f != DRIVES and os.path.exists(os.path.join(src, f[:-4] + ".json")))
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    return [c["id"] for c in reg["cities"] if os.path.exists(os.path.join(src, c["id"] + ".bin"))]


def run(fn, src, out, ids, jobs):
    tot_raw = tot_z = 0
    bad = []
    with ProcessPoolExecutor(jobs) as ex:
        futs = {ex.submit(fn, src, out, i): i for i in ids}
        for n, f in enumerate(futs):
            try:
                _, r, z = f.result()
                tot_raw += r
                tot_z += z
            except Exception as e:  # noqa: BLE001
                bad.append(f"{futs[f]}: {e}")
            if n % 200 == 0:
                print(f"   {n}/{len(ids)}", flush=True)
    return tot_raw, tot_z, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pack", "check"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--all", action="store_true", help="every pack in src, not only the registry's zones")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    if src == out:
        sys.exit("out must not be src: the packs there are the pipeline's")
    os.makedirs(out, exist_ok=True)
    ids = zones(src, a.all)
    fn = pack_zone if a.cmd == "pack" else check_zone
    r, z, bad = run(fn, src, out, ids, a.jobs)
    print(f"{a.cmd}: {len(ids) - len(bad)} zone packs, {r / 1e6:.1f} MB raw -> {z / 1e6:.1f} MB ({z / max(r, 1):.2f})")
    dsrc, dout = os.path.join(src, DRIVES), os.path.join(out, DRIVES)
    if os.path.exists(dsrc):
        raw = open(dsrc, "rb").read()
        if a.cmd == "pack":
            if raw[:2] == GZ_MAGIC:
                bad.append(f"{DRIVES}: already gzip")
            else:
                write(dout, gz(raw))
        elif zlib.decompress(open(dout, "rb").read(), 31) != raw:
            bad.append(f"{DRIVES}: inflates to something else")
        if os.path.exists(dout):
            print(f"{DRIVES}: {len(raw) / 1e6:.1f} MB -> {os.path.getsize(dout) / 1e6:.1f} MB")
    for b in bad:
        print("   REFUSED " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
