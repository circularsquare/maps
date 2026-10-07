"""Measure languagedots.pmtiles: bytes per layer per zoom, attribute share, duplicates, codecs.

Reads a COPY of the archive (holding the live one open on Windows blocks the build's rename).
"""
import gzip
import json
import random
import sys
import time
import zlib
from collections import defaultdict

import brotli
from pmtiles.reader import Reader, MmapSource, all_tiles
from pmtiles.tile import tileid_to_zxy

PATH = sys.argv[1]
SAMPLE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
random.seed(1)


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if c < 0x80:
            return r, i
        s += 7


def fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i); yield f, wt, v, None
        elif wt == 2:
            ln, i = varint(b, i); yield f, wt, None, b[i:i + ln]; i += ln
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8


def packed(b):
    out, i = [], 0
    while i < len(b):
        v, i = varint(b, i); out.append(v)
    return out


def write_varint(v):
    out = bytearray()
    while True:
        c = v & 0x7F; v >>= 7
        if v:
            out.append(c | 0x80)
        else:
            out.append(c); return bytes(out)


def ld(tag, payload):
    return bytes([tag]) + write_varint(len(payload)) + payload


def parse_layer(lb):
    name, feats, keys, vals, extent = None, [], [], [], 4096
    for f, wt, v, sub in fields(lb):
        if f == 1: name = bytes(sub).decode()
        elif f == 2:
            tags = geom = None
            for ff, _, _, s in fields(sub):
                if ff == 2: tags = packed(s)
                elif ff == 4: geom = packed(s)
            feats.append((tags, geom))
        elif f == 3: keys.append(bytes(sub).decode())
        elif f == 4: vals.append(bytes(sub))
        elif f == 5: extent = v
    return name, feats, keys, vals, extent


def encode_layer(name, feats, keys, vals, extent, drop=(), shift=0):
    """Re-encode a layer, optionally dropping keys and coarsening coordinates by `shift` bits."""
    keep = [j for j, k in enumerate(keys) if k not in drop]
    # renumber values that survive
    used = sorted({t[2 * q + 1] for t, _ in feats for q in range(len(t) // 2) if t[2 * q] in keep})
    vmap = {v: i for i, v in enumerate(used)}
    kmap = {k: i for i, k in enumerate(keep)}
    out = bytearray(ld(0x0a, name.encode()))
    for tags, geom in feats:
        nt = []
        for q in range(len(tags) // 2):
            if tags[2 * q] in kmap:
                nt += [kmap[tags[2 * q]], vmap[tags[2 * q + 1]]]
        g = list(geom)
        if shift:
            x = (g[1] >> 1) ^ -(g[1] & 1); y = (g[2] >> 1) ^ -(g[2] & 1)
            x >>= shift; y >>= shift
            g = [9, (x << 1) ^ (x >> 63), (y << 1) ^ (y >> 63)]
        body = ld(0x12, b"".join(write_varint(t) for t in nt)) + b"\x18\x01" + \
            ld(0x22, b"".join(write_varint(t) for t in g))
        out += ld(0x12, body)
    for j in keep:
        out += ld(0x1a, keys[j].encode())
    for v in used:
        out += ld(0x22, vals[v])
    out += b"\x28" + write_varint(extent >> shift) + b"\x78\x02"
    return ld(0x1a, bytes(out))


def gz(b, lvl=6):
    return len(gzip.compress(b, lvl, mtime=0))


def main():
    t0 = time.time()
    with open(PATH, "rb") as f:
        r = Reader(MmapSource(f))
        hdr = r.header()
        print(json.dumps({k: str(v) for k, v in hdr.items()}))
        stored = defaultdict(int)        # zoom -> stored (compressed) bytes
        ntiles = defaultdict(int)
        lraw = defaultdict(int)          # (layer, z) -> raw bytes
        lgz = defaultdict(int)           # (layer, z) -> gzip-6 bytes of the layer alone
        lfeat = defaultdict(int)
        samp = defaultdict(lambda: defaultdict(int))   # variant -> z -> bytes (sampled tiles)
        dup = defaultdict(lambda: defaultdict(int))    # (z) -> counters
        k = 0
        for (z, x, y), data in all_tiles(r.get_bytes):
            k += 1
            stored[z] += len(data)
            ntiles[z] += 1
            raw = gzip.decompress(data)
            lbs = []
            for fno, _, _, lb in fields(raw):
                if fno != 3: continue
                lb = bytes(lb)
                _, j = varint(lb, 0)          # first field is the name
                ln, j = varint(lb, j)
                name = lb[j:j + ln].decode()
                lbs.append(lb)
                lraw[(name, z)] += len(lb)
                lgz[(name, z)] += gz(lb)
            if random.random() < SAMPLE:
                layers = {}
                for lb in lbs:
                    name, feats, keys, vals, ext = parse_layer(lb)
                    layers[name] = (feats, keys, vals, ext, lb)
                    lfeat[(name, z)] += len(feats)
                s = samp
                s["stored_gz6"][z] += len(data)
                s["raw"][z] += len(raw)
                s["gz9"][z] += gz(raw, 9)
                s["br11"][z] += len(brotli.compress(raw, quality=11))
                s["br9"][z] += len(brotli.compress(raw, quality=9))
                variants = {"no_z": dict(drop=("z",)), "no_z_t": dict(drop=("z", "t")),
                            "no_c": dict(drop=("c",)), "no_z_c": dict(drop=("z", "c")),
                            "ext1024": dict(shift=2), "ext512": dict(shift=3),
                            "no_z_ext1024": dict(drop=("z",), shift=2)}
                for vname, kw in variants.items():
                    blob = b"".join(encode_layer(n, L[0], L[1], L[2], L[3], **kw)
                                    for n, L in layers.items())
                    s[vname][z] += gz(blob)
                # sanity: re-encoding unchanged reproduces size
                blob = b"".join(encode_layer(n, L[0], L[1], L[2], L[3]) for n, L in layers.items())
                s["reenc_same"][z] += gz(blob)
                # duplicates: same geometry + n,c,p between adjacent layers
                def keyset(n):
                    if n not in layers: return set()
                    feats, keys, vals, _, _ = layers[n]
                    ki = {kk: j for j, kk in enumerate(keys)}
                    out = set()
                    for tags, geom in feats:
                        d = {tags[2 * q]: vals[tags[2 * q + 1]] for q in range(len(tags) // 2)}
                        out.add((tuple(geom), d.get(ki["n"]), d.get(ki["c"]), d.get(ki["p"])))
                    return out
                ks = {n: keyset(n) for n in ("dotsm1", "dots", "dots1", "dots2")}
                allm = set().union(*ks.values())
                dup[z]["marks_total"] += sum(len(v) for v in ks.values())
                dup[z]["marks_unique"] += len(allm)
            if k % 20000 == 0:
                print(f"  {k:,} tiles, {time.time() - t0:.0f}s", flush=True)
    res = dict(stored=stored, ntiles=ntiles,
               lraw={f"{a}|{b}": v for (a, b), v in lraw.items()},
               lgz={f"{a}|{b}": v for (a, b), v in lgz.items()},
               lfeat={f"{a}|{b}": v for (a, b), v in lfeat.items()},
               samp={k: dict(v) for k, v in samp.items()},
               dup={k: dict(v) for k, v in dup.items()})
    json.dump(res, open(PATH + ".measure.json", "w"), indent=1)
    print(f"done {k:,} tiles in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
