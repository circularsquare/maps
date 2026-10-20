"""T-726: print an OSM PBF's header snapshot time (osmosis_replication_timestamp) without osmium.

    venv\\Scripts\\python.exe -I trainworld\\notes\\t726_pbfheader.py <file.osm.pbf> [...]

Reads only the first blob (a few hundred bytes).
"""
import datetime
import os
import struct
import sys
import zlib


def varint(b, i):
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        if not c & 0x80:
            return x, i
        s += 7


def fields(b):
    i = 0
    while i < len(b):
        k, i = varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = varint(b, i)
        elif w == 2:
            n, i = varint(b, i)
            v = b[i:i + n]
            i += n
        elif w == 1:
            v, i = b[i:i + 8], i + 8
        elif w == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(w)
        yield f, v


def snapshot(path):
    """(UTC snapshot time or None, replication base url or None) from the header blob."""
    with open(path, "rb") as fh:
        n = struct.unpack(">I", fh.read(4))[0]
        size = next(v for f, v in fields(fh.read(n)) if f == 3)
        blob = fh.read(size)
    raw = None
    for f, v in fields(blob):
        if f == 1:
            raw = v
        elif f == 3:
            raw = zlib.decompress(v)
    ts = url = None
    for f, v in fields(raw):
        if f == 32:
            ts = datetime.datetime.fromtimestamp(v, datetime.timezone.utc)
        elif f == 34:
            url = v.decode()
    return ts, url


if __name__ == "__main__":
    for p in sys.argv[1:]:
        ts, url = snapshot(p)
        print(os.path.basename(p), ts.isoformat() if ts else None, url)
