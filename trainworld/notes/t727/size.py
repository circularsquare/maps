"""Size of a compact routing-graph file for a region, as it might ship.

    python size.py jp_all us_all de_all ...

Per chain: two junction ids, one byte kind/rank/flags, vertex count, then its vertices
Douglas-Peucker'd to 3 m, quantised to 1e-5 degrees, delta-coded as zigzag varints; gzip.
Variants: A = rail, narrow gauge, light rail, subway, monorail, every rank (yards included);
B = A without service track (yards, sidings, crossovers) that no passenger route uses.
"""
import os
import sys
import gzip
import math
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parents[3] / "artifacts" / "trainworld" / "t727"  # outputs, not notes/


def varint(out, v):
    v = int(v)
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def zz(v):
    return (v << 1) ^ (v >> 63)


def dp_keep(x, y, tol):
    n = x.size
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    st = [(0, n - 1)]
    while st:
        i, j = st.pop()
        if j <= i + 1:
            continue
        dx, dy = x[j] - x[i], y[j] - y[i]
        L2 = dx * dx + dy * dy
        sx, sy = x[i + 1:j] - x[i], y[i + 1:j] - y[i]
        t = np.clip((sx * dx + sy * dy) / L2, 0, 1) if L2 > 0 else np.zeros(sx.size)
        d = np.hypot(sx - t * dx, sy - t * dy)
        m = int(np.argmax(d))
        if d[m] > tol:
            keep[i + 1 + m] = True
            st += [(i, i + 1 + m), (i + 1 + m, j)]
    return keep


for name in sys.argv[1:]:
    g = np.load(HERE / f"graph_{name}.npz")
    a, b, off, kind, rank, flags = g["a"], g["b"], g["off"], g["kind"], g["rank"], g["flags"]
    lon, lat = g["lon"], g["lat"]
    X = lon * np.cos(np.radians(lat)) * 111320
    Y = lat * 110950
    qlon = np.round(lon * 1e5).astype(np.int64)
    qlat = np.round(lat * 1e5).astype(np.int64)
    seg = np.hypot(np.diff(X), np.diff(Y))
    cum = np.concatenate(([0], np.cumsum(seg)))
    clen = cum[off[1:] - 1] - cum[off[:-1]]
    baseA = np.isin(kind, [0, 6, 1, 2, 4])
    variants = {"A": baseA, "B": baseA & ~((rank == 3) & ((flags & 1) == 0))}
    for vn, ok in variants.items():
        out = bytearray()
        nv = 0
        for c in np.flatnonzero(ok):
            s, e = off[c], off[c + 1]
            k = dp_keep(X[s:e], Y[s:e], 3.0)
            qx, qy = qlon[s:e][k], qlat[s:e][k]
            varint(out, a[c])
            varint(out, b[c])
            out.append(int(kind[c]) | (int(rank[c]) << 3) | ((int(flags[c]) & 7) << 5))
            varint(out, qx.size)
            px, py = 0, 0
            for xx, yy in zip(qx.tolist(), qy.tolist()):
                varint(out, zz(xx - px))
                varint(out, zz(yy - py))
                px, py = xx, yy
            nv += qx.size
        gz = len(gzip.compress(bytes(out), 6))
        ends = np.concatenate([a[ok], b[ok]])
        deg = np.bincount(ends)
        print(f"{name} {vn}: {ok.sum()} chains, {clen[ok].sum()/1000:,.0f} km, junctions deg3+ "
              f"{np.sum(deg >= 3)}, {nv} vertices after 3 m DP (of {np.sum(np.diff(off)[ok])}); "
              f"{len(out)/1e6:.1f} MB raw, {gz/1e6:.1f} MB gzip", flush=True)
