"""Scratch copy of languagedots/tiles.py's archive loop, to size compression options.

Reads dots/rings from languagedots/data/processed (read-only), writes only to this folder.
    python trial_tiles.py <countries> <variant> [max_zoom]
variants: base, z11 (max zoom 11), noz (no `z` attribute), ext1024, two (dots + dots2 only),
          br (brotli 11 instead of gzip 6), gz9, shared (one layer, identical marks shared,
          `l` = bitmask of the aggregation steps that draw it), lean (shared + noz + ext1024)
"""
import gzip
import os
import sys
import time

sys.dont_write_bytecode = True
LD = r"C:\Users\anita\projects\maps\languagedots"
sys.path.insert(0, LD)
os.chdir(LD)

import numpy as np
import pandas as pd
import brotli
from pmtiles.writer import Writer
from pmtiles.tile import Compression, TileType, zxy_to_tileid

import mvt
import tiles as T          # functions only; main() is not run

import tempfile
HERE = tempfile.gettempdir()   # the trial archive is written here and deleted
ccs = sys.argv[1].split(",")
variant = sys.argv[2]
maxz = int(sys.argv[3]) if len(sys.argv) > 3 else 12
if variant in ("z11", "max", "maxgz"):
    maxz = 11
EXTENT = 1024 if variant in ("ext1024", "lean", "max", "maxgz") else 4096
LAYERS = T.DETAIL_LAYERS
if variant == "two":
    LAYERS = [L for L in LAYERS if L[0] in ("dots", "dots2")]
if variant == "no_m1":
    LAYERS = [L for L in LAYERS if L[0] != "dotsm1"]
if variant == "no_d2":
    LAYERS = [L for L in LAYERS if L[0] != "dots2"]
NOZ = variant in ("noz", "lean", "max", "maxgz")
SHARED = variant in ("shared", "lean", "max", "maxgz")


def compress(b):
    if variant in ("br", "max"):
        return brotli.compress(b, quality=11)
    return gzip.compress(b, 9 if variant == "gz9" else 6, mtime=0)


def main():
    t0 = time.time()
    dots = pd.concat([T.load(T.PROC / f"dots_{cc}.geojson", cc, people=1000) for cc in ccs],
                     ignore_index=True)
    rf = [T.load(T.PROC / f"rings_{cc}.geojson", cc) for cc in ccs
          if (T.PROC / f"rings_{cc}.geojson").exists()]
    rings = pd.concat(rf, ignore_index=True) if rf else dots.iloc[:0]
    node_vocab = pd.Index(sorted(set(dots["n"]) | set(rings["n"])))
    cc_vocab = pd.Index(sorted(ccs))
    NODE_T = [mvt.value_str(s) for s in node_vocab]
    CC_T = [mvt.value_str(s) for s in cc_vocab]
    allp = pd.concat([dots, rings], ignore_index=True)
    nc = node_vocab.get_indexer(allp["n"]).astype(np.int64)
    cc_ = cc_vocab.get_indexer(allp["c"]).astype(np.int64)
    pp = allp["p"].to_numpy(dtype=float)
    wx, wy = allp["wx"].to_numpy(), allp["wy"].to_numpy()
    rng = np.random.default_rng(T.SHUFFLE_SEED)
    out = os.path.join(HERE, f"trial_{variant}.pmtiles")
    per_z = {}
    with open(out, "wb") as f:
        w = Writer(f)
        for z in range(0, maxz + 1):
            Z_T = [mvt.value_int(z)]
            ms = []
            for lname, bits in LAYERS:
                m = T.merge_at_zoom(nc, cc_, pp, wx, wy, z, bits)
                nt = 1 << z
                px = np.clip((m["wx"] * nt - m["tx"]) * EXTENT, 0, EXTENT - 1).astype(np.int64)
                py = np.clip((m["wy"] * nt - m["ty"]) * EXTENT, 0, EXTENT - 1).astype(np.int64)
                ms.append((lname, m, px, py))
            layers = []
            if SHARED:
                # one row per distinct (tile, px, py, n, c, p, t); `l` ORs the steps drawing it
                frames = []
                for i, (lname, m, px, py) in enumerate(ms):
                    frames.append(pd.DataFrame(dict(tx=m["tx"], ty=m["ty"], px=px, py=py, n=m["n"],
                                                    c=m["c"], p=m["p"], t=m["t"], l=1 << i)))
                d = pd.concat(frames, ignore_index=True)
                d = d.groupby(["tx", "ty", "px", "py", "n", "c", "p", "t"], sort=False,
                              as_index=False)["l"].sum()
                pc, p_table = mvt.intern(d["p"].to_numpy(), kind="int")
                tc, t_table = mvt.intern(d["t"].to_numpy(), kind="int")
                lc, l_table = mvt.intern(d["l"].to_numpy(), kind="int")
                key, o = T.bucket(d["tx"].to_numpy(), d["ty"].to_numpy(), z, rng)
                keys = ["n", "c", "p", "t", "l"] + ([] if NOZ else ["z"])
                cols = [d["n"].to_numpy(), d["c"].to_numpy(), pc, tc, lc]
                tabs = [NODE_T, CC_T, p_table, t_table, l_table]
                if not NOZ:
                    cols.append(np.zeros(len(d), np.int64)); tabs.append(Z_T)
                layers.append(("marks", key, d["px"].to_numpy()[o], d["py"].to_numpy()[o], keys,
                               np.stack(cols, axis=1)[o], tabs))
            else:
                for lname, m, px, py in ms:
                    pc, p_table = mvt.intern(m["p"], kind="int")
                    tc, t_table = mvt.intern(m["t"], kind="int")
                    key, o = T.bucket(m["tx"], m["ty"], z, rng)
                    keys = ["n", "c", "p", "t"] + ([] if NOZ else ["z"])
                    cols = [m["n"], m["c"], pc, tc]
                    tabs = [NODE_T, CC_T, p_table, t_table]
                    if not NOZ:
                        cols.append(np.zeros(len(m["p"]), np.int64)); tabs.append(Z_T)
                    layers.append((lname, key, px[o], py[o], keys, np.stack(cols, axis=1)[o], tabs))
            touched = np.unique(np.concatenate([L[1] for L in layers]))
            nt = 1 << z
            tids = np.fromiter((zxy_to_tileid(z, int(k // nt), int(k % nt)) for k in touched),
                               dtype=np.int64, count=len(touched))
            zb = 0
            for i in np.argsort(tids):
                key = touched[i]
                payload = []
                for name, lkey, lpx, lpy, lkeys, lcodes, ltables in layers:
                    a = np.searchsorted(lkey, key, "left")
                    b = np.searchsorted(lkey, key, "right")
                    if b > a:
                        payload.append((name, lpx[a:b], lpy[a:b], lkeys, lcodes[a:b], ltables))
                blob = compress(mvt.encode(payload, EXTENT))
                zb += len(blob)
                w.write_tile(int(tids[i]), blob)
            per_z[z] = zb
        w.finalize({"tile_type": TileType.MVT,
                    "tile_compression": Compression.BROTLI if variant == "br" else Compression.GZIP,
                    "min_zoom": 0, "max_zoom": maxz, "min_lon_e7": 0, "min_lat_e7": 0,
                    "max_lon_e7": 0, "max_lat_e7": 0, "center_zoom": 0, "center_lon_e7": 0,
                    "center_lat_e7": 0}, {"name": "trial"})
    tot = os.path.getsize(out)
    print(f"{variant:8} {tot / 1e6:7.2f} MB  " +
          " ".join(f"z{z}:{b / 1e6:.2f}" for z, b in per_z.items()) + f"  ({time.time() - t0:.0f}s)")
    os.remove(out)


main()
