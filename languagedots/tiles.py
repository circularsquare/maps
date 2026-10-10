"""
Build data/processed/languagedots.pmtiles and counts.json from the scattered dots.

religiondots' tiles.py, cut down: same merge, same encoder (mvt.py, copied unchanged), no
roll-up, coverage or second edition yet. Read religiondots/tiles.py for the reasoning; the short
version is that nearby dots of one language are MERGED into one mark carrying `k` rather than
dropped, so nobody disappears at low zoom, and every feature carries `c`, its country, so a
merge never crosses a border.

One layer, `marks`, holding the Aggregation control's four merge grids (`dotsm1`, `dots`, `dots1`,
`dots2`, see DETAIL_LAYERS) together: a mark that comes out the same on several grids (a cell
holding one language mostly does) is stored once, and `l` is a bitmask of the grids that draw it
(bit i = DETAIL_LAYERS[i]: 1 dotsm1, 2 dots, 4 dots1, 8 dots2). The viewer keeps the marks whose
`l` has the current step's bit, which is exactly the old per-step layer. Each feature also has
n (language), c (country), p (people), t (people of all languages in its cell) and z (tile zoom).
Languages too small for one dot are in them too, at their own weight; there is no ring layer.
Point extent 1024 and gzip 9 (2026-10-06, Anita "let's try shrinking archive"; followups.md
"Archive size": the shared layer saved ~34%, 1024 ~5%, gzip 9 ~2%, nothing visible lost).

Usage:
    python tiles.py --countries in,np,pk --max-zoom 12     # the viewer's ARCHIVE_MAXZ must match
    python tiles.py ... --out <dir>/x.pmtiles              # a trial: archive and counts.json there
"""
import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pmtiles.writer import Writer
from pmtiles.tile import Compression, TileType, zxy_to_tileid

import mvt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
OUT = PROC / "languagedots.pmtiles"
CELL_BITS = 6        # 64x64 merge cells per tile, ~8px; religiondots' 16px read as a lattice here
# The viewer's Aggregation control (spec §4.2) picks one of these layers: the same dots merged
# on a grid 2x coarser, 1x, 2x or 4x finer than CELL_BITS, so 1/4x, 1x, 4x or 16x as many marks.
# The name's suffix is the offset from CELL_BITS (`dotsm1` is -1); the viewer keys on it.
# MapLibre insists on 512px vector tiles, so "fetch deeper tiles" cannot be done in the viewer
# and is baked in here instead.
DETAIL_LAYERS = [("dotsm1", CELL_BITS - 1), ("dots", CELL_BITS), ("dots1", CELL_BITS + 1),
                 ("dots2", CELL_BITS + 2)]
# 1/1024 of a tile is half a pixel at the tile's own zoom, and under a tenth of the finest merge
# cell (256 per tile), so every cell still has its own pixels and no two marks of one grid merge
EXTENT = 1024
GZIP_LEVEL = 9
L_T = [mvt.value_int(v) for v in range(1 << len(DETAIL_LAYERS))]   # `l`'s value table
POPCOUNT = np.array([bin(v).count("1") for v in range(1 << len(DETAIL_LAYERS))])
SHUFFLE_SEED = 20261004


def load(path: Path, cc: str, people=None):
    """Points with `p`, the people each stands for: 1,000 for a dot, or a ring's own count.

    A ring (a language too small to reach one dot in the country) is drawn as a dot of its true
    weight, since mark area is proportional to people (spec §4.1): 300 speakers are a mark
    three-tenths the area of a dot. Anita, 2026-10-04: no separate ring symbol."""
    feats = json.loads(path.read_text(encoding="utf-8"))["features"]
    coords = np.array([f["geometry"]["coordinates"] for f in feats], dtype=float).reshape(-1, 2)
    df = pd.DataFrame({"lon": coords[:, 0], "lat": coords[:, 1],
                       "n": [f["properties"]["n"] for f in feats], "c": cc,
                       "p": [people or f["properties"]["count"] for f in feats]})
    # a ring of 0 people draws nothing, and its cell's people-weighted mean is 0/0 (Brazil's
    # unscattered small languages, 2026-10-04)
    df = df[df["p"] > 0].reset_index(drop=True)
    df["wx"] = (df["lon"] + 180.0) / 360.0
    s = np.sin(np.radians(df["lat"].clip(-85.05112878, 85.05112878)))
    df["wy"] = 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)
    return df


def merge_at_zoom(nc, cc, pp, wx, wy, z, bits=CELL_BITS):
    """One mark per (cell, language, country): people `p`, people-weighted mean position, and
    `t`, the people of EVERY language in that cell and country. The viewer caps a cell's total
    ink with `t` and shares it out by p/t, so a full cell shrinks all its languages alike
    (spec §4.1) instead of capping each mark on its own, which over-inked the small ones."""
    n = 1 << (z + bits)
    cx = np.minimum((wx * n).astype(np.int64), n - 1)
    cy = np.minimum((wy * n).astype(np.int64), n - 1)
    nspace, cspace = int(nc.max()) + 1, int(cc.max()) + 1
    cell = cx * n + cy
    key = (cell * nspace + nc) * cspace + cc
    codes, uniq = pd.factorize(key)
    p = np.bincount(codes, weights=pp, minlength=len(uniq))
    mx = np.bincount(codes, weights=wx * pp, minlength=len(uniq)) / p
    my = np.bincount(codes, weights=wy * pp, minlength=len(uniq)) / p
    u = uniq.astype(np.int64)
    u, o_cc = np.divmod(u, cspace)
    o_cell, o_nc = np.divmod(u, nspace)
    tcodes, _ = pd.factorize(o_cell * cspace + o_cc)
    t = np.bincount(tcodes, weights=p)[tcodes]
    return dict(tx=(o_cell // n) >> bits, ty=(o_cell % n) >> bits, wx=mx, wy=my,
                p=np.rint(p).astype(np.int64), t=np.rint(t).astype(np.int64), n=o_nc, c=o_cc)


def tile_pixels(wx, wy, tx, ty, z):
    nt = 1 << z
    px = np.clip((wx * nt - tx) * EXTENT, 0, EXTENT - 1).astype(np.int64)
    py = np.clip((wy * nt - ty) * EXTENT, 0, EXTENT - 1).astype(np.int64)
    return px, py


def bucket(tx, ty, z, rng):
    key = tx.astype(np.int64) * (1 << z) + ty.astype(np.int64)
    perm = rng.permutation(len(key))
    order = perm[np.argsort(key[perm], kind="stable")]
    return key[order], order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", required=True)
    ap.add_argument("--max-zoom", type=int, default=10)
    ap.add_argument("--refresh-meta", action="store_true",
                    help="rewrite only the per-country display text in counts.json")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="the archive; counts.json goes beside it (a trial build elsewhere)")
    args = ap.parse_args()
    out = args.out
    from countries import COUNTRIES
    ccs = [c.strip() for c in args.countries.split(",")]

    def meta(cc):
        m = COUNTRIES[cc]
        # `parts`: the viewer's which-source-draws-which-part list (countries.py docstring)
        return {k: m.get(k, "") for k in ("name", "source", "how", "grain", "gap", "note_public")} | \
               {"view": m.get("view"), "parts": m.get("parts")}

    if args.refresh_meta:
        counts = json.loads((PROC / "counts.json").read_text(encoding="utf-8"))
        for cc, e in counts["countries"].items():
            e.update(meta(cc))
        (PROC / "counts.json").write_text(json.dumps(counts, ensure_ascii=False), encoding="utf-8")
        print("refreshed counts.json")
        return

    dots = pd.concat([load(PROC / f"dots_{cc}.geojson", cc, people=1000) for cc in ccs],
                     ignore_index=True)
    rframes = [load(PROC / f"rings_{cc}.geojson", cc) for cc in ccs
               if (PROC / f"rings_{cc}.geojson").exists()]
    rings = pd.concat(rframes, ignore_index=True) if rframes else dots.iloc[:0]
    print(f"{len(dots):,} dots, {len(rings):,} sub-dot languages drawn at their own weight")

    per = {}
    for cc in ccs:
        d, r = dots[dots["c"] == cc], rings[rings["c"] == cc]
        b = d if len(d) else r   # a rings-only country (the Vatican) has no dots to frame
        per[cc] = meta(cc) | {
            "bbox": [float(b["lon"].min()), float(b["lat"].min()),
                     float(b["lon"].max()), float(b["lat"].max())],
            "dots": d["n"].value_counts().to_dict(),
            # people per language in marks other than whole dots: a sub-dot language's one
            # ring, and since 2026-10-08 every unit's leftover (scatter.py), many per language
            "rings": {n: int(p) for n, p in r.groupby("n")["p"].sum().items()},
        }
    (out.parent / "counts.json").write_text(json.dumps({"dot_value": 1000, "countries": per},
                                                 ensure_ascii=False, allow_nan=False),
                                      encoding="utf-8")   # a NaN here breaks the whole page

    node_vocab = pd.Index(sorted(set(dots["n"]) | set(rings["n"])))
    cc_vocab = pd.Index(sorted(ccs))
    NODE_T = [mvt.value_str(s) for s in node_vocab]
    CC_T = [mvt.value_str(s) for s in cc_vocab]
    allp = pd.concat([dots, rings], ignore_index=True)
    nc = node_vocab.get_indexer(allp["n"]).astype(np.int64)
    cc_ = cc_vocab.get_indexer(allp["c"]).astype(np.int64)
    pp = allp["p"].to_numpy(dtype=float)
    wx, wy = allp["wx"].to_numpy(), allp["wy"].to_numpy()

    tmp =out.with_suffix(".pmtiles.tmp")
    n_tiles = 0
    with open(tmp, "wb") as f:
        w = Writer(f)
        for z in range(0, args.max_zoom + 1):
            frames, report = [], []
            # `z` is the tile's own zoom, on every feature: the viewer needs it to size a cell's
            # cap in pixels at whatever zoom the tile is being drawn at (spec §4.1).
            Z_T = [mvt.value_int(z)]
            nt = 1 << z
            for i, (lname, bits) in enumerate(DETAIL_LAYERS):
                m = merge_at_zoom(nc, cc_, pp, wx, wy, z, bits)
                px, py = tile_pixels(m["wx"], m["wy"], m["tx"], m["ty"], z)
                # tile, x and y in one integer: tile key < 2^24 at z12, x and y < 2^10
                pos = (((m["tx"].astype(np.int64) * nt + m["ty"]) << 20) | (px << 10) | py)
                frames.append(pd.DataFrame({"pos": pos, "n": m["n"], "c": m["c"], "p": m["p"],
                                            "t": m["t"], "l": np.int64(1 << i)}))
                report.append(f"{lname} {len(pos):,}")
            # one row per distinct mark; `l` sums the grids' bits, so it is their OR as long as
            # no grid has the same mark twice, which `k` (how many rows were summed) checks
            g = pd.concat(frames, ignore_index=True).groupby(["pos", "n", "c", "p", "t"],
                                                             sort=False)["l"]
            d = pd.concat([g.sum(), g.size().rename("k")], axis=1).reset_index()
            del frames, g
            lv, kv = d["l"].to_numpy(), d["k"].to_numpy()
            if (POPCOUNT[lv] != kv).any():
                raise SystemExit(f"z{z}: a grid holds the same mark twice; `l` would be wrong")
            pos = d["pos"].to_numpy()
            tkey = pos >> 20
            px, py = (pos >> 10) & 1023, pos & 1023
            pc, p_table = mvt.intern(d["p"].to_numpy(), kind="int")
            tc, t_table = mvt.intern(d["t"].to_numpy(), kind="int")
            # in position order within each tile (2026-10-07): neighbouring marks then share
            # most of their bytes, and big low-zoom tiles gzip about half smaller than shuffled.
            # The draw order of equal marks no longer comes from here; the viewer breaks those
            # ties on a hash of position and language (index.html, markHash).
            o = np.argsort(pos, kind="stable")
            key = tkey[o]
            codes = np.stack([d["n"].to_numpy(), d["c"].to_numpy(), pc, tc, lv,
                              np.zeros(len(d), dtype=np.int64)], axis=1)[o]
            keys = ["n", "c", "p", "t", "l", "z"]
            tables = [NODE_T, CC_T, p_table, t_table, L_T, Z_T]
            px, py = px[o], py[o]
            report.append(f"stored {len(d):,}")
            del d
            touched, starts = np.unique(key, return_index=True)
            ends = np.append(starts[1:], len(key))
            tids = np.fromiter((zxy_to_tileid(z, int(k // nt), int(k % nt)) for k in touched),
                               dtype=np.int64, count=len(touched))
            for i in np.argsort(tids):
                a, b = starts[i], ends[i]
                payload = [("marks", px[a:b], py[a:b], keys, codes[a:b], tables)]
                w.write_tile(int(tids[i]),
                             gzip.compress(mvt.encode(payload, EXTENT), GZIP_LEVEL, mtime=0))
            n_tiles += len(touched)
            print(f"  z{z:<2} {len(touched):>6,} tiles, marks: " + ", ".join(report), flush=True)
        w.finalize(
            {"tile_type": TileType.MVT, "tile_compression": Compression.GZIP,
             "min_zoom": 0, "max_zoom": args.max_zoom,
             "min_lon_e7": int(dots["lon"].min() * 1e7), "min_lat_e7": int(dots["lat"].min() * 1e7),
             "max_lon_e7": int(dots["lon"].max() * 1e7), "max_lat_e7": int(dots["lat"].max() * 1e7),
             "center_zoom": 4, "center_lon_e7": int(dots["lon"].mean() * 1e7),
             "center_lat_e7": int(dots["lat"].mean() * 1e7)},
            {"name": "languagedots",
             "vector_layers": [{"id": "marks", "fields": {"n": "String", "c": "String",
                                                          "p": "Number", "t": "Number",
                                                          "l": "Number", "z": "Number"}}]})
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n_tiles:,} tiles) and counts.json")


if __name__ == "__main__":
    main()
