"""Check the built archive tile by tile against an independent reference implementation.

    python tools/check_tiles.py cz,pl,ro --max-zoom 9 --coarse

tiles.py stopped using mapbox_vector_tile on 2026-09-04 — it does 11,400 features/s and a
full build encodes about 28 million of them — and writes the MVT bytes itself (mvt.py).
That is a hand-rolled protobuf encoder in a project where nothing else is, so it needs a
check that does not share code with it.  This is that check.

It rebuilds what each tile SHOULD hold straight from the geojson, with the pre-2026-09-04
logic reproduced here in full — a pandas groupby for the merge, dict features, and
mapbox_vector_tile itself for the reference — then decodes the real archive and compares.

FEATURE ORDER IS NOT COMPARED, and must not be: the draw-order shuffle (§4.2) is a numpy
permutation now rather than random.Random, so the order legitimately differs.  Tiles are
compared as MULTISETS of (layer, x, y, properties), which still catches a dropped feature,
a duplicated one, a coordinate off by one, a mis-encoded varint or a property that landed
on the wrong value-table index.

DECODE WITH y_coord_down.  mapbox_vector_tile flips y on the way in AND on the way out,
and tiles.py suppresses the flip in both directions; decoding with the default instead
reports every single feature as a mismatch at 4096 - y, which looks like a catastrophe
and is a flag.
"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed")

CELL_BITS = 5
EXTENT = 4096


# ---- the pre-2026-09-04 implementation, kept verbatim as the reference --------------
def load(path, cc):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    lon, lat, node, extra, tier = [], [], [], [], []
    for ft in gj["features"]:
        c = ft["geometry"]["coordinates"]
        lon.append(c[0]); lat.append(c[1])
        node.append(ft["properties"]["n"])
        extra.append(ft["properties"].get("why", ""))
        tier.append(int(ft["properties"].get("t", 0)))
    df = pd.DataFrame({"lon": lon, "lat": lat, "n": node, "why": extra, "t": tier, "c": cc})
    df["wx"] = (df["lon"] + 180.0) / 360.0
    s = np.sin(np.radians(df["lat"].clip(-85.05112878, 85.05112878)))
    df["wy"] = 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)
    return df


def merge_at_zoom(df, z):
    n = 1 << (z + CELL_BITS)
    cx = np.minimum((df["wx"].to_numpy() * n).astype(np.int64), n - 1)
    cy = np.minimum((df["wy"].to_numpy() * n).astype(np.int64), n - 1)
    g = pd.DataFrame({"cx": cx, "cy": cy, "n": df["n"].to_numpy(), "c": df["c"].to_numpy(),
                      "t": df["t"].to_numpy(),
                      "wx": df["wx"].to_numpy(), "wy": df["wy"].to_numpy()})
    out = g.groupby(["cx", "cy", "n", "c", "t"], sort=False).agg(
        wx=("wx", "mean"), wy=("wy", "mean"), k=("wx", "size")).reset_index()
    out["tx"] = out["cx"].to_numpy() >> CELL_BITS
    out["ty"] = out["cy"].to_numpy() >> CELL_BITS
    return out


def to_tiles(rows, z, layer, sink, with_k):
    ntiles = 1 << z
    px = (rows["wx"].to_numpy() * ntiles - rows["tx"].to_numpy()) * EXTENT
    py = (rows["wy"].to_numpy() * ntiles - rows["ty"].to_numpy()) * EXTENT
    px = np.clip(px, 0, EXTENT - 1).astype(int)
    py = np.clip(py, 0, EXTENT - 1).astype(int)
    tx = rows["tx"].to_numpy(); ty = rows["ty"].to_numpy()
    node = rows["n"].to_numpy(); cc = rows["c"].to_numpy()
    k = rows["k"].to_numpy() if with_k else np.ones(len(rows), dtype=int)
    why = rows["why"].to_numpy() if "why" in rows else None
    tier = rows["t"].to_numpy() if "t" in rows else None
    for i in range(len(rows)):
        props = {"n": node[i], "c": cc[i]}
        if tier is not None and tier[i]:
            props["t"] = int(tier[i])
        if with_k:
            props["k"] = int(k[i])
        elif why is not None and why[i]:
            props["why"] = why[i]
        sink[(z, int(tx[i]), int(ty[i]))][layer].append({
            "geometry": {"type": "Point", "coordinates": [int(px[i]), int(py[i])]},
            "properties": props})


def bag(layers):
    """A tile as a multiset of (layer, xy, properties) — order deliberately discarded."""
    c = Counter()
    for name, feats in layers.items():
        for f in feats:
            xy = f["geometry"]["coordinates"]
            while isinstance(xy[0], (list, tuple)):
                xy = xy[0]
            props = tuple(sorted((k, int(v) if k in ("k", "t") else str(v))
                                 for k, v in f["properties"].items()))
            c[(name, tuple(xy), props)] += 1
    return c


def main():
    import mapbox_vector_tile
    from pmtiles.reader import Reader, MmapSource

    ap = argparse.ArgumentParser()
    ap.add_argument("countries", help="comma-separated, exactly as passed to tiles.py")
    ap.add_argument("--max-zoom", type=int, default=10)
    ap.add_argument("--min-zoom", type=int, default=0)
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--archive", default=os.path.join(PROC, "religiondots.pmtiles"))
    args = ap.parse_args()

    editions = [("", 1000)] + ([("_10k", 10000)] if args.coarse else [])
    sink = defaultdict(lambda: defaultdict(list))
    for suffix, dv in editions:
        dot_frames, ring_frames = [], []
        for cc in [c.strip() for c in args.countries.split(",")]:
            dot_frames.append(load(os.path.join(PROC, f"dots_{cc}{suffix}.geojson"), cc))
            rp = os.path.join(PROC, f"rings_{cc}{suffix}.geojson")
            if os.path.exists(rp):
                ring_frames.append(load(rp, cc))
        dots = pd.concat(dot_frames, ignore_index=True)
        rings = pd.concat(ring_frames, ignore_index=True) if ring_frames else None
        lp = "10k" if suffix else ""
        print(f"  1:{dv} reference: {len(dots):,} dots, "
              f"{0 if rings is None else len(rings):,} rings")
        for z in range(args.min_zoom, args.max_zoom + 1):
            to_tiles(merge_at_zoom(dots, z), z, f"dots{lp}", sink, with_k=True)
            nt = 1 << z
            a = dots.copy()
            a["tx"] = np.minimum((a["wx"] * nt).astype(np.int64), nt - 1)
            a["ty"] = np.minimum((a["wy"] * nt).astype(np.int64), nt - 1)
            a["k"] = 1
            to_tiles(a, z, f"atomic{lp}", sink, with_k=False)
            if rings is not None and len(rings):
                r = rings.copy()
                r["tx"] = np.minimum((r["wx"] * nt).astype(np.int64), nt - 1)
                r["ty"] = np.minimum((r["wy"] * nt).astype(np.int64), nt - 1)
                r["k"] = 1
                to_tiles(r, z, f"rings{lp}", sink, with_k=False)

    bad = 0
    with open(args.archive, "rb") as f:
        reader = Reader(MmapSource(f))
        for (z, x, y), layers in sorted(sink.items()):
            want = bag(layers)
            raw = reader.get(z, x, y)
            if raw is None:
                print(f"  z{z}/{x}/{y}: MISSING from the archive")
                bad += 1
                continue
            got = bag({k: v["features"] for k, v in mapbox_vector_tile.decode(
                gzip.decompress(raw), default_options={"y_coord_down": True}).items()})
            if want != got:
                bad += 1
                miss, extra = want - got, got - want
                print(f"  z{z}/{x}/{y}: MISMATCH — {sum(miss.values())} missing, "
                      f"{sum(extra.values())} unexpected")
                for item, n in list(miss.items())[:3]:
                    print(f"      want {n}x {item}")
                for item, n in list(extra.items())[:3]:
                    print(f"      got  {n}x {item}")
                if bad > 5:
                    print("  …stopping after six bad tiles")
                    break

    print(f"\n{len(sink):,} tiles checked, {bad} mismatched — "
          + ("IDENTICAL to the reference" if not bad else "DIFFERENT"))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
