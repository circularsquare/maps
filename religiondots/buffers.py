"""
Pack the scattered dots into flat binary buffers for the WebGL scatter layer.

WHY THIS EXISTS, and why it is not more tiles.

A tile pyramid is the right structure for data that THINS as you zoom out. This data does
not thin: §4.2 forbids dropping a dot, so a world view genuinely needs all 2.04M dots on
screen and tiling buys nothing at exactly the zoom that hurts. What it costs instead is
duplication — `atomic` stores every dot at every zoom, eleven times over, which is ~123 MB
of the 149.5 MB archive and essentially all of it.

Measured on the real archive, per feature on the wire, MVT is already about 5 bytes. So the
saving here is NOT a cleverer encoding; it is storing each dot ONCE, and not paying
MapLibre's per-feature cost to draw it (a circle is 4 vertices, a JS feature index entry,
and a filter plus paint expression re-evaluated per feature — `setFilter` re-parses every
loaded tile in the worker, which is the multi-second stall on every country switch).

    layer          where it lives after this
    dots/dots10k   PMTiles, unchanged — the merged pyramid is small and thins properly
    rings          PMTiles, unchanged
    atomic         HERE, as one buffer per country per edition

FORMAT.  Struct-of-arrays, little-endian, three sections back to back:

    x   uint32[n]     web mercator X, [0,1) mapped onto the full uint32 range
    y   uint32[n]     web mercator Y, likewise
    ni  uint16[n]     node index in the low 14 bits, tier (§7) in the top 2

10 bytes a dot. uint32 fixed-point rather than float32 on purpose: float32 mercator has an
ulp of about 1.2 m near the equator, which is a quarter pixel at z14 and four pixels at
z18 — a visible lattice. Fixed-point is ~1 cm everywhere, and costs the same 8 bytes.

(The matching hazard on the viewer side is worse and is NOT fixed here: MapLibre hands a
custom layer a float64 matrix that has to be downcast for `uniformMatrix4fv`, and at z18
that alone is several pixels of error. The viewer has to fold a local origin into the
matrix in float64 before downcasting. Noted here because the two fixes only work together.)

DRAW ORDER — §4.2c, and the reason the file is laid out in buckets.

§4.2c requires the visible dot at a pixel to be a uniform draw from the dots covering it,
which is why `tiles.py` shuffles each tile before encoding. Shuffling is exactly what makes
MVT expensive — measured on the z3 India tile, the shuffle costs 49% of the layer, because
delta-encoding is precisely what a shuffle destroys. It is worth every byte there and it is
free here, but only if the file is not simply one globally shuffled list: a global shuffle
would forbid viewport culling, chunked loading and any spatial compression at once.

So dots are Hilbert-sorted, cut into buckets of BUCKET dots, and then:

  * shuffled WITHIN each bucket — dots only overlap locally, so local uniformity is the
    whole of what §4.2c actually asks for; and
  * the buckets are written in RANDOM order, so where two buckets overlap, which one paints
    over the other is random rather than a function of position. Adjacent buckets have
    similar composition anyway, but a systematic rule there is how São Paulo came to read
    Spiritualist, and the fix costs nothing.

Each bucket carries a bbox, so the viewer draws one instanced call per visible bucket and
skips the rest. Both shuffles are seeded, so a build is reproducible.

Usage:
    python buffers.py --countries us,ca,in            # 1:1,000
    python buffers.py --countries us,ca,in --coarse   # also the 1:10,000 edition
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Windows consoles here are cp1252 and the country names are not — same guard as tiles.py.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
OUT = HERE / "data" / "buffers"

BUCKET = 4096           # dots per bucket: ~500 buckets for India, so culling is per-frame cheap
HILBERT_ORDER = 16      # 65536x65536 grid over the mercator square, ~600m at the equator
SEED = 20260904


def hilbert_d(x: np.ndarray, y: np.ndarray, order: int = HILBERT_ORDER) -> np.ndarray:
    """Hilbert index of each point on a 2^order grid. Vectorised; x, y are uint32 grid coords.

    The standard bit-by-bit rotation, lifted to numpy so 2M points cost milliseconds rather
    than a Python loop. Hilbert rather than Morton because Morton's big jumps across quadrant
    boundaries put spatially distant dots in one bucket, which would widen bucket bboxes and
    make viewport culling much weaker.
    """
    x = x.astype(np.uint64).copy()
    y = y.astype(np.uint64).copy()
    d = np.zeros(len(x), dtype=np.uint64)
    s = np.uint64(1) << np.uint64(order - 1)
    while s > 0:
        rx = ((x & s) > 0).astype(np.uint64)
        ry = ((y & s) > 0).astype(np.uint64)
        d += s * s * ((np.uint64(3) * rx) ^ ry)
        # rotate the quadrant so the curve stays continuous
        swap = ry == 0
        flip = swap & (rx == 1)
        x_f = np.where(flip, s - np.uint64(1) - x, x)
        y_f = np.where(flip, s - np.uint64(1) - y, y)
        x_n = np.where(swap, y_f, x_f)
        y_n = np.where(swap, x_f, y_f)
        x, y = x_n.astype(np.uint64), y_n.astype(np.uint64)
        s >>= np.uint64(1)
    return d


def load(path: Path):
    """dots_<cc>.geojson -> (lon, lat, node id strings, tier). Same shape as tiles.py's load."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    n = len(feats)
    lon = np.empty(n, dtype=np.float64)
    lat = np.empty(n, dtype=np.float64)
    node = np.empty(n, dtype=object)
    tier = np.zeros(n, dtype=np.uint8)
    for i, ft in enumerate(feats):
        c = ft["geometry"]["coordinates"]
        lon[i] = c[0]
        lat[i] = c[1]
        p = ft["properties"]
        node[i] = p["n"]
        tier[i] = int(p.get("t", 0))        # §7; absent = measured
    return lon, lat, node, tier


def mercator(lon: np.ndarray, lat: np.ndarray):
    """lon/lat -> web mercator in [0,1), as float64. Clipped to the mercator latitude limit."""
    wx = (lon + 180.0) / 360.0
    s = np.sin(np.radians(np.clip(lat, -85.05112878, 85.05112878)))
    wy = 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)
    return wx, wy


def pack(cc: str, suffix: str, vocab: dict, rng: np.random.Generator):
    """Write one country/edition buffer. Returns its manifest entry, or None if absent."""
    src = PROC / f"dots_{cc}{suffix}.geojson"
    if not src.exists():
        return None
    lon, lat, node, tier = load(src)
    n = len(lon)
    wx, wy = mercator(lon, lat)

    # uint32 fixed point. nextafter keeps a dot at exactly 1.0 from wrapping to 0.
    FULL = float(1 << 32)
    xi = np.minimum((wx * FULL).astype(np.uint64), (1 << 32) - 1).astype(np.uint32)
    yi = np.minimum((wy * FULL).astype(np.uint64), (1 << 32) - 1).astype(np.uint32)

    # Hilbert order on a coarser grid than the stored precision — the curve only has to
    # group neighbours into buckets, not to distinguish dots within one.
    gx = (xi >> (32 - HILBERT_ORDER)).astype(np.uint32)
    gy = (yi >> (32 - HILBERT_ORDER)).astype(np.uint32)
    order = np.argsort(hilbert_d(gx, gy), kind="stable")
    xi, yi, node, tier = xi[order], yi[order], node[order], tier[order]

    # node index + tier packed into one uint16 (see the format note above)
    unknown = sorted({s for s in node if s not in vocab})
    for s in unknown:
        vocab[s] = len(vocab)
    ni = np.fromiter((vocab[s] for s in node), dtype=np.uint16, count=n)
    if ni.max(initial=0) >= (1 << 14):
        raise SystemExit("more than 16,383 nodes — widen the node field")
    ni |= (tier.astype(np.uint16) & 0x3) << 14

    nb = max(1, math.ceil(n / BUCKET))
    starts = [i * BUCKET for i in range(nb)]
    # §4.2c, half one: shuffle within each bucket.
    for st in starts:
        en = min(st + BUCKET, n)
        perm = rng.permutation(en - st) + st
        xi[st:en], yi[st:en], ni[st:en] = xi[perm], yi[perm], ni[perm]

    # §4.2c, half two: write the buckets themselves in random order, so no bucket
    # systematically paints over its neighbour.
    bucket_order = rng.permutation(nb)
    out_x = np.empty(n, dtype=np.uint32)
    out_y = np.empty(n, dtype=np.uint32)
    out_n = np.empty(n, dtype=np.uint16)
    buckets, at = [], 0
    for b in bucket_order:
        st = int(starts[b])
        en = min(st + BUCKET, n)
        cnt = en - st
        out_x[at:at + cnt] = xi[st:en]
        out_y[at:at + cnt] = yi[st:en]
        out_n[at:at + cnt] = ni[st:en]
        buckets.append([int(out_x[at:at + cnt].min()), int(out_y[at:at + cnt].min()),
                        int(out_x[at:at + cnt].max()), int(out_y[at:at + cnt].max()),
                        at, cnt])
        at += cnt

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{cc}{suffix}.bin"
    with open(path, "wb") as f:
        f.write(out_x.astype("<u4").tobytes())
        f.write(out_y.astype("<u4").tobytes())
        f.write(out_n.astype("<u2").tobytes())
    mb = path.stat().st_size / 1e6
    print(f"  {cc}{suffix:<4} {n:>9,} dots  {mb:>6.2f} MB  {nb:>4} buckets")
    return {"file": path.name, "dots": n, "buckets": buckets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="us")
    ap.add_argument("--coarse", action="store_true",
                    help="also pack the 1:10,000 edition (§4.1b), which is what the viewer "
                         "paints first while the fine buffer loads")
    args = ap.parse_args()

    # One vocabulary for every country and both editions, in taxonomy order where possible,
    # so a node index means the same thing in every buffer and the viewer's palette texture
    # can be built once from the taxonomy rather than per country.
    tree = json.loads((HERE / "taxonomy" / "religions.json").read_text(encoding="utf-8"))
    vocab = {node["id"]: i for i, node in enumerate(tree["nodes"])}
    n_taxonomy = len(vocab)

    rng = np.random.default_rng(SEED)
    editions = [("", 1000)] + ([("_10k", 10000)] if args.coarse else [])
    manifest = {"bucket": BUCKET, "seed": SEED, "editions": {}}
    for suffix, dv in editions:
        print(f"1:{dv:,}")
        entries = {}
        for cc in [c.strip() for c in args.countries.split(",")]:
            e = pack(cc, suffix, vocab, rng)
            if e is None:
                raise SystemExit(f"missing dots_{cc}{suffix}.geojson — run scatter.py first")
            entries[cc] = e
        manifest["editions"]["10k" if suffix else "fine"] = {
            "dot_value": dv, "countries": entries}

    if len(vocab) > n_taxonomy:
        # A node in the dots that the taxonomy does not list means religions.json is stale;
        # the viewer would show it as a bare id with no colour. Worth a loud line, not a crash.
        print(f"  WARNING: {len(vocab) - n_taxonomy} node(s) not in religions.json — "
              f"run taxonomy/build_tree.py")
    manifest["nodes"] = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    (OUT / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"wrote {OUT / 'manifest.json'}  ({len(manifest['nodes'])} nodes)")


if __name__ == "__main__":
    main()
