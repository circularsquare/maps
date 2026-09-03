"""
Build a PMTiles archive from the scattered dots, merging rather than dropping.

This is spec.md §4.2 made real, and it is why tippecanoe is not used. Tippecanoe's job at
low zoom is to DROP features until a tile fits its byte budget; whichever dots survive is
close to arbitrary, so a small group blinks in and out as you zoom or pan. Here every dot
survives at every zoom — nearby dots of the same religion are MERGED into one mark carrying
`k`, the number merged, and the viewer draws area proportional to k. Nobody disappears, and
within any one view a mark's area is still strictly proportional to people (§4.1).

Merge grid: each tile is divided into 2^CELL_BITS squares per axis. All dots of one religion
inside one square become one mark, placed at their mean position — not the square's centre,
so the marks follow the real point cloud instead of snapping to a lattice.

Rings (§4.3) ride along as their own layer, unmerged: there is already only one per body per
county, and a ring carries no magnitude to merge.

Usage:
    python tiles.py                    # z0-10 from data/processed/*.geojson
    python tiles.py --max-zoom 8
"""
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import mapbox_vector_tile
from pmtiles.writer import Writer
from pmtiles.tile import Compression, TileType, zxy_to_tileid

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
OUT = PROC / "religiondots.pmtiles"

CELL_BITS = 5        # 32x32 merge cells per tile -> ~16px cells on a 512px tile
EXTENT = 4096


def load(path: Path):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    lon, lat, node, extra = [], [], [], []
    for ft in gj["features"]:
        c = ft["geometry"]["coordinates"]
        lon.append(c[0]); lat.append(c[1])
        node.append(ft["properties"]["n"])
        extra.append(ft["properties"].get("why", ""))
    df = pd.DataFrame({"lon": lon, "lat": lat, "n": node, "why": extra})
    # web mercator, normalised to [0,1]
    df["wx"] = (df["lon"] + 180.0) / 360.0
    s = np.sin(np.radians(df["lat"].clip(-85.05112878, 85.05112878)))
    df["wy"] = 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)
    return df


def merge_at_zoom(df: pd.DataFrame, z: int) -> pd.DataFrame:
    """One row per (merge cell, religion): mean position and how many dots merged."""
    n = 1 << (z + CELL_BITS)
    cx = np.minimum((df["wx"].to_numpy() * n).astype(np.int64), n - 1)
    cy = np.minimum((df["wy"].to_numpy() * n).astype(np.int64), n - 1)
    g = pd.DataFrame({"cx": cx, "cy": cy, "n": df["n"].to_numpy(),
                      "wx": df["wx"].to_numpy(), "wy": df["wy"].to_numpy()})
    out = g.groupby(["cx", "cy", "n"], sort=False).agg(
        wx=("wx", "mean"), wy=("wy", "mean"), k=("wx", "size")).reset_index()
    out["tx"] = out["cx"].to_numpy() >> CELL_BITS
    out["ty"] = out["cy"].to_numpy() >> CELL_BITS
    return out


def to_tiles(rows: pd.DataFrame, z: int, layer: str, sink: dict, with_k: bool):
    """Bucket merged rows into tiles and stash MVT-ready features."""
    ntiles = 1 << z
    px = (rows["wx"].to_numpy() * ntiles - rows["tx"].to_numpy()) * EXTENT
    py = (rows["wy"].to_numpy() * ntiles - rows["ty"].to_numpy()) * EXTENT
    px = np.clip(px, 0, EXTENT - 1).astype(int)
    py = np.clip(py, 0, EXTENT - 1).astype(int)

    tx = rows["tx"].to_numpy(); ty = rows["ty"].to_numpy()
    node = rows["n"].to_numpy()
    k = rows["k"].to_numpy() if with_k else np.ones(len(rows), dtype=int)
    why = rows["why"].to_numpy() if "why" in rows else None

    for i in range(len(rows)):
        props = {"n": node[i]}
        if with_k:
            props["k"] = int(k[i])
        elif why is not None and why[i]:
            props["why"] = why[i]
        sink[(z, int(tx[i]), int(ty[i]))][layer].append({
            "geometry": {"type": "Point", "coordinates": [int(px[i]), int(py[i])]},
            "properties": props,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-zoom", type=int, default=0)
    ap.add_argument("--max-zoom", type=int, default=10)
    ap.add_argument("--no-atomic", action="store_true",
                    help="skip the unmerged layer (smaller archive, no consolidation toggle)")
    ap.add_argument("--countries", default="us",
                    help="comma-separated, e.g. us,ca — merged into one archive so the merge "
                         "(§4.2) works across borders rather than stopping at them")
    args = ap.parse_args()

    print("reading dots…")
    dot_frames, ring_frames = [], []
    for cc in args.countries.split(","):
        cc = cc.strip()
        dp, rp = PROC / f"dots_{cc}.geojson", PROC / f"rings_{cc}.geojson"
        if not dp.exists():
            raise SystemExit(f"missing {dp.name} — run: python scatter.py --country {cc}")
        d = load(dp)
        dot_frames.append(d)
        r = load(rp) if rp.exists() else None
        if r is not None:
            ring_frames.append(r)
        print(f"  {cc}: {len(d):,} dots, {len(r) if r is not None else 0:,} rings")
    dots = pd.concat(dot_frames, ignore_index=True)
    rings = pd.concat(ring_frames, ignore_index=True) if ring_frames else None
    print(f"  total {len(dots):,} dots"
          + (f", {len(rings):,} rings" if rings is not None else ""))

    sink = defaultdict(lambda: defaultdict(list))
    for z in range(args.min_zoom, args.max_zoom + 1):
        merged = merge_at_zoom(dots, z)
        to_tiles(merged, z, "dots", sink, with_k=True)
        biggest = int(merged["k"].max())
        print(f"  z{z:<2} {len(merged):>9,} marks  (largest merges {biggest:,} dots)")

        # The unmerged dots as their own layer, so the viewer can switch consolidation off
        # and get the plain scatter. Every dot at every zoom, so this is the expensive half
        # of the archive — see the size report at the end.
        if not args.no_atomic:
            a = dots.copy()
            nt = 1 << z
            a["tx"] = np.minimum((a["wx"] * nt).astype(np.int64), nt - 1)
            a["ty"] = np.minimum((a["wy"] * nt).astype(np.int64), nt - 1)
            a["k"] = 1
            to_tiles(a, z, "atomic", sink, with_k=False)

        if rings is not None:
            r = rings.copy()
            nt = 1 << z
            r["tx"] = np.minimum((r["wx"] * nt).astype(np.int64), nt - 1)
            r["ty"] = np.minimum((r["wy"] * nt).astype(np.int64), nt - 1)
            r["k"] = 1
            to_tiles(r, z, "rings", sink, with_k=False)

    print(f"encoding {len(sink):,} tiles…")
    encoded = {}
    for (z, x, y), layers in sink.items():
        payload = [{"name": name, "features": feats} for name, feats in layers.items()]
        buf = mapbox_vector_tile.encode(
            payload, default_options={"extents": EXTENT, "y_coord_down": True})
        encoded[zxy_to_tileid(z, x, y)] = gzip.compress(buf, 6)

    print("writing pmtiles…")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        w = Writer(f)
        for tid in sorted(encoded):
            w.write_tile(tid, encoded[tid])
        w.finalize(
            {
                "tile_type": TileType.MVT,
                "tile_compression": Compression.GZIP,
                "min_zoom": args.min_zoom,
                "max_zoom": args.max_zoom,
                "min_lon_e7": int(dots["lon"].min() * 1e7),
                "min_lat_e7": int(dots["lat"].min() * 1e7),
                "max_lon_e7": int(dots["lon"].max() * 1e7),
                "max_lat_e7": int(dots["lat"].max() * 1e7),
                "center_zoom": 4,
                "center_lon_e7": int(dots["lon"].mean() * 1e7),
                "center_lat_e7": int(dots["lat"].mean() * 1e7),
            },
            {
                "name": "religiondots",
                "vector_layers": [
                    {"id": "dots", "fields": {"n": "String", "k": "Number"}},
                    {"id": "atomic", "fields": {"n": "String"}},
                    {"id": "rings", "fields": {"n": "String", "why": "String"}},
                ],
            },
        )
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({mb:.1f} MB, {len(encoded):,} tiles)")

    # The viewer used to count features itself; with tiles it only ever holds the viewport,
    # so the panel's totals have to be precomputed.
    counts = {"dot_value": None,
              "dots": dots["n"].value_counts().to_dict(),
              "rings": rings["n"].value_counts().to_dict() if rings is not None else {}}
    with open(PROC / "counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f)
    print(f"wrote {PROC / 'counts.json'}")


if __name__ == "__main__":
    main()
