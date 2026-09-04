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

Every feature carries `c`, its country, and the merge groups by it — so a cell straddling a
border draws one mark per country rather than one mark of ambiguous nationality. That reverses
the original reason for putting several countries in one archive (§4.2: "so the merge works
across a border"). It has to: the viewer now shows and colours one country at a time, and a
mark merged across the 49th parallel would carry a count that belongs to neither side of it.
The archive stays shared because the pyramid and the tile boundaries are, not the marks.

Usage:
    python tiles.py                    # z0-10 from data/processed/*.geojson
    python tiles.py --max-zoom 8
"""
import argparse
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import mapbox_vector_tile
from pmtiles.writer import Writer
from pmtiles.tile import Compression, TileType, zxy_to_tileid

from countries import COUNTRIES

# Windows consoles here are cp1252 and the data is not: source names, categories and country
# notes carry Č, š, ú, ł and much else. Without this the script dies inside a print() after the
# real work has succeeded, which reads like a pipeline failure and is not one.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
PROC = HERE / "data" / "processed"
OUT = PROC / "religiondots.pmtiles"

CELL_BITS = 5        # 32x32 merge cells per tile -> ~16px cells on a 512px tile
EXTENT = 4096
# draw-order shuffle only; fixed so the archive is reproducible build to build
SHUFFLE_SEED = 20260827


def load(path: Path, cc: str):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    lon, lat, node, extra, tier = [], [], [], [], []
    for ft in gj["features"]:
        c = ft["geometry"]["coordinates"]
        lon.append(c[0]); lat.append(c[1])
        node.append(ft["properties"]["n"])
        extra.append(ft["properties"].get("why", ""))
        tier.append(int(ft["properties"].get("t", 0)))      # spec §7; absent = measured
    df = pd.DataFrame({"lon": lon, "lat": lat, "n": node, "why": extra, "t": tier, "c": cc})
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
    # `t` is part of the KEY, not an aggregate: a merged mark is one colour, and a cell
    # holding both measured and derived dots of the same religion has to stay two marks or
    # one of the two tiers would be drawn as the other (spec §7). It splits almost nothing —
    # a source is normally one tier throughout a unit — and where it does split, it is
    # splitting exactly the cells whose confidence genuinely differs.
    g = pd.DataFrame({"cx": cx, "cy": cy, "n": df["n"].to_numpy(), "c": df["c"].to_numpy(),
                      "t": df["t"].to_numpy(),
                      "wx": df["wx"].to_numpy(), "wy": df["wy"].to_numpy()})
    out = g.groupby(["cx", "cy", "n", "c", "t"], sort=False).agg(
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
    cc = rows["c"].to_numpy()
    k = rows["k"].to_numpy() if with_k else np.ones(len(rows), dtype=int)
    why = rows["why"].to_numpy() if "why" in rows else None
    tier = rows["t"].to_numpy() if "t" in rows else None

    for i in range(len(rows)):
        props = {"n": node[i], "c": cc[i]}
        if tier is not None and tier[i]:        # omitted when measured, as in the geojson
            props["t"] = int(tier[i])
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
    ap.add_argument("--coarse", action="store_true",
                    help="also pack the 1:10,000 edition, as layers dots10k / atomic10k / "
                         "rings10k, so the viewer can offer people-per-dot as a setting. "
                         "Needs `scatter.py --country <cc> --dot-value 10000` to have been "
                         "run for every country in --countries.")
    ap.add_argument("--refresh-meta", action="store_true",
                    help="rewrite only the per-country display fields in counts.json from "
                         "countries.py and exit — the tiles and the tallies are untouched. "
                         "For editing a name, source, basis, note or view box, which "
                         "otherwise costs a full retile to change one sentence.")
    ap.add_argument("--countries", default="us",
                    help="comma-separated, e.g. us,ca — one archive, but marks never merge "
                         "across a border: every feature is tagged with its country and the "
                         "viewer draws one country at a time")
    args = ap.parse_args()

    if args.refresh_meta:
        path = PROC / "counts.json"
        if not path.exists():
            raise SystemExit(f"no {path.name} to refresh — run a full build first")
        counts = json.loads(path.read_text(encoding="utf-8"))
        for cc, entry in counts.get("countries", {}).items():
            meta = COUNTRIES.get(cc)
            if not meta:
                print(f"  {cc}: not in countries.py — left as it was")
                continue
            entry["name"] = meta.get("name", cc.upper())
            entry["name_in"] = meta.get("name_in") or entry["name"]
            entry["source"] = meta.get("source", "")
            entry["basis"] = meta.get("basis", "")
            entry["note"] = meta.get("note_public", "")
            # the data bbox stays whatever the build measured; only the framing is editable
            entry["view"] = list(meta.get("view") or entry.get("bbox") or [])
            print(f"  {cc}: {entry['name']}  view {[round(v, 1) for v in entry['view']]}")
        path.write_text(json.dumps(counts), encoding="utf-8")
        print(f"refreshed {path}")
        return

    # EDITIONS — the same countries scattered at two dot values, in ONE archive (§4.1b).
    #
    # They cannot share a layer: a 1:10,000 dot is a different feature from a 1:1,000 dot,
    # not a filtered subset of one, because scatter.py re-runs the whole spatial carry at
    # the new value rather than dropping nine dots in ten. So each edition gets its own
    # `dots`/`atomic`/`rings` triple and the viewer swaps visibility, exactly as the
    # consolidation toggle already does (§4.2b).
    #
    # Subsampling the fine edition instead would have been cheaper and wrong twice: the
    # counts would only hold in expectation, breaking §4.1's "count the dots"; and a group
    # with three dots nationally would appear or vanish on the seed, where a real 1:10,000
    # run drops it deterministically and §4.3 gives it a ring instead.
    editions = [("", 1000)]
    if args.coarse:
        editions.append(("_10k", 10000))

    print("reading dots…")
    ed_data = {}
    for suffix, dv in editions:
        dot_frames, ring_frames = [], []
        for cc in args.countries.split(","):
            cc = cc.strip()
            dp = PROC / f"dots_{cc}{suffix}.geojson"
            rp = PROC / f"rings_{cc}{suffix}.geojson"
            if not dp.exists():
                raise SystemExit(
                    f"missing {dp.name} — run: python scatter.py --country {cc}"
                    + (f" --dot-value {dv}" if suffix else ""))
            d = load(dp, cc)
            dot_frames.append(d)
            r = load(rp, cc) if rp.exists() else None
            if r is not None:
                ring_frames.append(r)
            print(f"  1:{dv:<6} {cc}: {len(d):,} dots, "
                  f"{len(r) if r is not None else 0:,} rings")
        ed_dots = pd.concat(dot_frames, ignore_index=True)
        ed_rings = pd.concat(ring_frames, ignore_index=True) if ring_frames else None
        print(f"  1:{dv} total {len(ed_dots):,} dots"
              + (f", {len(ed_rings):,} rings" if ed_rings is not None else ""))
        ed_data[suffix] = (ed_dots, ed_rings, dv)

    # The fine edition remains "the" dots for the archive header, the bbox and the legend
    # tallies, so nothing downstream of counts.json changes shape when --coarse is off.
    dots, rings, _ = ed_data[""]

    sink = defaultdict(lambda: defaultdict(list))
    for suffix, dv in editions:
        ed_dots, ed_rings, _ = ed_data[suffix]
        lp = "10k" if suffix else ""            # layer-name suffix: dots / dots10k
        if suffix:
            print(f"1:{dv} edition -> layers dots{lp} / atomic{lp} / rings{lp}")
        for z in range(args.min_zoom, args.max_zoom + 1):
            merged = merge_at_zoom(ed_dots, z)
            to_tiles(merged, z, f"dots{lp}", sink, with_k=True)
            biggest = int(merged["k"].max())
            print(f"  z{z:<2} {len(merged):>9,} marks  (largest merges {biggest:,} dots)")

            # The unmerged dots as their own layer, so the viewer can switch consolidation
            # off and get the plain scatter. Every dot at every zoom, so this is the
            # expensive half of the archive — see the size report at the end.
            if not args.no_atomic:
                a = ed_dots.copy()
                nt = 1 << z
                a["tx"] = np.minimum((a["wx"] * nt).astype(np.int64), nt - 1)
                a["ty"] = np.minimum((a["wy"] * nt).astype(np.int64), nt - 1)
                a["k"] = 1
                to_tiles(a, z, f"atomic{lp}", sink, with_k=False)

            if ed_rings is not None:
                r = ed_rings.copy()
                nt = 1 << z
                r["tx"] = np.minimum((r["wx"] * nt).astype(np.int64), nt - 1)
                r["ty"] = np.minimum((r["wy"] * nt).astype(np.int64), nt - 1)
                r["k"] = 1
                to_tiles(r, z, f"rings{lp}", sink, with_k=False)

    print(f"encoding {len(sink):,} tiles…")
    encoded = {}
    # Shuffle each tile's features before encoding, because MVT features paint in file order and
    # the last one drawn at a pixel is the one you see. Unshuffled, scatter.py emits dots grouped
    # by node in sorted order within each polygon, so the alphabetically-last religion present in
    # an area paints over every other one. Measured in central São Paulo: the last 5% of features
    # emitted was 100% a single node, against a true composition of 58% Catholic — which is why
    # the city read Spiritualist zoomed out and Catholic zoomed in, with 43x fewer Spiritualists.
    #
    # A uniform shuffle makes the visible dot at any pixel a uniform draw from the dots covering
    # it, so the zoomed-out picture is a representative sample of the zoomed-in one. Seeded, so
    # the archive stays reproducible.
    #
    # It costs archive size: MVT delta-encodes consecutive points and shuffling maximises the
    # deltas. The size line at the end of the run is the number to watch.
    shuffler = random.Random(SHUFFLE_SEED)
    for (z, x, y), layers in sink.items():
        for feats in layers.values():
            shuffler.shuffle(feats)
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
                    {"id": f"dots{lp}",
                     "fields": {"n": "String", "c": "String", "k": "Number",
                                "t": "Number"}}
                    for lp in (["", "10k"] if args.coarse else [""])
                ] + [
                    {"id": f"atomic{lp}", "fields": {"n": "String", "c": "String",
                                                     "t": "Number"}}
                    for lp in (["", "10k"] if args.coarse else [""])
                ] + [
                    {"id": f"rings{lp}",
                     "fields": {"n": "String", "c": "String", "why": "String"}}
                    for lp in (["", "10k"] if args.coarse else [""])
                ],
            },
        )
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({mb:.1f} MB, {len(encoded):,} tiles)")

    # The viewer used to count features itself; with tiles it only ever holds the viewport,
    # so the panel's totals have to be precomputed. Per country, because the legend is now
    # per country: which nodes exist at all differs between sources far more than their sizes
    # do, and a tree showing every node every source has ever had is a tree mostly greyed out.
    per_country = {}
    for cc in sorted(set(dots["c"])):
        d = dots[dots["c"] == cc]
        r = rings[rings["c"] == cc] if rings is not None else None
        meta = COUNTRIES.get(cc, {})
        # The data bbox is where the dots actually are; `view` is where to fly, and differs
        # only where a country has distant outlying population — fitting the US to its data
        # bbox spans Hawaii to Maine and shows the reader an ocean.
        box = [float(d["lon"].min()), float(d["lat"].min()),
               float(d["lon"].max()), float(d["lat"].max())]
        per_country[cc] = {
            "name": meta.get("name", cc.upper()),
            "name_in": meta.get("name_in") or meta.get("name", cc.upper()),
            "source": meta.get("source", ""),
            "basis": meta.get("basis", ""),
            "note": meta.get("note_public", ""),
            "bbox": box,
            "view": list(meta.get("view") or box),
            "dots": d["n"].value_counts().to_dict(),
            "rings": r["n"].value_counts().to_dict() if r is not None else {},
        }
        # The legend counts marks, and at 1:10,000 a country has different ones: two of
        # India's seventeen nodes stop drawing a dot at all and become rings instead. So
        # the coarse edition needs its own tallies or the legend would report the fine
        # edition's numbers against the coarse edition's picture.
        if args.coarse:
            cd, cr, _ = ed_data["_10k"]
            cd = cd[cd["c"] == cc]
            cr = cr[cr["c"] == cc] if cr is not None else None
            per_country[cc]["dots10k"] = cd["n"].value_counts().to_dict()
            per_country[cc]["rings10k"] = (cr["n"].value_counts().to_dict()
                                           if cr is not None else {})
        print(f"  {cc}: {len(d):,} dots over {d['n'].nunique()} nodes, "
              f"bbox {[round(v, 1) for v in box]}"
              + (f"  |  1:10k {len(cd):,} dots over {cd['n'].nunique()} nodes, "
                 f"{0 if cr is None else len(cr)} rings" if args.coarse else ""))

    counts = {"dot_value": 1000,
              "dot_values": [1000, 10000] if args.coarse else [1000],
              "countries": per_country,
              "dots": dots["n"].value_counts().to_dict(),
              "rings": rings["n"].value_counts().to_dict() if rings is not None else {}}
    if args.coarse:
        # The same two tallies for the all-countries view, which reads WORLD rather than a
        # country entry and would otherwise show fine-edition numbers over a coarse map.
        cd, cr, _ = ed_data["_10k"]
        counts["dots10k"] = cd["n"].value_counts().to_dict()
        counts["rings10k"] = (cr["n"].value_counts().to_dict() if cr is not None else {})
    with open(PROC / "counts.json", "w", encoding="utf-8") as f:
        json.dump(counts, f)
    print(f"wrote {PROC / 'counts.json'}")


if __name__ == "__main__":
    main()
