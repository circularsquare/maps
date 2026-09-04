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

The MVT bytes are written by mvt.py rather than mapbox_vector_tile, which is a hand-rolled
protobuf encoder and is checked by tools/check_tiles.py. Read the head of mvt.py for why:
the general encoder does 11,400 features/s on features that are all Points, and a --coarse
build has about 28 million of them.

Usage:
    python tiles.py                    # z0-10 from data/processed/*.geojson
    python tiles.py --max-zoom 8
    python tiles.py --countries us,ca,cz,br,au,ie,mx,nz,uk,pl,ro,ee,hr,in,de --coarse
"""
import argparse
import contextlib
import gzip
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pmtiles.writer import Writer
from pmtiles.tile import Compression, TileType, zxy_to_tileid

import mvt
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
# draw-order shuffle only; fixed so the archive is reproducible build to build.
#
# It genuinely is now, and was not before 2026-09-04: gzip stamps the current time into
# every member header unless told otherwise, so two builds of identical input differed
# everywhere. With mtime pinned in _encode_chunk the only bytes that still move are FIVE
# of them, in the gzip headers pmtiles' own writer puts on the root directory and the
# metadata (pmtiles/writer.py calls gzip.compress with no mtime and is not ours to fix).
# Everything that is a tile is now byte for byte the same, --jobs 1 or --jobs 8.
SHUFFLE_SEED = 20260827


def load(path: Path, cc: str):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    coords = np.array([ft["geometry"]["coordinates"] for ft in feats], dtype=float)
    props = [ft["properties"] for ft in feats]
    df = pd.DataFrame({
        "lon": coords[:, 0] if len(feats) else np.empty(0),
        "lat": coords[:, 1] if len(feats) else np.empty(0),
        "n": [p["n"] for p in props],
        "why": [p.get("why", "") for p in props],
        # spec §7; absent = measured
        "t": np.array([p.get("t", 0) for p in props], dtype=np.int64),
        "c": cc})
    # web mercator, normalised to [0,1]
    df["wx"] = (df["lon"] + 180.0) / 360.0
    s = np.sin(np.radians(df["lat"].clip(-85.05112878, 85.05112878)))
    df["wy"] = 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)
    return df


def merge_at_zoom(nc: np.ndarray, cc: np.ndarray, tier: np.ndarray,
                  wx: np.ndarray, wy: np.ndarray, z: int):
    """One mark per (merge cell, religion): mean position and how many dots merged.

    The five group keys are packed into one int64 and factorised on that, rather than
    handed to a five-column pandas groupby. It is the same grouping — cx and cy are
    bounded by 2^(z+CELL_BITS) and the codes by the vocabularies below, so the packing is
    injective and asserted to be — and it runs a few times faster on two million rows,
    which matters because this happens once per zoom per edition.

    `t` is part of the KEY, not an aggregate: a merged mark is one colour, and a cell
    holding both measured and derived dots of the same religion has to stay two marks or
    one of the two tiers would be drawn as the other (spec §7). It splits almost nothing —
    a source is normally one tier throughout a unit — and where it does split, it is
    splitting exactly the cells whose confidence genuinely differs.
    """
    n = 1 << (z + CELL_BITS)
    cx = np.minimum((wx * n).astype(np.int64), n - 1)
    cy = np.minimum((wy * n).astype(np.int64), n - 1)
    cell = cx * n + cy
    nspace, cspace, tspace = int(nc.max()) + 1, int(cc.max()) + 1, 4
    assert cell.max() * nspace * cspace * tspace < 2 ** 62, "merge key would overflow"
    key = ((cell * nspace + nc) * cspace + cc) * tspace + tier
    codes, uniq = pd.factorize(key)
    k = np.bincount(codes, minlength=len(uniq))
    mx = np.bincount(codes, weights=wx, minlength=len(uniq)) / k
    my = np.bincount(codes, weights=wy, minlength=len(uniq)) / k
    u = uniq.astype(np.int64)
    u, o_tier = np.divmod(u, tspace)
    u, o_cc = np.divmod(u, cspace)
    u, o_nc = np.divmod(u, nspace)
    return dict(tx=(u // n) >> CELL_BITS, ty=(u % n) >> CELL_BITS,
                wx=mx, wy=my, k=k.astype(np.int64), n=o_nc, c=o_cc, t=o_tier)


def tile_pixels(wx: np.ndarray, wy: np.ndarray, tx: np.ndarray, ty: np.ndarray, z: int):
    """Position within the tile, in MVT extent units."""
    nt = 1 << z
    px = np.clip((wx * nt - tx) * EXTENT, 0, EXTENT - 1).astype(np.int64)
    py = np.clip((wy * nt - ty) * EXTENT, 0, EXTENT - 1).astype(np.int64)
    return px, py


def bucket(tx: np.ndarray, ty: np.ndarray, z: int, rng):
    """Sort features into tiles, in a shuffled order within each tile.

    The shuffle is spec §4.2's, and it is not cosmetic: MVT features paint in file order,
    so the last one drawn at a pixel is the one you see. scatter.py emits dots grouped by
    node in sorted order within each polygon, so unshuffled, the alphabetically-last
    religion present in an area paints over every other one. Measured in central São
    Paulo: the last 5% of features emitted was 100% a single node, against a true
    composition of 58% Catholic — which is why the city read Spiritualist zoomed out and
    Catholic zoomed in, with 43x fewer Spiritualists.

    Shuffling the whole array ONCE and then sorting into tiles with a STABLE sort gives
    each tile a uniform permutation of its own features, which is what the old per-tile
    `random.shuffle` did, at a fraction of the cost. (The permutation itself differs from
    the pre-2026-09-04 one — a numpy Generator rather than random.Random — so the archive
    bytes changed once when this landed. It is still seeded and still reproducible.)
    """
    key = tx.astype(np.int64) * (1 << z) + ty.astype(np.int64)
    perm = rng.permutation(len(key))
    order = perm[np.argsort(key[perm], kind="stable")]
    return key[order], order


def _encode_chunk(tasks):
    """Encode and gzip a run of tiles. Module level so it survives Windows spawn.

    mtime=0 because gzip stamps the CURRENT TIME into every member header otherwise, and
    that alone made the archive differ byte for byte between two builds of identical
    input — the SHUFFLE_SEED above has always said the archive is reproducible and until
    2026-09-04 it quietly was not. It matters for telling a real change from no change.
    """
    return [(tid, gzip.compress(mvt.encode(payload, EXTENT), 6, mtime=0))
            for tid, payload in tasks]


def _chunk(tasks, jobs):
    """Split a zoom into enough pieces to balance out: tiles are wildly uneven in size,
    so one chunk per worker leaves whoever drew the dense ones finishing alone."""
    n = max(1, min(len(tasks), jobs * 8))
    size = -(-len(tasks) // n)
    return [tasks[i:i + size] for i in range(0, len(tasks), size)]


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
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="processes encoding tiles. Defaults to half the cores; --jobs 1 "
                         "runs it inline, which is what to do when a traceback out of a "
                         "worker is hiding where the problem is.")
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

    def _write_counts():
        """The legend's totals, per country.

        The viewer used to count features itself; with tiles it only ever holds the
        viewport, so the panel's totals have to be precomputed. Per country, because the
        legend is now per country: which nodes exist at all differs between sources far
        more than their sizes do, and a tree showing every node every source has ever had
        is a tree mostly greyed out.

        A function rather than a straight line of main() so that a build which cannot move
        the archive into place still writes the counts that match it.
        """
        per_country = {}
        for cc in sorted(set(dots["c"])):
            d = dots[dots["c"] == cc]
            r = rings[rings["c"] == cc] if rings is not None else None
            meta = COUNTRIES.get(cc, {})
            # The data bbox is where the dots actually are; `view` is where to fly, and
            # differs only where a country has distant outlying population — fitting the US
            # to its data bbox spans Hawaii to Maine and shows the reader an ocean.
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
            cd = cr = None
            if args.coarse:
                cd, cr, _dv = ed_data["_10k"]
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
            # The same two tallies for the all-countries view, which reads WORLD rather than
            # a country entry and would otherwise show fine numbers over a coarse map.
            cd, cr, _dv = ed_data["_10k"]
            counts["dots10k"] = cd["n"].value_counts().to_dict()
            counts["rings10k"] = (cr["n"].value_counts().to_dict() if cr is not None else {})
        with open(PROC / "counts.json", "w", encoding="utf-8") as f:
            json.dump(counts, f)
        print(f"wrote {PROC / 'counts.json'}")

    # ---- VOCABULARIES. An MVT layer refers to its property values by index into its own
    # value table, so the strings have to be interned. Do it once here, over every edition
    # at once, and the per-zoom and per-tile work below is integer arithmetic all the way
    # down to the bytes. mvt.encode() prunes each tile's table to the values that tile
    # actually uses — interning globally and shipping the whole table in all 28,000 tiles
    # would add about 100 MB to the archive.
    node_vocab = pd.Index(sorted({s for ed in ed_data.values() for f in ed[:2]
                                  if f is not None for s in f["n"].unique()}))
    cc_vocab = pd.Index(sorted({s for ed in ed_data.values() for f in ed[:2]
                                if f is not None for s in f["c"].unique()}))
    why_vocab = pd.Index(sorted({s for ed in ed_data.values() if ed[1] is not None
                                 for s in ed[1]["why"].unique()}))
    NODE_T = [mvt.value_str(s) for s in node_vocab]
    CC_T = [mvt.value_str(s) for s in cc_vocab]
    WHY_T = [mvt.value_str(s) for s in why_vocab]
    # `t` is written only when it is 1 or 2 — a missing one reads as measured, and a
    # property on every dot of every country would cost tile size for the common case.
    TIER_T = [mvt.value_int(1), mvt.value_int(2)]

    def tier_codes(t):
        return np.where(t > 0, t - 1, mvt.OMIT)

    # Intern each edition ONCE, not once per zoom. The atomic layer is the whole dot set
    # at all eleven zooms, so looking the node strings up per zoom is forty-odd million
    # hash lookups for an answer that cannot have changed.
    coded = {}
    for suffix, dv in editions:
        ed_dots, ed_rings, _ = ed_data[suffix]
        entry = {"n": node_vocab.get_indexer(ed_dots["n"]).astype(np.int64),
                 "c": cc_vocab.get_indexer(ed_dots["c"]).astype(np.int64),
                 "t": ed_dots["t"].to_numpy(dtype=np.int64),
                 "wx": ed_dots["wx"].to_numpy(), "wy": ed_dots["wy"].to_numpy()}
        if ed_rings is not None and len(ed_rings):
            entry["rings"] = {
                "n": node_vocab.get_indexer(ed_rings["n"]).astype(np.int64),
                "c": cc_vocab.get_indexer(ed_rings["c"]).astype(np.int64),
                "why": why_vocab.get_indexer(ed_rings["why"]).astype(np.int64),
                "wx": ed_rings["wx"].to_numpy(), "wy": ed_rings["wy"].to_numpy()}
        coded[suffix] = entry

    print(f"tiling z{args.min_zoom}-{args.max_zoom}…")
    rng = np.random.default_rng(SHUFFLE_SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_tiles = 0
    # Build beside the archive and move it into place at the end. A build that dies part
    # way through used to leave a truncated .pmtiles that the viewer loads and draws
    # wrong rather than failing; and `npx serve` holds the archive open while the map is
    # up, which makes truncating it in place an error and replacing it fine.
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    pool_ctx = (ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1
                else contextlib.nullcontext())
    with pool_ctx as pool, open(tmp, "wb") as f:
        w = Writer(f)
        # ZOOM IS THE OUTER LOOP, and that is the difference between a 600 MB build and a
        # 3.5 GB one. Every tile id at zoom z sorts below every tile id at z+1, so a zoom
        # can be encoded, written and dropped before the next one starts, instead of
        # holding the whole pyramid in memory until the end.
        for z in range(args.min_zoom, args.max_zoom + 1):
            layers, report = [], []
            for suffix, dv in editions:
                ed = coded[suffix]
                lp = "10k" if suffix else ""     # layer-name suffix: dots / dots10k
                nc, cc_, tr = ed["n"], ed["c"], ed["t"]
                wx, wy = ed["wx"], ed["wy"]

                m = merge_at_zoom(nc, cc_, tr, wx, wy, z)
                px, py = tile_pixels(m["wx"], m["wy"], m["tx"], m["ty"], z)
                kc, k_table = mvt.intern(m["k"], kind="int")
                key, o = bucket(m["tx"], m["ty"], z, rng)
                layers.append((f"dots{lp}", key, px[o], py[o], ["n", "c", "t", "k"],
                               np.stack([m["n"], m["c"], tier_codes(m["t"]), kc],
                                        axis=1)[o],
                               [NODE_T, CC_T, TIER_T, k_table]))
                report.append(f"1:{dv} {len(m['k']):>8,} marks "
                              f"(largest merges {int(m['k'].max()):,})")

                # The unmerged dots as their own layer, so the viewer can switch
                # consolidation off and get the plain scatter. Every dot at every zoom, so
                # this is the expensive half of the archive — see the size report at the end.
                if not args.no_atomic:
                    nt = 1 << z
                    tx = np.minimum((wx * nt).astype(np.int64), nt - 1)
                    ty = np.minimum((wy * nt).astype(np.int64), nt - 1)
                    px, py = tile_pixels(wx, wy, tx, ty, z)
                    key, o = bucket(tx, ty, z, rng)
                    layers.append((f"atomic{lp}", key, px[o], py[o], ["n", "c", "t"],
                                   np.stack([nc, cc_, tier_codes(tr)], axis=1)[o],
                                   [NODE_T, CC_T, TIER_T]))

                if "rings" in ed:
                    r = ed["rings"]
                    nt = 1 << z
                    tx = np.minimum((r["wx"] * nt).astype(np.int64), nt - 1)
                    ty = np.minimum((r["wy"] * nt).astype(np.int64), nt - 1)
                    px, py = tile_pixels(r["wx"], r["wy"], tx, ty, z)
                    key, o = bucket(tx, ty, z, rng)
                    layers.append((f"rings{lp}", key, px[o], py[o], ["n", "c", "why"],
                                   np.stack([r["n"], r["c"], r["why"]], axis=1)[o],
                                   [NODE_T, CC_T, WHY_T]))

            # Every tile touched by any layer at this zoom, in tile-id order — which is
            # Hilbert order, so the archive comes out clustered.
            touched = np.unique(np.concatenate([L[1] for L in layers]))
            nt = 1 << z
            tids = np.fromiter((zxy_to_tileid(z, int(k // nt), int(k % nt))
                                for k in touched), dtype=np.int64, count=len(touched))
            tasks = []
            for i in np.argsort(tids):
                key = touched[i]
                payload = []
                for name, lkey, lpx, lpy, lkeys, lcodes, ltables in layers:
                    a = np.searchsorted(lkey, key, "left")
                    b = np.searchsorted(lkey, key, "right")
                    if b > a:
                        payload.append((name, lpx[a:b], lpy[a:b], lkeys,
                                        lcodes[a:b], ltables))
                tasks.append((int(tids[i]), payload))

            # Encoding and gzipping a tile needs nothing but that tile, so the zoom is
            # handed out in contiguous chunks. Contiguous rather than striped because
            # `tasks` is already in tile-id order and map() returns chunk results in
            # order, so the archive stays clustered with no re-sort. Nothing is shared:
            # each chunk carries copies of its own slices.
            for blobs in (map(_encode_chunk, _chunk(tasks, args.jobs))
                          if pool is None else
                          pool.map(_encode_chunk, _chunk(tasks, args.jobs))):
                for tid, blob in blobs:
                    w.write_tile(tid, blob)
            n_tiles += len(touched)
            print(f"  z{z:<2} {len(touched):>6,} tiles   " + "  |  ".join(report))

        print("writing pmtiles…")
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
    mb = tmp.stat().st_size / 1e6
    # THE DEV SERVER HOLDS THE ARCHIVE OPEN. `npx serve` keeps a handle on every file it
    # has served, and Windows refuses both truncation and rename against it, so a retile
    # with the map still up used to die on an OSError with no tiles written at all. The
    # build is finished by this point either way — say which file to move and stop, rather
    # than throw away sixty seconds of work over a file lock.
    try:
        os.replace(tmp, OUT)
    except OSError as e:
        _write_counts()
        raise SystemExit(
            f"\nbuilt {tmp.name} ({mb:.1f} MB, {n_tiles:,} tiles) but could not put it in "
            f"place:\n  {e}\n\nSomething has {OUT.name} open — normally the `npx serve` "
            f"running the local map.\nStop it and run:  mv {tmp} {OUT}\n"
            f"counts.json is already written and matches the new archive.")
    print(f"wrote {OUT}  ({mb:.1f} MB, {n_tiles:,} tiles)")

    _write_counts()


if __name__ == "__main__":
    main()
