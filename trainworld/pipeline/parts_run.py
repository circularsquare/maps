"""T-2213: the land parts (routes.py's `part` and `parts`) of finished packs, extract by extract.

    python pipeline/parts_run.py [<id> ... | --from <file of ids> | --all] [--workers N] [--downloads N]
                                 [--dry] [--redo]
    python pipeline/parts_run.py --status [<id> ... | --all]

A pack's extracts are its `routes.extracts` (SUBSTITUTE swaps one too big for its pack). Download
threads fetch the extracts ahead of the readers (a copy kept in data/raw/osm, .ways2.npz or
.osm.pbf, is used instead; downloads go to data/raw/osm_parts/, are deleted once read, and wait
while the disk has under MIN_FREE). Per extract, in a reader process: its ways, and for every pack
it serves, the pack's water pixels at 50 m under the ways that join land ->
data/work/parts/px/<id>/<extract>.npy. Then per pack whose extracts are all done: its parts
(routes.land_parts); unless --dry, the pack backed up to data/work/pre-T2213/, the parts written
(routes.write_parts) and its zone file's part fields (market_zones.parts_only); a summary in
data/work/parts/<id>.json either way (--dry: says whether the merged pack would be merged again).
Restartable: an extract whose pixels every pack has, and a pack with a summary (written, unless
--dry), are skipped; --redo does the packs again from the pixels.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import multiprocessing as mp
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routes  # noqa: E402

ROOT = routes.ROOT
PACKS = routes.PACKS
OUT = os.path.join(ROOT, "data", "work", "parts")
DL = os.path.join(ROOT, "data", "raw", "osm_parts")
TILE = 0.5  # degrees: segments are found by the tile of their first end
BIG_PBF = 700e6  # bytes; at most two extracts this big are read at once (memory)
MIN_FREE = 15e9  # bytes free on the disk before a download starts
BACKUP = os.path.join(ROOT, "data", "work", "pre-T2213")
# an extract named for a pack whose land in it is all in a smaller one (qie's corner of Vietnam)
SUBSTITUTE = {"asia/sea": "asia/vietnam"}


def slug(e):
    return e.replace("/", "_")


def px_path(cid, e):
    return os.path.join(OUT, "px", cid, slug(e) + ".npy")


def summary_path(cid):
    return os.path.join(OUT, f"{cid}.json")


def plan(ids):
    """{id: extracts} for the named packs (or every pack with routes), skipping merged-pack ids
    with no res-9 copy."""
    import glob
    if ids is None:
        ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PACKS, "*.json"))
                     if "." not in os.path.basename(p)[:-5] and "intercity" not in p)
    out = {}
    for c in ids:
        with open(os.path.join(PACKS, f"{c}.json"), encoding="utf-8") as f:
            h = json.load(f)
        ex = (h.get("routes") or {}).get("extracts")
        if "cells" in h and "water" in h and ex:
            out[c] = sorted({SUBSTITUTE.get(e, e) for e in ex})
    return out


# ---------------------------------------------------------------- workers

_big = None


def _init(sem):
    global _big
    _big = sem
    routes.OSM = DL  # downloads go to their own folder


def cached(e):
    """A copy of the extract kept in data/raw/osm (its ways, or the .osm.pbf), or None."""
    stem = os.path.join(ROOT, "data", "raw", "osm", slug(e))
    for p in (stem + ".ways2.npz", stem + ".osm.pbf"):
        if os.path.exists(p):
            return p
    return None


def fetch(e, tries=3):
    """The extract on disk: a kept copy, or downloaded into DL (waiting while the disk is short)."""
    import shutil
    p = cached(e)
    if p:
        return p
    for k in range(tries):
        while shutil.disk_usage(ROOT).free < MIN_FREE:
            time.sleep(30)
        try:
            return routes.download(e)
        except Exception:
            part = routes.pbf_path(e) + ".part"
            if os.path.exists(part):
                os.remove(part)
            if k == tries - 1:
                raise
            time.sleep(60)


def load_ways(e, path):
    """The extract's ways from `path` (fetch's); a download is deleted once read."""
    if path.endswith(".npz"):
        return dict(np.load(path)), "kept ways"
    size = os.path.getsize(path)
    big = size > BIG_PBF
    if big:
        _big.acquire()
    try:
        routes.OSM = os.path.dirname(path)
        w = routes.read_ways(e, keep_pbf=os.path.dirname(path) != DL, cache=False)
    finally:
        routes.OSM = DL
        if big:
            _big.release()
    return w, f"{size / 1e6:.0f} MB"


def extract_job(e, ids, path):
    """The water pixels under the extract's land-joining ways, for each of `ids`."""
    import water
    from read_pack import load_pack
    t0 = time.time()
    try:
        ways, how = load_ways(e, path)
        off, lon, lat, _ = routes.merged([ways], routes.PART_FLAGS)
        del ways
        way = np.repeat(np.arange(len(off) - 1), np.diff(off))
        s = np.nonzero(way[1:] == way[:-1])[0]
        del way
        alon, alat, blon, blat = lon[s], lat[s], lon[s + 1], lat[s + 1]
        del lon, lat, s
        tx, ty = np.floor(alon / TILE).astype(np.int64), np.floor(alat / TILE).astype(np.int64)
        key = tx * 100_000 + ty
        order = np.argsort(key, kind="stable")
        key = key[order]
        alon, alat, blon, blat = alon[order], alat[order], blon[order], blat[order]
        t_read = time.time() - t0
        n_px = 0
        for cid in ids:
            h, a = load_pack(cid, PACKS)
            bar, origin = routes.water50(h, a)
            H, W = bar.shape
            lon0, lat0 = h["origin"]["lon"], h["origin"]["lat"]
            pad = routes.PART_PAD_M
            lo, la = water.to_lonlat(np.array([origin[0] - pad, origin[0] + W * routes.WALK_CELL_M + pad]),
                                     np.array([origin[1] - pad, origin[1] + H * routes.WALK_CELL_M + pad]), lon0, lat0)
            sel = []
            for gx in range(int(np.floor(lo[0] / TILE)) - 1, int(np.floor(lo[1] / TILE)) + 2):
                i0 = np.searchsorted(key, gx * 100_000 + int(np.floor(la[0] / TILE)) - 1)
                i1 = np.searchsorted(key, gx * 100_000 + int(np.floor(la[1] / TILE)) + 2)
                if i1 > i0:
                    sel.append(np.arange(i0, i1))
            sel = np.concatenate(sel) if sel else np.zeros(0, np.int64)
            xa, ya = routes.local_xy(alon[sel], alat[sel], lon0, lat0)
            xb, yb = routes.local_xy(blon[sel], blat[sel], lon0, lat0)
            px = routes.road_water_pixels(bar, origin, xa, ya, xb, yb)
            os.makedirs(os.path.dirname(px_path(cid, e)), exist_ok=True)
            np.save(px_path(cid, e) + ".part.npy", px)
            os.replace(px_path(cid, e) + ".part.npy", px_path(cid, e))
            n_px += len(px)
        return e, True, f"{e}: {how}, {len(key):,} segments read in {t_read:.0f} s; {len(ids)} packs, {n_px:,} water pixels under roads ({time.time() - t0:.0f} s)"
    except Exception:
        return e, False, f"{e}: FAILED\n{traceback.format_exc()}"


def backup(cid):
    """The pack's files (and a merged pack's res-9 copy and stamp) into data/work/pre-T2213/, once."""
    import shutil
    import merge_cells as mc
    for src, sub in ((PACKS, "packs"), (mc.WORK, "res9")):
        for ext in (".json", ".bin", ".merged.json", ".market.json"):
            p = os.path.join(src, cid + ext)
            q = os.path.join(BACKUP, sub, cid + ext)
            if os.path.exists(p) and not os.path.exists(q):
                os.makedirs(os.path.dirname(q), exist_ok=True)
                shutil.copyfile(p, q + ".part")
                os.replace(q + ".part", q)


def pack_job(cid, exs, dry):
    """The pack's parts from its extracts' pixels; unless dry, written (after a backup), and its zone
    file's part fields; its summary."""
    t0 = time.time()
    try:
        h, a = routes.res9_pack(cid)
        bar, origin = routes.water50(h, a)
        opened = np.unique(np.concatenate([np.load(px_path(cid, e)) for e in exs]))
        lab, edge, info = routes.land_labels(bar, opened)
        part, block, _ = routes.land_parts(h, a, lab, edge, origin, info, exs)
        market = None
        if dry:
            how = routes.write_parts(cid, part, block, dry=True)
        else:
            backup(cid)
            how = routes.write_parts(cid, part, block)
            if os.path.exists(os.path.join(PACKS, f"{cid}.market.json")):
                import contextlib
                import io
                import market_zones
                with contextlib.redirect_stdout(io.StringIO()):
                    market = "parts (its sites off the merged cells)" if market_zones.parts_only([cid]) else "parts"
        keys = ("people", "jobs", "cells", "open", "landmass", "lon", "lat")
        top = [dict({k: block[k][i] for k in keys}, part=i) for i in range(min(block["parts"], 60))]
        s = {"id": cid, "city": h.get("city"), "country": h.get("country"), "written": not dry, "how": how, "market": market,
             "parts": block["parts"], "people": float(sum(block["people"])),
             "off_first": float(sum(block["people"][1:])), "interior": float(sum(p for p, o in zip(block["people"], block["open"]) if not o)),
             "pixels_opened": info["pixels_opened"], "islets_closed": info["islets_closed"], "top": top,
             "seconds": round(time.time() - t0, 1)}
        with open(summary_path(cid) + ".part", "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False)
        os.replace(summary_path(cid) + ".part", summary_path(cid))
        return cid, True, (f"{cid}: {block['parts']} parts, {s['off_first']:,.0f} of {s['people']:,.0f} people off the first; "
                           f"{how}{'; zone file ' + market if market else ''} ({s['seconds']:.0f} s)")
    except BaseException:  # write_parts' sys.exit too
        return cid, False, f"{cid}: FAILED\n{traceback.format_exc()}"


# ---------------------------------------------------------------- driver

def done_pack(cid, dry):
    p = summary_path(cid)
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as f:
        return dry or json.load(f).get("written", False)


def main(args):
    workers = int(args.pop(args.index("--workers") + 1)) if "--workers" in args else 3
    downloads = int(args.pop(args.index("--downloads") + 1)) if "--downloads" in args else 4
    if "--from" in args:  # a file of ids, one per line
        with open(args.pop(args.index("--from") + 1), encoding="utf-8") as f:
            args += f.read().split()
    ids = None if "--all" in args else [x for x in args if not x.startswith("--")]
    dry, redo = "--dry" in args, "--redo" in args
    want = plan(ids or None)
    if "--status" in args:
        nex = {e for ex in want.values() for e in ex}
        have = {e for e in nex if all(os.path.exists(px_path(c, e)) for c, ex in want.items() if e in ex)}
        print(f"{len(want)} packs, {sum(done_pack(c, False) for c in want)} written; {len(have)} of {len(nex)} extracts read")
        return
    os.makedirs(OUT, exist_ok=True)
    todo = {c: ex for c, ex in want.items() if redo or not done_pack(c, dry)}
    by_ex = {}
    for c, ex in todo.items():
        for e in ex:
            if not os.path.exists(px_path(c, e)):
                by_ex.setdefault(e, []).append(c)
    print(f"parts: {len(todo)} of {len(want)} packs to do, {len(by_ex)} extracts to read, {workers} readers, "
          f"{downloads} downloads at once" + (" (dry)" if dry else ""), flush=True)
    t0 = time.time()
    failed_ex = set()
    mgr = mp.Manager()
    sem = mgr.BoundedSemaphore(2)
    routes.OSM = DL
    os.makedirs(DL, exist_ok=True)
    # downloads run ahead of the readers, at most `downloads + workers` extracts on disk unread
    ahead = threading.BoundedSemaphore(downloads + workers)
    lock = threading.Lock()
    count = [0]

    def get(e):
        ahead.acquire()
        try:
            return e, fetch(e), None
        except Exception as x:
            return e, None, f"{type(x).__name__}: {x}"

    def done(res):
        e, ok, line = res
        ahead.release()
        with lock:
            count[0] += 1
            if not ok:
                failed_ex.add(e)
            print(f"[{count[0]}/{len(by_ex)} {time.time() - t0:.0f}s] {line}", flush=True)

    # the extracts serving most packs first: their packs can be finished soonest
    order = sorted(by_ex, key=lambda e: (-len(by_ex[e]), e))
    with ThreadPoolExecutor(downloads) as tp, mp.Pool(workers, initializer=_init, initargs=(sem,), maxtasksperchild=1) as pool:
        for f in as_completed([tp.submit(get, e) for e in order]):
            e, path, err = f.result()
            if err:
                done((e, False, f"{e}: download FAILED ({err})"))
                continue
            pool.apply_async(extract_job, (e, by_ex[e], path), callback=done,
                             error_callback=lambda x, e=e: done((e, False, f"{e}: FAILED ({x})")))
        pool.close()
        pool.join()
    ready = [(c, ex, dry) for c, ex in todo.items() if all(os.path.exists(px_path(c, e)) for e in ex)]
    failed = []
    with mp.Pool(workers, initializer=_init, initargs=(sem,)) as pool:
        for k, (c, ok, line) in enumerate(pool.imap_unordered(_pack_star, ready)):
            if not ok:
                failed.append(c)
            print(f"[{k + 1}/{len(ready)} {time.time() - t0:.0f}s] {line}", flush=True)
    print(f"parts: done in {time.time() - t0:.0f} s; {len(ready) - len(failed)} packs {'summarised' if dry else 'written'}, "
          f"{len(failed)} failed {' '.join(failed)}; {len(failed_ex)} extracts failed {' '.join(sorted(failed_ex))}; "
          f"{len(todo) - len(ready)} packs waiting on a failed extract", flush=True)


def _pack_star(t):
    return pack_job(*t)


if __name__ == "__main__":
    main(sys.argv[1:])
