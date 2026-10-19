"""T-727 prototype: a routable rail graph from noritetsu's extract (data/proc/<region>).

    python build_graph.py jp

Reads noritetsu's ways.pkl, rels.pkl, coords.npz (read only) and writes graph_<region>.npz here:
chains of OSM nodes between junctions, with their kind, rank, flags and lon/lat.
Filter = build_tiles.py's: main and branch track, plus anything a passenger route runs over.
"""
import os
import sys
import time
import pickle
import gzip
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

NORI = Path(r"C:\Users\anita\projects\maps\noritetsu")
HERE = Path(__file__).resolve().parents[3] / "artifacts" / "trainworld" / "t727"  # outputs, not notes/

KIND = {"rail": 0, "light_rail": 1, "subway": 2, "tram": 3, "monorail": 4, "funicular": 5,
        "narrow_gauge": 6, "preserved": 7}
URBAN = {"light_rail", "subway", "tram", "monorail", "funicular"}


def rank_of(kind, tags):
    if tags.get("service"):
        return 3
    usage = tags.get("usage")
    if usage == "tourism" and kind == "narrow_gauge":
        return 1
    if usage in ("industrial", "military", "tourism", "freight", "test", "distribution"):
        return 2
    if usage == "main":
        return 0
    if usage == "branch":
        return 1
    return 0 if kind in URBAN else 1


def main(region):
    t0 = time.time()
    d = NORI / "data" / "proc" / region
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    print(f"loaded {len(ways)} ways, {cid.size} coords in {time.time()-t0:.0f}s", flush=True)
    on_route = set()
    for tags, members in rels.values():
        if tags.get("type") != "route":
            continue
        for ty, ref, role in members:
            if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                on_route.add(ref)
    del rels

    keep = []  # (wid, pos array, kind, rank, flags)
    stats = {"dropped_rank": 0, "dropped_kind": 0}
    for wid, (tags, nodes) in ways.items():
        rw = tags.get("railway")
        if rw not in KIND:
            stats["dropped_kind"] += 1
            continue
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        ok = cid[pos] == nodes
        if not ok.all():
            idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
            starts, ends = idx[0::2], idx[1::2]
            if starts.size == 0:
                continue
            b = np.argmax(ends - starts)
            pos = pos[starts[b]:ends[b]]
            if pos.size < 2:
                continue
        if ONLINE is not None and wid not in ONLINE:
            stats["dropped_kind"] += 1
            continue
        kind = "heritage" if rw == "preserved" else rw
        rank = rank_of(rw, tags)
        pax = wid in on_route
        if rank >= 2:
            if pax:
                rank = 1
            elif not ALL:
                stats["dropped_rank"] += 1
                continue
        flags = 0
        if pax:
            flags |= 1
        if tags.get("bridge") not in (None, "no"):
            flags |= 2
        if tags.get("tunnel") not in (None, "no"):
            flags |= 4
        if tags.get("highspeed") == "yes":
            flags |= 8
        keep.append((wid, pos.astype(np.int64), KIND[rw], rank, flags))
    del ways
    print(f"kept {len(keep)} ways; {stats}", flush=True)

    # junctions: a coord used more than once over kept ways, or a way end
    allpos = np.concatenate([k[1] for k in keep])
    uniq, counts = np.unique(allpos, return_counts=True)
    isj = np.zeros(cid.size, dtype=bool)
    isj[uniq[counts >= 2]] = True
    for k in keep:
        isj[k[1][0]] = True
        isj[k[1][-1]] = True

    ch_a, ch_b, ch_off, ch_kind, ch_rank, ch_flags, ch_wid = [], [], [0], [], [], [], []
    flat = []
    for wid, pos, kind, rank, flags in keep:
        cut = np.flatnonzero(isj[pos])
        for i in range(len(cut) - 1):
            s, e = cut[i], cut[i + 1]
            seg = pos[s:e + 1]
            flat.append(seg)
            ch_off.append(ch_off[-1] + seg.size)
            ch_a.append(seg[0])
            ch_b.append(seg[-1])
            ch_kind.append(kind)
            ch_rank.append(rank)
            ch_flags.append(flags)
            ch_wid.append(wid)
    flat = np.concatenate(flat)
    # renumber junction coords densely
    jpos = np.unique(np.concatenate([ch_a, ch_b]))
    a = np.searchsorted(jpos, ch_a)
    b = np.searchsorted(jpos, ch_b)
    lon = cx[flat] / 1e7
    lat = cy[flat] / 1e7
    np.savez_compressed(HERE / f"graph_{region}{'_all' if ALL else ''}{'_online' if ONLINE is not None else ''}.npz",a=a, b=b, off=np.array(ch_off),
                        kind=np.array(ch_kind, np.int8), rank=np.array(ch_rank, np.int8),
                        flags=np.array(ch_flags, np.int8), wid=np.array(ch_wid, np.int64),
                        lon=lon, lat=lat, jlon=cx[jpos] / 1e7, jlat=cy[jpos] / 1e7, osmnode=cid[flat])
    # degree of each junction (chain ends meeting there)
    deg = np.bincount(np.concatenate([a, b]), minlength=jpos.size)
    print(f"{len(ch_a)} chains, {jpos.size} junction nodes "
          f"(deg1 {np.sum(deg==1)}, deg2 {np.sum(deg==2)}, deg3+ {np.sum(deg>=3)}), "
          f"{flat.size} chain vertices; {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ALL = "--all" in sys.argv
    ONLINE = None
    if "--online" in sys.argv:
        import json
        with open(NORI / "dist" / "data" / sys.argv[1] / "ways.json", encoding="utf-8") as f:
            ONLINE = {int(w) for w in json.load(f)["ways"]}
    main(sys.argv[1])
