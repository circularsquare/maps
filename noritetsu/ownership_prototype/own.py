"""PROTOTYPE: each piece of track belongs to exactly one line.

    python own.py <cc> [--bbox lon0,lat0,lon1,lat1] [--tag name]

Reads dist/data/<cc>/ and data/proc/<cc>/ (never writes there). Writes to ./out/<tag>/.

1. Every drawn way gets ONE owner:
   - a register line, by geometry (register_way_lines' test: >= 60% of the way inside a 40 m
     buffer of the line, kind family must agree), the way's own name settles it, then a
     high-speed flag that AGREES is preferred (but a mismatch no longer excludes), then the
     nearest line; exact ties (within 0.5 m) by natural sort of ref, then name, then id;
   - else the register line an OSM twin of it was merged into, if that twin's route used it;
   - else one OSM line (not a named train) whose route relation uses the way: lowest ref by
     natural sort, empty refs last, then name, then id;
   - else nobody (only named trains run there: "svc"; nothing runs there: "none").
2. Every section of every line gets a FOOTPRINT: the owner sections it runs over, as ranges
   [lo, hi] along them. A register section's footprint is itself. An OSM section's comes from
   its own vertices: each segment is looked up to the exact OSM way it lies on (the geometry
   is cut from those ways, coordinates rounded alike), the way says which LINE owns it, and the
   segment is projected onto the nearest section of that line to say which section and where.
3. Owned track of an OSM owner section = the union of the footprints of its own line's sections
   onto it. Register sections own themselves whole.
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True

import argparse
import json
import math
import pickle
import re
import time
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Point

from build_model import norm_line_name, group_lines, kind_family  # scratch copy
from build_tiles import KIND, rank_of                               # scratch copy

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")

WAY_BUFFER_M = 40.0
WAY_MIN_FRAC = 0.6
WAY_TIE_M = 8.0
EXACT_TIE_M = 0.5
SAME_RAILS_M = 8.0
# Urban kinds that are often the same rails under different tags: Lausanne's m1 is
# railway=light_rail on the ground and "subway" in the Swiss register. They may own each
# other's ways only on the same rails (SAME_RAILS_M) and never for a guided line.
TRAMLIKE = {"tram", "light_rail", "subway"}
DENSIFY_M = 50.0
SNAP_M = 3.0          # an OSM section segment with no exact way: nearest drawn way this close
OWNER_SEC_M = 160.0    # a segment projects onto its owner line's nearest section this close


def segment_keys(xi, yi):
    """A 64-bit hash per segment of a polyline given as integer 1e-5 degree coordinates,
    the same whichever way round the segment is walked."""
    x1, y1, x2, y2 = xi[:-1], yi[:-1], xi[1:], yi[1:]
    swap = (x1 > x2) | ((x1 == x2) & (y1 > y2))
    a = np.where(swap, x2, x1).astype(np.uint64)
    b = np.where(swap, y2, y1).astype(np.uint64)
    c = np.where(swap, x1, x2).astype(np.uint64)
    d = np.where(swap, y1, y2).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = a * np.uint64(0x9E3779B97F4A7C15)
        h = (h ^ (h >> np.uint64(29))) + b * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(31))) + c * np.uint64(0x94D049BB133111EB)
        h = (h ^ (h >> np.uint64(27))) + d * np.uint64(0xD6E8FEB86659FD93)
        h ^= h >> np.uint64(32)
    return h


def natkey(s):
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower()) for p in re.findall(r"\d+|\D+", s or ""))


def owner_key(l):
    ref = (l.get("ref") or "").strip()
    return (ref == "", natkey(ref), l.get("name") or "", l["id"])


def merge(iv, tol):
    """Union of [lo, hi] intervals, closing gaps up to tol."""
    if not iv:
        return []
    iv = sorted([list(x) for x in iv])
    out = [iv[0]]
    for lo, hi in iv[1:]:
        if lo <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def app_merge(iv, km):
    """The app's span merge: gaps under 150 m closed (capped at 10% of the section), ends
    snapped within the same tolerance."""
    if not iv:
        return []
    tol = min(0.15 / km, 0.1) if km > 0 else 0
    out = merge(iv, tol)
    if out[0][0] <= tol:
        out[0][0] = 0.0
    if out[-1][1] >= 1 - tol:
        out[-1][1] = 1.0
    return out


def span_len(iv):
    return min(1.0, sum(hi - lo for lo, hi in iv))


def intersect(a, b):
    out, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


class Proj:
    def __init__(self, lat0, lon0):
        self.kx = 111320.0 * math.cos(math.radians(lat0))
        self.ky = 110574.0
        self.lon0, self.lat0 = lon0, lat0

    def __call__(self, lon, lat):
        return np.column_stack([(np.asarray(lon) - self.lon0) * self.kx,
                                (np.asarray(lat) - self.lat0) * self.ky])


def load_country(cc):
    d = ROOT / "dist" / "data" / cc
    L = json.load(open(d / "lines.json", encoding="utf-8"))["lines"]
    S = json.load(open(d / "stations.json", encoding="utf-8"))["stations"]
    A = json.load(open(d / "aliases.json", encoding="utf-8"))
    G = {}
    for l in L:
        p = d / "geom" / f"{l['id']}.json"
        G[l["id"]] = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    return L, S, A, G


def build(cc, bbox=None, log=print):
    t0 = time.time()
    L, S, A, G = load_country(cc)
    byid = {l["id"]: l for l in L}
    closed = set()
    for l in L:
        shut = set(l.get("closed") or [])
        for a, b, km, gid in l["sections"]:
            if f"{a}|{b}" in shut:
                closed.add(gid)
    # projection centre
    lats = [s["y"] for s in S.values()]
    lons = [s["x"] for s in S.values()]
    if bbox:
        lat0, lon0 = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    else:
        lat0, lon0 = float(np.median(lats)), float(np.median(lons))
    P = Proj(lat0, lon0)

    def in_box(ll, pad=0.0):
        if not bbox:
            return True
        x0, y0 = ll[:, 0].min(), ll[:, 1].min()
        x1, y1 = ll[:, 0].max(), ll[:, 1].max()
        return not (x1 < bbox[0] - pad or x0 > bbox[2] + pad or y1 < bbox[1] - pad or y0 > bbox[3] + pad)

    # ---------------------------------------------------------------- sections
    secs = {}
    for l in L:
        reg = l.get("src", "osm") != "osm"
        g = G.get(l["id"], {})
        for a, b, km, gid in l["sections"]:
            pts = g.get(f"{a}|{b}")
            if not pts or len(pts) < 2:
                continue
            ll = np.asarray(pts, dtype=np.float64)
            if not in_box(ll, 0.02):
                continue
            xy = P(ll[:, 0], ll[:, 1])
            ctr = ll.mean(axis=0)
            inside = (not bbox) or (bbox[0] <= ctr[0] <= bbox[2] and bbox[1] <= ctr[1] <= bbox[3])
            secs[gid] = {"line": l["id"], "a": a, "b": b, "km": km, "ll": ll,
                         "geom": LineString(xy), "reg": reg, "service": bool(l["service"]),
                         "closed": gid in closed, "inside": inside}
    G.clear()
    log(f"[{time.time()-t0:5.1f}s] {len(secs)} sections with geometry in scope")

    # ---------------------------------------------------------------- drawn ways
    d = ROOT / "data" / "proc" / cc
    ways = pickle.load(open(d / "ways.pkl", "rb"))
    rels = pickle.load(open(d / "rels.pkl", "rb"))
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    on_route = {ref for tags, members in rels.values() if tags.get("type") == "route"
                for ty, ref, role in members
                if ty == "w" and (not role or role.startswith(("forward", "backward")))}
    W = []          # list of dicts
    seg_keys, seg_way = [], []     # rounded segment hash -> way index, as sorted arrays
    for wid, (tags, nodes) in ways.items():
        kind = KIND.get(tags.get("railway"))
        if kind is None:
            continue
        if rank_of(kind, tags) >= 2 and wid not in on_route:
            continue
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        ok = cid[pos] == nodes
        if ok.sum() < 2:
            continue
        pos = pos[ok]
        lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
        ll = np.column_stack([lon, lat])
        if not in_box(ll, 0.03):
            continue
        g = LineString(P(lon, lat))
        if g.length <= 0:
            continue
        j = len(W)
        xi = np.array([int(round(round(float(v), 5) * 1e5)) for v in lon], dtype=np.int64)
        yi = np.array([int(round(round(float(v), 5) * 1e5)) for v in lat], dtype=np.int64)
        seg_keys.append(segment_keys(xi, yi))
        seg_way.append(np.full(len(xi) - 1, j, dtype=np.int64))
        W.append({"wid": wid, "kind": kind, "geom": g, "len": g.length, "ll": ll,
                  "hs": tags.get("highspeed") == "yes", "nodes": nodes[ok],
                  "names": {norm_line_name(p, tags.get("operator", ""))
                            for p in re.split(r"[・;/]", tags.get("name") or "") if p.strip()},
                  "tags": tags})
    wtree = STRtree([w["geom"] for w in W])
    wmid = [w["geom"].interpolate(0.5, normalized=True) for w in W]
    widx = {w["wid"]: j for j, w in enumerate(W)}
    sk = np.concatenate(seg_keys) if seg_keys else np.zeros(0, np.uint64)
    sw = np.concatenate(seg_way) if seg_way else np.zeros(0, np.int64)
    order = np.argsort(sk, kind="stable")
    sk, sw = sk[order], sw[order]
    del seg_keys, seg_way
    log(f"[{time.time()-t0:5.1f}s] {len(W)} drawn ways, {len(sk)} segments indexed")

    # ---------------------------------------------------------------- register owners
    cand = defaultdict(dict)
    for l in L:
        if l.get("src", "osm") == "osm":
            continue
        fam = kind_family(l["kind"])
        hs_line = l.get("highspeed")
        hs_sec = l.get("highspeed_sections") or {}
        key = norm_line_name(l["name"], l.get("operator", ""))
        near, dist, flag = defaultdict(float), {}, {}
        for a, b, km, gid in l["sections"]:
            s = secs.get(gid)
            if s is None or s["closed"]:
                continue
            g = s["geom"]
            buf = g.buffer(WAY_BUFFER_M, quad_segs=4)
            for j in wtree.query(buf):
                w = W[j]
                near[j] += w["geom"].intersection(buf).length / w["len"]
                dj = g.distance(wmid[j])
                if dj < dist.get(j, math.inf):
                    dist[j] = dj
                    f = hs_sec.get(f"{a}|{b}")
                    flag[j] = hs_line if f is None else f
        for j, f in near.items():
            if f < WAY_MIN_FRAC:
                continue
            wf = kind_family(W[j]["kind"])
            cross = False
            if wf != fam:
                if not ({wf, fam} <= TRAMLIKE and not l.get("guided") and dist[j] <= SAME_RAILS_M):
                    continue
                cross = True
            hsm = None if flag[j] is None else (bool(flag[j]) == W[j]["hs"])
            cand[j][l["id"]] = (dist[j], key in W[j]["names"], hsm, cross)

    wown, wstat, wnote = {}, {}, {}
    reg_contest = []      # (j, owner, [others], exact)
    n_hs_override = 0
    line_secs = defaultdict(list)
    for g, s in secs.items():
        if s["reg"] and not s["closed"]:
            line_secs[s["line"]].append(s["geom"])

    def mean_dist(j, lid):
        """Mean distance from points every 10 m along way j to line lid's track."""
        w = W[j]["geom"]
        n = max(2, min(60, int(w.length / 10)))
        pts = shapely.points(np.asarray([w.interpolate(i / (n - 1), normalized=True).coords[0]
                                         for i in range(n)]))
        env = w.buffer(60).envelope
        gs = [g for g in line_secs[lid] if g.intersects(env)] or line_secs[lid]
        D = shapely.distance(pts[:, None], np.array(gs, dtype=object)[None, :])
        return float(np.nanmin(D, axis=1).mean())
    n_tie_geom = n_tie_rule = 0
    for j, cd in cand.items():
        pool = {k: v for k, v in cd.items() if not v[3]} or cd
        named = {k: v for k, v in pool.items() if v[1]}
        if named:
            pool = named
        match = {k: v for k, v in pool.items() if v[2] is not False}
        if match:
            pool = match
        else:
            n_hs_override += 1
        dmin = min(v[0] for v in pool.values())
        tie = [k for k, v in pool.items() if v[0] <= dmin + WAY_TIE_M]
        exact = [k for k, v in pool.items() if v[0] <= dmin + EXACT_TIE_M]
        if len(exact) > 1:
            # Nearest at the way's midpoint is a poor test where two lines cross or touch
            # there: the whole way's mean distance decides, and only a true tie (two lines
            # drawn over the same rails) falls to the ref rule.
            md = {k: mean_dist(j, k) for k in exact}
            mmin = min(md.values())
            exact2 = [k for k in exact if md[k] <= mmin + EXACT_TIE_M]
            if len(exact2) < len(exact):
                n_tie_geom += 1
            else:
                n_tie_rule += 1
            exact = exact2
        o = min(exact, key=lambda k: owner_key(byid[k]))
        wown[j], wstat[j] = o, "reg"
        if len(tie) > 1:
            reg_contest.append((j, o, sorted(set(tie) - {o}), len(exact) > 1, sorted(set(exact) - {o})))
    log(f"[{time.time()-t0:5.1f}s] {len(wown)} ways owned by a register line by geometry; "
        f"{len(reg_contest)} had another register line within {WAY_TIE_M} m; "
        f"{n_hs_override} took a register line whose high-speed flag disagrees (no agreeing line near); "
        f"midpoint ties: {n_tie_geom} settled by mean distance, {n_tie_rule} by the ref rule")

    # ---------------------------------------------------------------- route ways per line
    groups, routes = group_lines(rels, lambda *_: None)
    lalias = A.get("lines", {})
    line_ways = defaultdict(set)
    for lid, mtags, rids in groups:
        tgt = lalias.get(lid, lid)
        for rid in rids:
            for ty, ref, role in routes[rid][1]:
                if ty == "w" and (not role or role.startswith(("forward", "backward"))):
                    line_ways[tgt].add(ref)
    way_users = defaultdict(set)
    for lid, ws in line_ways.items():
        if lid not in byid:
            continue
        for wid in ws:
            j = widx.get(wid)
            if j is not None:
                way_users[j].add(lid)
    # ---------------------------------------------------------------- the country's own track only
    reg_json = json.load(open(ROOT / "dist" / "regions.json", encoding="utf-8"))["regions"].get(cc)
    n_abroad = 0
    if reg_json:
        from shapely.geometry import Polygon
        parts = json.loads(reg_json["parts"]) if isinstance(reg_json["parts"], str) else reg_json["parts"]
        polys = []
        for ring in parts:
            r = ring[0] if ring and isinstance(ring[0][0], list) else ring
            arr = np.asarray(r, dtype=np.float64)
            if len(arr) >= 3:
                polys.append(Polygon(P(arr[:, 0], arr[:, 1])).buffer(0))
        outline = shapely.union_all(polys).buffer(150)
        shapely.prepare(outline)
        for j, w in enumerate(W):
            if j in wown:
                continue
            if not outline.contains(wmid[j]):
                wstat[j] = "abroad"
                n_abroad += 1
    log(f"[{time.time()-t0:5.1f}s] {n_abroad} drawn ways outside the country (no register owner): left to the neighbour")

    # ---------------------------------------------------------------- pass 2: station throats
    # A way no register line passed the 40 m / 60% test for, used only by operating patterns
    # (OSM lines whose route ways are otherwise mostly register track), lying mostly within
    # THROAT_M of a register line of its kind: platform roads and crossovers at big stations.
    THROAT_M, THROAT_SHARE = 150.0, 0.6
    reg_share = {}
    for lid, ws in line_ways.items():
        if lid not in byid or byid[lid].get("src", "osm") != "osm":
            continue
        tot = regk = 0.0
        for wid in ws:
            j = widx.get(wid)
            if j is None:
                continue
            tot += W[j]["len"]
            regk += W[j]["len"] if j in wown else 0.0
        reg_share[lid] = regk / tot if tot else 0.0
    reg_secs = [g for g, s in secs.items() if s["reg"] and not s["closed"]]
    rtree = STRtree([secs[g]["geom"] for g in reg_secs])
    n_throat, km_throat = 0, 0.0
    throat = {}
    for j in range(len(W)):
        if j in wown or wstat.get(j) == "abroad":
            continue
        users = [k for k in way_users.get(j, ()) if not byid[k]["service"]]
        if not users or any(byid[k].get("src", "osm") != "osm" for k in users):
            continue
        if min(reg_share.get(k, 0.0) for k in users) < 0.6:
            continue
        w = W[j]
        buf = w["geom"].buffer(THROAT_M, quad_segs=2)
        best = None
        for i in rtree.query(buf):
            g = reg_secs[i]
            l = byid[secs[g]["line"]]
            if kind_family(l["kind"]) != kind_family(w["kind"]):
                continue
            share = w["geom"].intersection(secs[g]["geom"].buffer(THROAT_M, quad_segs=2)).length / w["len"]
            dd = secs[g]["geom"].distance(wmid[j])
            if share >= THROAT_SHARE and (best is None or dd < best[0]):
                best = (dd, secs[g]["line"])
        if best:
            throat[j] = best[1]
    for j, lid in throat.items():
        wown[j], wstat[j] = lid, "reg-throat"
        n_throat += 1
        km_throat += W[j]["len"] / 1000
    log(f"[{time.time()-t0:5.1f}s] pass 2: {n_throat} ways ({km_throat:.1f} km) used only by operating "
        f"patterns, within {THROAT_M:.0f} m of a register line, given to it")

    n_twin = 0
    osm_contest = []      # (j, owner, [others])
    for j in range(len(W)):
        if j in wown or wstat.get(j) == "abroad":
            continue
        users = way_users.get(j, set())
        regu = [k for k in users if byid[k].get("src", "osm") != "osm"]
        if regu:            # a twin merged into a register line ran here
            wown[j] = min(regu, key=lambda k: owner_key(byid[k]))
            wstat[j] = "reg-twin"
            n_twin += 1
            continue
        osmu = [k for k in users if not byid[k]["service"]]
        if osmu:
            o = min(osmu, key=lambda k: owner_key(byid[k]))
            wown[j], wstat[j] = o, "osm"
            if len(osmu) > 1:
                osm_contest.append((j, o, sorted(set(osmu) - {o})))
            continue
        wstat[j] = "svc" if users else "none"
    log(f"[{time.time()-t0:5.1f}s] {n_twin} ways given to a register line through its merged OSM twin; "
        f"{sum(1 for v in wstat.values() if v == 'osm')} owned by an OSM line, "
        f"{len(osm_contest)} of them shared by several OSM lines (the fixed rule decides)")

    # ---------------------------------------------------------------- owner sections index
    own_secs = [g for g, s in secs.items() if not s["closed"] and (s["reg"] or not s["service"])]
    osec_geom = [secs[g]["geom"] for g in own_secs]
    osec_tree = STRtree(osec_geom)
    osec_line = np.array([secs[g]["line"] for g in own_secs], dtype=object)

    # ---------------------------------------------------------------- footprints
    foot = {}
    comp = {}       # gid -> Counter of km by status
    STATUS = ["unmatched", "reg", "reg-throat", "reg-twin", "osm", "svc", "none", "abroad"]
    st_code = np.array([STATUS.index(wstat.get(j, "none")) for j in range(len(W))] + [0], dtype=np.int64)
    lines_idx = {lid: i for i, lid in enumerate(byid)}
    lid_of = list(byid)
    own_code = np.array([lines_idx[wown[j]] if j in wown else -1 for j in range(len(W))] + [-1],
                        dtype=np.int64)
    for gid, s in secs.items():
        if s["reg"]:
            foot[gid] = [[gid, 0.0, 1.0]] if not s["closed"] else []
            comp[gid] = Counter({"reg": s["geom"].length})
            continue
        ll = s["ll"]
        xi = np.rint(ll[:, 0] * 1e5).astype(np.int64)
        yi = np.rint(ll[:, 1] * 1e5).astype(np.int64)
        keys = segment_keys(xi, yi)
        pos = np.searchsorted(sk, keys)
        np.clip(pos, 0, max(len(sk) - 1, 0), out=pos)
        wj = np.where((len(sk) > 0) & (sk[pos] == keys), sw[pos], -1) if len(sk) else np.full(len(keys), -1)
        xy = np.asarray(s["geom"].coords)
        n = len(xy) - 1
        miss = np.flatnonzero(wj < 0)
        if miss.size:
            mids = shapely.points((xy[miss] + xy[miss + 1]) / 2)
            got = wtree.query_nearest(mids, max_distance=SNAP_M, all_matches=False)
            for pi, wi in zip(got[0], got[1]):
                wj[miss[pi]] = wi
        seglen = np.hypot(*(xy[1:] - xy[:-1]).T)
        stc = np.where(wj < 0, 0, st_code[wj])
        oc = np.where(wj < 0, -1, own_code[wj])
        cnt = Counter({STATUS[k]: float(v) for k, v in
                       enumerate(np.bincount(stc, weights=seglen, minlength=len(STATUS))) if v > 0})
        comp[gid] = cnt
        keep = seglen > 0
        if not keep.any():
            foot[gid] = []
            continue
        # densify to DENSIFY_M pieces
        kk = np.maximum(1, np.ceil(seglen / DENSIFY_M).astype(np.int64)) * keep
        idx = np.repeat(np.arange(n), kk)
        start = np.repeat(np.cumsum(kk) - kk, kk)
        t0_ = (np.arange(idx.size) - start) / np.repeat(np.maximum(kk, 1), kk)
        t1_ = t0_ + 1.0 / np.repeat(np.maximum(kk, 1), kk)
        d_ = xy[idx + 1] - xy[idx]
        A_ = xy[idx] + d_ * t0_[:, None]
        B_ = xy[idx] + d_ * t1_[:, None]
        O_ = oc[idx]
        ivs = defaultdict(list)
        for oi in np.unique(O_[O_ >= 0]):
            o = lid_of[oi]
            I = np.flatnonzero(O_ == oi)
            pa, pb = A_[I], B_[I]
            mids = shapely.points((pa + pb) / 2)
            env = shapely.box(min(pa[:, 0].min(), pb[:, 0].min()) - OWNER_SEC_M,
                              min(pa[:, 1].min(), pb[:, 1].min()) - OWNER_SEC_M,
                              max(pa[:, 0].max(), pb[:, 0].max()) + OWNER_SEC_M,
                              max(pa[:, 1].max(), pb[:, 1].max()) + OWNER_SEC_M)
            ci = osec_tree.query(env)
            ci = ci[osec_line[ci] == o]
            if ci.size == 0:
                cnt["far"] += sum(float(np.hypot(*(b - a))) for a, b in zip(pa, pb))
                continue
            geoms = np.array([osec_geom[k] for k in ci], dtype=object)
            D = shapely.distance(mids[:, None], geoms[None, :])
            ch = D.argmin(axis=1)
            dm = D[np.arange(len(ch)), ch]
            for k in np.unique(ch):
                sel = np.flatnonzero((ch == k) & (dm <= OWNER_SEC_M))
                if not sel.size:
                    continue
                g = geoms[k]
                ta = shapely.line_locate_point(g, shapely.points(pa[sel]), normalized=True)
                tb = shapely.line_locate_point(g, shapely.points(pb[sel]), normalized=True)
                tgt = own_secs[ci[k]]
                ivs[tgt].extend(zip(np.minimum(ta, tb).tolist(), np.maximum(ta, tb).tolist()))
            far = np.flatnonzero(dm > OWNER_SEC_M)
            if far.size:
                cnt["far"] += float(np.hypot(*(pb[far] - pa[far]).T).sum())
        out = []
        for tgt, iv in ivs.items():
            tl = secs[tgt]["geom"].length
            for lo, hi in merge(iv, 20.0 / tl if tl > 0 else 0):
                if hi - lo > 1e-6:
                    out.append([tgt, round(float(lo), 4), round(float(hi), 4)])
        foot[gid] = out
    log(f"[{time.time()-t0:5.1f}s] footprints for {len(foot)} sections")

    # ---------------------------------------------------------------- owned spans
    own = {}
    for gid, s in secs.items():
        if s["closed"]:
            continue
        if s["reg"]:
            own[gid] = [[0.0, 1.0]]
    tmp = defaultdict(list)
    for gid, s in secs.items():
        if s["reg"] or s["service"]:
            continue
        for tgt, lo, hi in foot[gid]:
            if secs[tgt]["line"] == s["line"]:
                tmp[tgt].append((lo, hi))
    for tgt, iv in tmp.items():
        own[tgt] = app_merge(iv, secs[tgt]["km"])

    meta = {"cc": cc, "bbox": bbox, "lat0": lat0, "lon0": lon0}
    model = {"meta": meta,
             "secs": {g: {k: v for k, v in s.items() if k not in ("geom", "ll")} for g, s in secs.items()},
             "foot": foot, "own": own, "comp": {g: dict(c) for g, c in comp.items()},
             "ways": [{"wid": w["wid"], "kind": w["kind"], "len": w["len"],
                       "lon": float(w["ll"][:, 0].mean()), "lat": float(w["ll"][:, 1].mean()),
                       "n0": int(w["nodes"][0]), "n1": int(w["nodes"][-1]),
                       "nodes": [int(x) for x in w["nodes"]],
                       "xy": np.asarray(w["geom"].coords).round(1).tolist(),
                       "name": w["tags"].get("name", ""), "hs": w["hs"]} for w in W],
             "wown": wown, "wstat": wstat, "reg_contest": reg_contest, "osm_contest": osm_contest,
             "cand": {j: v for j, v in cand.items()},
             "way_users": {j: sorted(v) for j, v in way_users.items()},
             "n_hs_override": n_hs_override, "n_twin": n_twin}
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cc")
    ap.add_argument("--bbox", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    bbox = [float(x) for x in a.bbox.split(",")] if a.bbox else None
    tag = a.tag or a.cc
    out = HERE / "out" / tag
    out.mkdir(parents=True, exist_ok=True)
    m = build(a.cc, bbox)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(m, f)
    # the shipping shape: footprints only for sections that are not owners of themselves alone
    ship = {str(g): f for g, f in m["foot"].items() if not (len(f) == 1 and f[0][0] == g and f[0][1] == 0 and f[0][2] == 1)}
    with open(out / "footprints.json", "w", encoding="utf-8") as f:
        json.dump({"region": a.cc, "foot": ship}, f, separators=(",", ":"))
    print("wrote", out)


if __name__ == "__main__":
    main()
