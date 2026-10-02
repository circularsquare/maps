"""ONE OWNER FOR EVERY PIECE OF TRACK, and what every section of every line runs over.

Anita, 2026-10-01: no double counting. Every piece of track belongs to exactly one line, for
crediting only: what a ride completes, a line's percentage, the country and operator totals.
Rides are still entered on any service over any path; this only decides what they credit.

    foot, report = ownership.run(region, lines, geoms, route_users, stations, log)

WHO OWNS A DRAWN WAY (the ways build_tiles draws), in this order:

  1. A register line, by geometry: register_way_lines' test (at least WAY_MIN_FRAC of the way
     within WAY_BUFFER_M of the line, kind family agreeing). The way's own name naming a
     candidate settles it; then a candidate whose high-speed flag agrees with the way's
     highspeed tag is preferred, though a disagreeing one is no longer excluded when it is the
     only line there; then the nearest at the way's midpoint; candidates within EXACT_TIE_M of
     each other are measured again by mean distance along the whole way, and only a true tie
     (two register lines drawn over the same rails) falls to the fixed rule below. Tram, light
     rail and subway may own each other's ways on the same rails (SAME_RAILS_M), never for a
     guided line: Lausanne's m1 is light_rail on the ground and "subway" in the register.
  2. Station throats: a way no register line passed for, used only by operating patterns
     (OSM lines whose route ways are mostly register track), lying mostly within THROAT_M of a
     register line of its kind, goes to the nearest such line.
  3. The register line an OSM twin was merged into, where that twin's relation used the way.
  4. One OSM line, not a named train, whose route relation uses the way, by the fixed rule.
  5. Nobody: only named trains run there, nothing does, or the way lies abroad (outside the
     country's outline in dist/regions.json): the neighbour's.

THE FIXED RULE, for shared track no register decides: the lowest line ref by natural sort
("2" < "10" < "A3"), empty refs last, then the name, then the line id.

SINGLE-TRACK COMPANIONS. A register can list the two tracks or tubes of one route as two
lines: the Ceneri base tunnel is CBT Est and CBT Ovest, and stretches of the Gotthard line
have their left track as a km-line of its own ("Brunnen - Sisikon (Gleis links)"). An OSM
section follows one track, so each of those could only be completed by trains mapped over
that one track, which in practice is never. So where a single-track register line runs beside
another register line between two points on it, with no OSM route over it that does not also
run over the other, its ways go to the other line (between two such lines that are each
other's companion, the fixed rule picks the owner). It then owns nothing, drops out of the
lists, and riding either track credits the line that stays. Logged per country.

THE FOOTPRINT of a section is what riding it credits: a list of [owner section, from, to, a,
b], the stretch from..to along the owner section (fractions of its length, `to < from` when
it runs the other way) that the stretch a..b of this section lies on. An OSM section is cut
from the very ways of its route relation, so each of its segments is found on its way by node
ids, no buffer. The way's owner says which line; the way's pieces (PIECE_M long) are
projected once onto that line's nearest section within OWNER_SEC_M, which says which section
and where. A register section owns itself whole, except where it lies on rails another
register line owns (two lines drawn over one pair of tracks, Belgium's 161 and 161A at
Genval): that stretch is the other line's, strictly, and logged.

WHAT EACH SECTION OWNS is not shipped: it is the union of the footprints of its own line's
sections onto it (the app works it out as each country loads). Lengths and fractions along a
section are measured the way the app's sliceLine measures them, on a flat projection scaled
at the section's first point.
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Polygon

ROOT = Path(__file__).resolve().parent
R = 20037508.34 / 180.0

PIECE_M = 50.0            # a way is projected onto its owner in pieces this long
OWNER_SEC_M = 160.0       # a piece maps onto its owner line's nearest section within this
SNAP_M = 3.0              # a section segment found on no way by node ids: nearest drawn way
EXACT_TIE_M = 0.5         # register candidates this close are a tie, measured again
TIE_M = 8.0               # register lines this close to a way are "on its rails" (losers)
SAME_RAILS_M = 8.0
OWN_TRAMLIKE = {"tram", "light_rail", "subway"}
THROAT_M, THROAT_SHARE, PATTERN_REG_SHARE = 150.0, 0.6, 0.6
ABROAD_M = 150.0
JUMP_M = 100.0            # a footprint run breaks where the owner position jumps this far
LOST_MIN_KM = 0.05        # a register section loses a stretch to another only from this long
# Single-track companions
COMP_M, COMP_SHARE, COMP_TRACKS, COMP_LEN, COMP_FAR_M = 250.0, 0.4, 1.4, 1.5, 600.0


# ---------------------------------------------------------------- small helpers

def natkey(s):
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower())
                 for p in re.findall(r"\d+|\D+", s or ""))


def owner_key(line):
    """The fixed rule: lowest ref by natural sort, empty refs last, then name, then id."""
    ref = (line.get("ref") or "").strip()
    return (ref == "", natkey(ref), line.get("name") or "", line["id"])


def merc(lon, lat):
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.05, 85.05)
    return np.column_stack([np.asarray(lon, dtype=np.float64) * R,
                            np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180) * R])


def scale_at(lat):
    """Web Mercator metres per true metre at a latitude."""
    return 1.0 / max(math.cos(math.radians(float(lat))), 0.05)


def merge(iv, tol=0.0):
    """Union of [lo, hi] intervals, closing gaps up to tol."""
    iv = sorted([float(a), float(b)] for a, b in iv)
    if not iv:
        return []
    out = [iv[0]]
    for lo, hi in iv[1:]:
        if lo <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def app_tol(km):
    """The app's gap closing: 150 m, capped at 10% of the section."""
    return min(0.15 / km, 0.1) if km > 0 else 0.0


def inside(iv, x):
    for lo, hi in iv:
        if lo <= x <= hi:
            return True
    return False


def seg_hash(lo, hi):
    lo = lo.astype(np.uint64)
    hi = hi.astype(np.uint64)
    with np.errstate(over="ignore"):
        h = lo * np.uint64(0x9E3779B97F4A7C15)
        h = (h ^ (h >> np.uint64(29))) + hi * np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return h


# ---------------------------------------------------------------- sections

class Sec:
    """One section's geometry, in Web Mercator for projecting and in the app's own metric for
    the fractions written out."""
    __slots__ = ("gid", "li", "lid", "reg", "service", "closed", "km", "ll", "geom", "cm",
                 "ca", "sc", "ids")

    def __init__(self, gid, li, line, km, pts, closed):
        self.gid, self.li, self.lid = gid, li, line["id"]
        self.reg = line.get("src", "osm") != "osm"
        self.service = bool(line.get("service"))
        self.closed = closed
        self.km = km
        ll = np.asarray([[p[0], p[1]] for p in pts], dtype=np.float64)
        self.ll = ll
        xy = merc(ll[:, 0], ll[:, 1])
        self.geom = LineString(xy)
        d = np.hypot(*(xy[1:] - xy[:-1]).T)
        self.cm = np.concatenate([[0.0], np.cumsum(d)])
        kx = math.cos(math.radians(ll[0, 1]))
        da = np.hypot((ll[1:, 0] - ll[:-1, 0]) * kx, ll[1:, 1] - ll[:-1, 1])
        ca = np.concatenate([[0.0], np.cumsum(da)])
        self.ca = ca / ca[-1] if ca[-1] > 0 else np.linspace(0, 1, len(ca))
        self.sc = scale_at(ll[:, 1].mean())
        # build_model.Pts: the OSM node id of each point, -1 where a border cut made one up.
        ids = getattr(pts, "ids", None)
        self.ids = None if ids is None else np.asarray(ids, dtype=np.int64)

    def frac(self, d):
        """Mercator distance along the section -> the app's fraction of its length."""
        d = np.nan_to_num(np.asarray(d, dtype=np.float64), nan=0.0)
        n = len(self.cm)
        if n < 2 or self.cm[-1] <= 0:
            return np.zeros_like(d)
        i = np.clip(np.searchsorted(self.cm, d, side="right") - 1, 0, n - 2)
        seg = self.cm[i + 1] - self.cm[i]
        t = np.where(seg > 0, (d - self.cm[i]) / np.where(seg > 0, seg, 1), 0.0)
        return np.clip(self.ca[i] + np.clip(t, 0, 1) * (self.ca[i + 1] - self.ca[i]), 0, 1)


# ---------------------------------------------------------------- the step

def run(region, lines, geoms, route_users, stations, log, state=None, built_regions=None):
    from build_model import kind_family, norm_line_name, WAY_MIN_FRAC
    import time
    t_start = time.time()

    def say(msg):
        log(f"own: [{time.time() - t_start:5.1f}s] {msg}")

    byid = {l["id"]: l for l in lines}
    lidx = {l["id"]: i for i, l in enumerate(lines)}
    okey = {l["id"]: owner_key(l) for l in lines}

    # -------------------------------------------------- sections
    secs = {}
    by_line = defaultdict(list)
    for li, l in enumerate(lines):
        shut = set(l.get("closed") or [])
        g = geoms.get(l["id"], {})
        for a, b, km, gid in l["sections"]:
            pts = g.get(f"{a}|{b}")
            if not pts or len(pts) < 2:
                continue
            s = Sec(gid, li, l, km, pts, f"{a}|{b}" in shut)
            secs[gid] = s
            if not s.closed:
                by_line[li].append(gid)
    say(f"{len(secs)} sections with geometry")
    n_gid = 1 + max((sec[3] for l in lines for sec in l["sections"]), default=-1)
    sec_km = np.ones(n_gid + 1)          # the extra slot answers for -1
    sec_reg = np.zeros(n_gid + 1, dtype=bool)
    for g, s in secs.items():
        sec_km[g] = s.km
        sec_reg[g] = s.reg
    line_segs = {}

    def segs_of(li):
        """Every segment of line li's running sections, indexed: an STRtree of short pieces
        answers a nearest query in log time where whole sections, thousands of vertices
        long, made each distance a walk along them."""
        if li not in line_segs:
            gs = by_line.get(li, [])
            if not gs:
                line_segs[li] = None
                return None
            p0, p1, sg, c0, c1, prv, nxt = [], [], [], [], [], [], []
            for g in gs:
                s = secs[g]
                xy = np.asarray(s.geom.coords)
                m = len(xy) - 1
                p0.append(xy[:-1])
                p1.append(xy[1:])
                sg.append(np.full(m, g, dtype=np.int64))
                c0.append(s.ca[:-1])
                c1.append(s.ca[1:])
                ok = np.ones(m, dtype=bool)
                ok[0] = False
                prv.append(ok)
                ok = np.ones(m, dtype=bool)
                ok[-1] = False
                nxt.append(ok)
            p0, p1 = np.concatenate(p0), np.concatenate(p1)
            tree = STRtree(shapely.linestrings(np.stack([p0, p1], axis=1)))
            line_segs[li] = (tree, p0, p1, np.concatenate(sg), np.concatenate(c0),
                             np.concatenate(c1), np.concatenate(prv), np.concatenate(nxt))
        return line_segs[li]

    def onto(P, k, p0, p1, c0, c1):
        """Points P projected onto segments k: (distance, fraction along the section)."""
        d = p1[k] - p0[k]
        L2 = (d * d).sum(axis=1)
        t = np.clip(((P - p0[k]) * d).sum(axis=1) / np.where(L2 > 0, L2, 1), 0, 1)
        q = p0[k] + d * t[:, None]
        return np.hypot(*(P - q).T), c0[k] + t * (c1[k] - c0[k])

    def project(li, A, B, sc, radius_m):
        """Pieces A->B (Mercator) onto the nearest running section of line li within
        radius_m (true metres, scalar or per piece): (section gid or -1, from, to)."""
        n = len(A)
        sec = np.full(n, -1, dtype=np.int64)
        f0, f1 = np.zeros(n), np.zeros(n)
        got = segs_of(li) if n else None
        if got is None:
            return sec, f0, f1
        tree, p0, p1, sg, c0, c1, prv, nxt = got
        pad = np.broadcast_to(np.asarray(radius_m, dtype=np.float64) * sc, (n,))
        mids = shapely.points((A + B) / 2)
        (pi, si), d = tree.query_nearest(mids, max_distance=float(pad.max()),
                                         return_distance=True, all_matches=False)
        keep = d <= pad[pi]
        pi, si = pi[keep], si[keep]
        sec[pi] = sg[si]
        # Each end onto the segment the middle found, or the one before or after it on the
        # same section where the piece runs round a bend.
        for P, f in ((A[pi], f0), (B[pi], f1)):
            best_d, best_f = onto(P, si, p0, p1, c0, c1)
            for step, ok in ((-1, prv[si]), (1, nxt[si])):
                k = np.where(ok, si + step, si)
                dd, ff = onto(P, k, p0, p1, c0, c1)
                better = ok & (dd < best_d)
                best_d = np.where(better, dd, best_d)
                best_f = np.where(better, ff, best_f)
            f[pi] = best_f
        return sec, f0, f1

    # -------------------------------------------------- drawn ways
    if state is None:
        state = drawn_ways(region)
    ways, cid, cx, cy = state["ways"], state["cid"], state["cx"], state["cy"]
    wids = state["wids"]
    nW = len(wids)
    # All ways at once: one lookup of every node, then cut back into ways.
    raw = [np.asarray(ways[wid][1], dtype=np.int64) for wid in wids]
    cnt = np.array([len(r) for r in raw], dtype=np.int64)
    alln = np.concatenate(raw) if raw else np.zeros(0, np.int64)
    del raw
    pos = np.searchsorted(cid, alln)
    np.clip(pos, 0, cid.size - 1, out=pos)
    ok = cid[pos] == alln
    way_of = np.repeat(np.arange(nW), cnt)[ok]
    alln, pos = alln[ok], pos[ok]
    lon, lat = cx[pos] / 1e7, cy[pos] / 1e7
    allxy = merc(lon, lat)
    cnt = np.bincount(way_of, minlength=nW)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    W_sc = 1.0 / np.maximum(np.cos(np.radians(
        np.bincount(way_of, weights=lat, minlength=nW) / np.maximum(cnt, 1))), 0.05)
    same = way_of[1:] == way_of[:-1] if alln.size else np.zeros(0, bool)
    si = np.flatnonzero(same)                    # segment i joins point i and i + 1
    seg_u, seg_v, seg_w = alln[si], alln[si + 1], way_of[si]
    seg_a, seg_b = allxy[si], allxy[si + 1]
    S = int(si.size)
    seg_true = np.hypot(*(seg_b - seg_a).T) / W_sc[seg_w] if S else np.zeros(0)
    W_len = np.bincount(seg_w, weights=seg_true, minlength=nW) if S else np.zeros(nW)
    off = np.concatenate([[0], np.cumsum(np.bincount(seg_w, minlength=nW))]).astype(np.int64)
    W_nodes = [alln[start[j]:start[j] + cnt[j]] for j in range(nW)]
    W_kind = state["wkind"]
    W_hs, W_names, W_tracks = [], [], []
    for wid in wids:
        tags = ways[wid][0]
        W_hs.append(tags.get("highspeed") == "yes")
        name = tags.get("name")
        W_names.append({norm_line_name(p, tags.get("operator", ""))
                        for p in re.split(r"[・;/]", name) if p.strip()} if name else set())
        try:
            W_tracks.append(min(4, max(1, int(str(tags.get("tracks", "1")).split(";")[0]))))
        except ValueError:
            W_tracks.append(1)
    W_geo = state.get("wgeo") or [LineString(allxy[start[j]:start[j] + cnt[j]])
                                  for j in range(nW)]
    wtree = STRtree(W_geo)
    widx = {wid: j for j, wid in enumerate(wids)}

    # Every drawn way's segments, by the unordered pair of node ids at their ends.
    s_lo, s_hi = np.minimum(seg_u, seg_v), np.maximum(seg_u, seg_v)
    s_key = seg_hash(s_lo, s_hi)
    order = np.argsort(s_key, kind="stable")
    k_sorted = s_key[order]

    def find_segs(u, v):
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        key = seg_hash(lo, hi)
        p = np.searchsorted(k_sorted, key)
        p = np.clip(p, 0, max(S - 1, 0))
        if S == 0:
            return np.full(len(u), -1, dtype=np.int64)
        s = order[p]
        hit = (k_sorted[p] == key) & (s_lo[s] == lo) & (s_hi[s] == hi)
        return np.where(hit, s, -1)
    say(f"{nW} drawn ways, {S} segments indexed by node ids")

    # -------------------------------------------------- 1. register lines, by geometry
    reg_lines = [l for l in lines if l.get("src", "osm") != "osm"]
    skey_gid = {}
    for l in reg_lines:
        shut = set(l.get("closed") or [])
        for a, b, km, gid in l["sections"]:
            if f"{a}|{b}" not in shut and gid in secs:
                skey_gid[(l["id"], f"{a}|{b}")] = gid
    near = defaultdict(float)          # (j, line id) -> share of the way inside the buffer
    dist = {}                          # (j, line id) -> (metres, nearest section key)
    for (lid, skey), beside in (state.get("sec_ways") or {}).items():
        if (lid, skey) not in skey_gid:
            continue                   # dropped as unridden, or not running
        for j, got, dj in beside:
            near[(j, lid)] += got / max(W_geo[j].length, 1e-9)
            if dj < dist.get((j, lid), (math.inf,))[0]:
                dist[(j, lid)] = (dj, skey)
    cand = defaultdict(dict)
    for (j, lid), f in near.items():
        if f < WAY_MIN_FRAC:
            continue
        l = byid[lid]
        fam, wf = kind_family(l["kind"]), kind_family(W_kind[j])
        dj, skey = dist[(j, lid)]
        cross = False
        if wf != fam:
            if not ({wf, fam} <= OWN_TRAMLIKE and not l.get("guided") and dj <= SAME_RAILS_M):
                continue
            cross = True
        flag = (l.get("highspeed_sections") or {}).get(skey, l.get("highspeed"))
        hsm = None if flag is None else (bool(flag) == W_hs[j])
        key = norm_line_name(l["name"], l.get("operator", ""))
        cand[j][lid] = (dj, key in W_names[j], hsm, cross)

    reg_secs_of = defaultdict(list)
    for gid, s in secs.items():
        if s.reg and not s.closed:
            reg_secs_of[s.lid].append(s.geom)

    def mean_dist(j, lid):
        w = W_geo[j]
        n = max(2, min(60, int(w.length / W_sc[j] / 10)))
        pts = shapely.points(np.asarray([w.interpolate(i / (n - 1), normalized=True).coords[0]
                                         for i in range(n)]))
        env = w.buffer(60 * W_sc[j]).envelope
        gs = [g for g in reg_secs_of[lid] if g.intersects(env)] or reg_secs_of[lid]
        D = shapely.distance(pts[:, None], np.array(gs, dtype=object)[None, :])
        return float(np.nanmin(D, axis=1).mean()) / W_sc[j]

    owner = np.full(nW, -1, dtype=np.int64)       # line index
    status = np.zeros(nW, dtype=np.int8)          # see STATUS
    STATUS = ["none", "reg", "reg-throat", "reg-twin", "osm", "svc", "abroad", "reg-pair"]
    losers = defaultdict(set)                     # way -> register lines on its rails that lost
    same_rails = []                               # (way, owner, [lines tied exactly])
    n_hs_override = n_tie_geom = n_tie_rule = 0
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
        exact = [k for k, v in pool.items() if v[0] <= dmin + EXACT_TIE_M]
        if len(exact) > 1:
            md = {k: mean_dist(j, k) for k in exact}
            mmin = min(md.values())
            exact2 = [k for k in exact if md[k] <= mmin + EXACT_TIE_M]
            if len(exact2) < len(exact):
                n_tie_geom += 1
            else:
                n_tie_rule += 1
            exact = exact2
        o = min(exact, key=lambda k: okey[k])
        owner[j], status[j] = lidx[o], 1
        if len(exact) > 1:
            same_rails.append((j, o, sorted(set(exact) - {o})))
        for k, v in cd.items():
            if k != o and v[0] <= dmin + TIE_M:
                losers[j].add(k)
    say(f"{int((status == 1).sum())} ways owned by a register line by geometry; "
        f"{n_hs_override} took a register line whose high-speed flag disagrees (none agreeing "
        f"near); midpoint ties: {n_tie_geom} settled by mean distance, {n_tie_rule} by the "
        f"fixed rule (register lines drawn on the same rails)")

    # -------------------------------------------------- route users
    users = {}
    line_ways = defaultdict(list)
    for wid, lids in route_users.items():
        j = widx.get(wid)
        if j is None:
            continue
        got = {x for x in lids if x in byid}
        if got:
            users[j] = got
            for x in got:
                line_ways[x].append(j)

    # -------------------------------------------------- 5. abroad (only ways no register owns)
    n_abroad = 0
    outline = None
    reg_json = (built_regions or {}).get(region)
    if reg_json and reg_json.get("parts"):
        polys = []
        for ring in reg_json["parts"]:
            r = ring[0] if ring and isinstance(ring[0][0], list) else ring
            arr = np.asarray(r, dtype=np.float64)
            if len(arr) >= 3:
                polys.append(Polygon(merc(arr[:, 0], arr[:, 1])).buffer(0))
        if polys:
            lat_mid = (reg_json["bbox"][1] + reg_json["bbox"][3]) / 2
            outline = shapely.union_all(polys).buffer(ABROAD_M * scale_at(lat_mid))
            shapely.prepare(outline)
    wmid = shapely.points(np.array([W_geo[j].interpolate(0.5, normalized=True).coords[0]
                                    for j in range(nW)])) if nW else np.array([])
    if outline is not None:
        free = np.flatnonzero(owner < 0)
        if free.size:
            out_ = ~shapely.contains(outline, wmid[free])
            status[free[out_]] = 6
            n_abroad = int(out_.sum())
    say(f"{n_abroad} drawn ways outside the country's outline and owned by no register line: "
        f"left to the neighbour")

    # -------------------------------------------------- 2. station throats
    reg_share = {}
    for lid, js in line_ways.items():
        l = byid[lid]
        if l.get("src", "osm") != "osm":
            continue
        tot = sum(W_len[j] for j in js)
        regk = sum(W_len[j] for j in js if status[j] == 1)
        reg_share[lid] = regk / tot if tot else 0.0
    rsec = [g for g, s in secs.items() if s.reg and not s.closed]
    rtree = STRtree([secs[g].geom for g in rsec]) if rsec else None
    n_throat, km_throat = 0, 0.0
    if rtree is not None:
        for j in range(nW):
            if owner[j] >= 0 or status[j] == 6:
                continue
            us = [k for k in users.get(j, ()) if not byid[k].get("service")]
            if not us or any(byid[k].get("src", "osm") != "osm" for k in us):
                continue
            if min(reg_share.get(k, 0.0) for k in us) < PATTERN_REG_SHARE:
                continue
            sc = W_sc[j]
            buf = W_geo[j].buffer(THROAT_M * sc, quad_segs=2)
            best = None
            for i in rtree.query(buf):
                s = secs[rsec[i]]
                if kind_family(lines[s.li]["kind"]) != kind_family(W_kind[j]):
                    continue
                share = (W_geo[j].intersection(s.geom.buffer(THROAT_M * sc, quad_segs=2)).length
                         / max(W_geo[j].length, 1e-9))
                dd = s.geom.distance(wmid[j])
                if share >= THROAT_SHARE and (best is None or dd < best[0]):
                    best = (dd, s.li)
            if best:
                owner[j], status[j] = best[1], 2
                n_throat += 1
                km_throat += W_len[j] / 1000
    say(f"station throats: {n_throat} ways ({km_throat:.1f} km) used only by operating "
        f"patterns, within {THROAT_M:.0f} m of a register line, given to it")

    # -------------------------------------------------- 3, 4. twins, then OSM lines
    n_twin = 0
    osm_contest = []
    for j in range(nW):
        if owner[j] >= 0 or status[j] == 6:
            continue
        us = users.get(j, set())
        regu = [k for k in us if byid[k].get("src", "osm") != "osm"]
        if regu:
            owner[j], status[j] = lidx[min(regu, key=lambda k: okey[k])], 3
            n_twin += 1
            continue
        osmu = [k for k in us if not byid[k].get("service")]
        if osmu:
            o = min(osmu, key=lambda k: okey[k])
            owner[j], status[j] = lidx[o], 4
            if len(osmu) > 1:
                osm_contest.append((j, o, sorted(set(osmu) - {o})))
            continue
        status[j] = 5 if us else 0
    say(f"{n_twin} ways given to a register line through its merged OSM twin; "
        f"{int((status == 4).sum())} owned by an OSM line, {len(osm_contest)} of them run over "
        f"by several (the fixed rule decides); {int((status == 5).sum())} run over only by "
        f"named trains, owned by nobody")

    # -------------------------------------------------- single-track companions
    companions = find_companions(lines, byid, lidx, okey, secs, by_line, owner, status, users,
                                 W_geo, W_len, W_sc, W_tracks, W_kind, wtree, kind_family, say)
    pair_far = np.zeros(nW, dtype=bool)
    for x, (y, _info) in companions.items():
        xi, yi = lidx[x], lidx[y]
        for j in np.flatnonzero(owner == xi):
            owner[j], status[j] = yi, 7
            losers[int(j)].add(x)
            pair_far[j] = True

    # -------------------------------------------------- pieces of every owned way, projected
    # Each way segment is cut into pieces of at most PIECE_M, the same cut everywhere, so a
    # piece is identified by (segment, index) and projected once whatever runs over it.
    seg_len_m = seg_true
    pcnt = np.maximum(1, np.ceil(seg_len_m / PIECE_M)).astype(np.int64)
    pstart = np.concatenate([[0], np.cumsum(pcnt)[:-1]]).astype(np.int64) if S \
        else np.zeros(0, np.int64)
    P = int(pcnt.sum()) if S else 0
    P_seg = np.repeat(np.arange(S), pcnt)
    q = np.arange(P) - pstart[P_seg]
    P_t0 = q / pcnt[P_seg]
    P_t1 = (q + 1) / pcnt[P_seg]
    del q
    P_sec = np.full(P, -1, dtype=np.int64)
    P_f0 = np.zeros(P)
    P_f1 = np.zeros(P)
    P_way = seg_w[P_seg] if P else np.zeros(0, np.int64)

    def piece_ends(idx):
        s = P_seg[idx]
        dv = seg_b[s] - seg_a[s]
        return (seg_a[s] + dv * P_t0[idx][:, None], seg_a[s] + dv * P_t1[idx][:, None],
                W_sc[seg_w[s]])

    P_own = owner[P_way] if P else np.zeros(0, np.int64)
    radius = np.where(pair_far[P_way], COMP_FAR_M, OWNER_SEC_M) if P else np.zeros(0)
    order_own = np.argsort(P_own, kind="stable")
    bounds = np.searchsorted(P_own[order_own], np.arange(len(lines) + 1))
    for li in np.unique(P_own[P_own >= 0]):
        idx = order_own[bounds[li]:bounds[li + 1]]
        A, B, sc = piece_ends(idx)
        P_sec[idx], P_f0[idx], P_f1[idx] = project(int(li), A, B, sc, radius[idx])
    plen = (P_t1 - P_t0) * seg_len_m[P_seg] if P else np.zeros(0)
    n_far = float(plen[(P_own >= 0) & (P_sec < 0)].sum())
    say(f"{P} pieces of ways projected onto their owner's sections; "
        f"{n_far / 1000:.1f} km of owned way found no section of its owner within reach")

    # -------------------------------------------------- footprints of OSM sections
    foot = {}
    comp = Counter()
    n_fallback = 0
    for gid, s in secs.items():
        if s.reg:
            continue
        ll = s.ll
        n = len(ll)
        if s.ids is not None and len(s.ids) == n:
            u, v = s.ids[:-1], s.ids[1:]
            sid = find_segs(u, v)
        else:
            u = None
            sid = np.full(n - 1, -1, dtype=np.int64)
        ca = s.ca
        rows = []                     # (segment index along s, a0, a1, owner sec, from, to)
        found = sid >= 0
        if found.any():
            fi = np.flatnonzero(found)
            ss = sid[fi]
            fwd = seg_u[ss] == u[fi]
            cnt = pcnt[ss]
            rep = np.repeat(np.arange(fi.size), cnt)
            within = np.arange(rep.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            fw = fwd[rep]
            pidx = pstart[ss][rep] + np.where(fw, within, cnt[rep] - 1 - within)
            t0 = np.where(fw, P_t0[pidx], 1 - P_t1[pidx])
            t1 = np.where(fw, P_t1[pidx], 1 - P_t0[pidx])
            seg_i = fi[rep]
            a0 = ca[seg_i] + t0 * (ca[seg_i + 1] - ca[seg_i])
            a1 = ca[seg_i] + t1 * (ca[seg_i + 1] - ca[seg_i])
            pf0 = np.where(fw, P_f0[pidx], P_f1[pidx])
            pf1 = np.where(fw, P_f1[pidx], P_f0[pidx])
            st = status[P_way[pidx]]
            for code in np.unique(st):
                comp[STATUS[code]] += float(((a1 - a0) * (st == code)).sum()) * s.km
            rows.append((seg_i, a0, a1, P_sec[pidx], pf0, pf1))
        miss = np.flatnonzero(~found)
        if miss.size:
            # A point made up where a border cuts a segment, or a straight line over a gap:
            # the nearest drawn way within SNAP_M, projected here and now.
            xy = merc(ll[:, 0], ll[:, 1])
            mids = shapely.points((xy[miss] + xy[miss + 1]) / 2)
            got = wtree.query_nearest(mids, max_distance=SNAP_M * s.sc, all_matches=False)
            near_way = dict(zip(got[0].tolist(), got[1].tolist()))
            for pi, i in enumerate(miss.tolist()):
                j = near_way.get(pi)
                piece = (ca[i + 1] - ca[i]) * s.km
                if j is None or owner[j] < 0:
                    comp["unmatched" if j is None else STATUS[status[j]]] += piece
                    continue
                n_fallback += 1
                comp[STATUS[status[j]]] += piece
                m = max(1, int(math.ceil(piece * 1000 / PIECE_M)))
                tt = np.linspace(0, 1, m + 1)
                pts = xy[i] + (xy[i + 1] - xy[i]) * tt[:, None]
                sec, f0, f1 = project(int(owner[j]), pts[:-1], pts[1:], np.full(m, s.sc),
                                      OWNER_SEC_M)
                a0 = ca[i] + tt[:-1] * (ca[i + 1] - ca[i])
                a1 = ca[i] + tt[1:] * (ca[i + 1] - ca[i])
                rows.append((np.full(m, i), a0, a1, sec, f0, f1))
        if not rows:
            foot[gid] = []
            continue
        seg_i = np.concatenate([r[0] for r in rows])
        o = np.argsort(seg_i, kind="stable")
        a0, a1, ps, f0, f1 = (np.concatenate([r[k] for r in rows])[o] for k in range(1, 6))
        foot[gid] = runs(a0, a1, ps, f0, f1, sec_km)
    say(f"footprints for {sum(1 for s in secs.values() if not s.reg)} OSM sections "
        f"({n_fallback} segments found by nearest way rather than by node ids)")

    # -------------------------------------------------- register sections: same rails
    # A register line X that was on the rails of a way another register line Y won: where X
    # keeps no way of its own along that stretch, the stretch is Y's.
    lost_by = defaultdict(list)         # X line index -> piece indices it lost
    for j, xs in losers.items():
        if owner[j] < 0 or lines[int(owner[j])].get("src", "osm") == "osm":
            continue
        a = pstart[off[j]] if off[j + 1] > off[j] else None
        if a is None:
            continue
        b = pstart[off[j + 1] - 1] + pcnt[off[j + 1] - 1]
        for x in xs:
            if x in lidx and lidx[x] != owner[j]:
                lost_by[lidx[x]].append(np.arange(a, b))
    lost_rows = defaultdict(list)       # X section gid -> arrays (fx0, fx1, tY, fY0, fY1)
    for xi, chunks in lost_by.items():
        idx = np.concatenate(chunks)
        idx = idx[P_sec[idx] >= 0]      # the owner must have it, or X keeps it
        if not idx.size:
            continue
        A, B, sc = piece_ends(idx)
        sx, fx0, fx1 = project(xi, A, B, sc, radius[idx])
        ok = sx >= 0
        for g in np.unique(sx[ok]):
            m = idx[sx == g]
            mm = sx == g
            lost_rows[int(g)].append((fx0[mm], fx1[mm], P_sec[m], P_f0[m], P_f1[m]))
    kp = np.flatnonzero((P_sec >= 0) & sec_reg[P_sec] & (P_own >= 0))
    kp = kp[np.argsort(P_sec[kp], kind="stable")]
    kp_sec = P_sec[kp]
    lost_km = defaultdict(float)        # (X, Y) line ids -> km
    lost_where = defaultdict(list)      # (X, Y) -> [(lon, lat, km)]
    for g, chunks in lost_rows.items():
        s = secs[g]
        lo_, hi_ = np.searchsorted(kp_sec, g), np.searchsorted(kp_sec, g, side="right")
        kk = kp[lo_:hi_]
        K = merge(zip(np.minimum(P_f0[kk], P_f1[kk]), np.maximum(P_f0[kk], P_f1[kk])),
                  app_tol(s.km))
        x0, x1, ty, y0, y1 = (np.concatenate([c[k] for c in chunks]) for k in range(5))
        mid = (x0 + x1) / 2
        only = np.array([not inside(K, x) for x in mid.tolist()], dtype=bool)
        if not only.any():
            continue
        x0, x1, ty, y0, y1 = x0[only], x1[only], ty[only], y0[only], y1[only]
        LO = merge(zip(np.minimum(x0, x1), np.maximum(x0, x1)), 0.03 / max(s.km, 1e-6))
        if sum(hi - lo for lo, hi in LO) * s.km < LOST_MIN_KM:
            continue
        entries = []
        cur = 0.0
        for lo, hi in LO:
            if lo - cur > 1e-4:
                entries.append([g, round(cur, 4), round(lo, 4), round(cur, 4), round(lo, 4)])
            cur = max(cur, hi)
        if 1.0 - cur > 1e-4:
            entries.append([g, round(cur, 4), 1.0, round(cur, 4), 1.0])
        flip = x0 > x1
        pa, pb = np.where(flip, x1, x0), np.where(flip, x0, x1)
        pf0, pf1 = np.where(flip, y1, y0), np.where(flip, y0, y1)
        o = np.lexsort((pa, ty))
        got = runs(pa[o], pb[o], ty[o], pf0[o], pf1[o], sec_km, overlap=True)
        entries += got
        entries.sort(key=lambda e: e[3])
        foot[g] = entries
        midpt = s.ll[len(s.ll) // 2]
        for e in got:
            k = (e[4] - e[3]) * s.km
            lost_km[(s.lid, secs[e[0]].lid)] += k
            lost_where[(s.lid, secs[e[0]].lid)].append((float(midpt[0]), float(midpt[1]), k))
    say(f"register sections on rails another register line owns: {len(lost_km)} pairs of "
        f"lines, {sum(lost_km.values()):.1f} km counted for the line that owns the rails")

    # -------------------------------------------------- tidy: drop footprints that are self
    n_self = 0
    for g in list(foot):
        f = foot[g]
        if (len(f) == 1 and f[0][0] == g and f[0][1] <= 1e-3 and f[0][2] >= 0.999
                and f[0][3] <= 1e-3 and f[0][4] >= 0.999 and not secs[g].service):
            del foot[g]
            n_self += 1
    say(f"{len(foot)} sections whose footprint is not simply themselves, {n_self} OSM sections "
        f"own themselves whole")

    report = dict(
        comp=comp, osm_contest=osm_contest, same_rails=same_rails, lost_km=lost_km,
        lost_where=lost_where, companions=companions, status=status, owner=owner, users=users,
        W_len=W_len, W_geo=W_geo, W_nodes=W_nodes, wids=wids, reg_share=reg_share, secs=secs,
        foot=foot, lines=lines)
    log_report(report, stations, say)
    return foot, report


def runs(a0, a1, ps, f0, f1, sec_km, overlap=False):
    """Pieces along a section, in order -> footprint entries [t, from, to, a, b]: consecutive
    pieces on one owner section, without a jump along it, are one entry. `overlap`: the pieces
    may overlap along the section (both tracks of a pair projected onto one line), sorted by
    owner section and then by a0; a run then spans their extent."""
    out = []
    n = len(a0)
    if not n:
        return out
    if overlap:
        i = 0
        while i < n:
            t = int(ps[i])
            lo, hi, k_lo, k_hi = a0[i], a1[i], i, i
            k = i + 1
            tol = 0.03 / max(sec_km[t], 1e-6)
            while (k < n and ps[k] == t and a0[k] <= hi + tol
                   and abs(f0[k] - f1[k_hi]) * sec_km[t] * 1000 <= JUMP_M):
                if a1[k] > hi:
                    hi, k_hi = a1[k], k
                k += 1
            out.append([t, round(float(f0[k_lo]), 4), round(float(f1[k_hi]), 4),
                        round(float(lo), 4), round(float(hi), 4)])
            i = k
        return [e for e in out if e[4] - e[3] >= 1e-4 or abs(e[2] - e[1]) >= 1e-4]
    valid = ps >= 0
    km = sec_km[ps]
    brk = np.ones(n, dtype=bool)
    if n > 1:
        same = (ps[1:] == ps[:-1]) & valid[1:] & valid[:-1]
        jump = np.abs(f0[1:] - f1[:-1]) * km[1:] * 1000 > JUMP_M
        gap = (a0[1:] - a1[:-1]) > 1e-6
        brk[1:] = ~same | jump | gap
    first = np.flatnonzero(brk)
    last = np.concatenate([first[1:] - 1, [n - 1]])
    keep = valid[first]
    first, last = first[keep], last[keep]
    e_t = ps[first]
    e_f, e_to = np.round(f0[first], 4), np.round(f1[last], 4)
    e_a, e_b = np.round(a0[first], 4), np.round(a1[last], 4)
    ok = ((e_b - e_a) >= 1e-4) | (np.abs(e_to - e_f) >= 1e-4)
    for t, f, to, a, b in zip(e_t[ok].tolist(), e_f[ok].tolist(), e_to[ok].tolist(),
                              e_a[ok].tolist(), e_b[ok].tolist()):
        out.append([t, f, to, a, b])
    return out


def find_companions(lines, byid, lidx, okey, secs, by_line, owner, status, users, W_geo, W_len,
                    W_sc, W_tracks, W_kind, wtree, kind_family, say):
    """Single-track register lines that are one track of another register line's route:
    {line id: (the line whose track it is, info)}. See the module docstring.

    X gives its track to Y when: both are register lines of one kind family and one operator;
    X's own ways add up to at most COMP_TRACKS times its length (one track); at least
    COMP_SHARE of X lies within COMP_M of Y, and both of X's ends do; X is at most COMP_LEN
    times as long as Y between those ends (beside Y, not a loop off it); and every OSM route
    over X's track also runs over Y's track there."""
    out = {}
    reg = [l for l in lines if l.get("src", "osm") != "osm"]
    owned = defaultdict(list)
    for j in np.flatnonzero(owner >= 0):
        owned[int(owner[j])].append(int(j))
    rs = [g for l in reg for g in by_line.get(lidx[l["id"]], [])]
    if not rs:
        return out
    rs_line = np.array([secs[g].lid for g in rs], dtype=object)
    sec_tree = STRtree([secs[g].geom for g in rs])
    cands = {}
    for l in reg:
        x = l["id"]
        li = lidx[x]
        gsx = by_line.get(li, [])
        js = owned.get(li, [])
        lenx = sum(secs[g].km for g in gsx) * 1000
        if not gsx or not js or lenx <= 0:
            continue
        if sum(W_len[j] * W_tracks[j] for j in js) > COMP_TRACKS * lenx:
            continue
        sc = secs[gsx[0]].sc
        pad = COMP_M * sc
        # The line's ends: stations at one section only.
        deg = Counter(s for sec in l["sections"] for s in sec[:2])
        ends = {sec[3]: (sec[0], sec[1]) for sec in l["sections"]}
        endpts = []
        for g in gsx:
            s = secs[g]
            a, b = ends.get(g, (None, None))
            if deg.get(a) == 1:
                endpts.append(np.asarray(s.geom.coords[0]))
            if deg.get(b) == 1:
                endpts.append(np.asarray(s.geom.coords[-1]))
        if len(endpts) != 2:
            continue
        endpts = shapely.points(np.array(endpts))
        # Points every 25 m or so along X, and the register lines near each.
        gx = shapely.multilinestrings([secs[g].geom for g in gsx]) if len(gsx) > 1 \
            else secs[gsx[0]].geom
        n = int(min(400, max(4, lenx / 25)))
        probe = shapely.line_interpolate_point(gx, np.linspace(0, 1, n), normalized=True)
        pi, si = sec_tree.query(probe, predicate="dwithin", distance=pad)
        near_lines = defaultdict(set)
        for p, y in zip(pi.tolist(), rs_line[si].tolist()):
            near_lines[y].add(p)
        ei, es = sec_tree.query(endpts, predicate="dwithin", distance=pad)
        end_lines = defaultdict(set)
        for p, y in zip(ei.tolist(), rs_line[es].tolist()):
            end_lines[y].add(p)
        ux = set()
        for j in js:
            ux |= users.get(j, set())
        near_ways = None
        best = None
        for y, ps in near_lines.items():
            share = len(ps) / n
            if y == x or share < COMP_SHARE or len(end_lines.get(y, ())) < 2:
                continue
            if kind_family(byid[y]["kind"]) != kind_family(l["kind"]):
                continue
            # One railway's two tracks: the register names one operator for both.
            if (byid[y].get("operator") or "") != (l.get("operator") or ""):
                continue
            # Beside the other line, not a loop off it: about as long as the stretch of the
            # other line between its two ends.
            dy = float(shapely.distance(endpts[0], endpts[1])) / sc
            gys = by_line.get(lidx[y], [])
            if len(gys) == 1:
                gy = secs[gys[0]].geom
                dy = max(dy, abs(float(shapely.line_locate_point(gy, endpts[1]))
                                 - float(shapely.line_locate_point(gy, endpts[0]))) / sc)
            if lenx > COMP_LEN * dy + 300:
                continue
            if near_ways is None:
                cand = wtree.query(gx, predicate="dwithin", distance=pad)
                near_ways = [int(j) for j in cand]
            uy = set()
            for j in near_ways:
                if owner[j] == lidx[y]:
                    uy |= users.get(j, set())
            if not ux <= uy:
                continue
            if best is None or share > best[1]:
                best = (y, share)
        if best:
            cands[x] = (best[0], {"share": best[1], "km": lenx / 1000})
    # Each other's companion: the fixed rule keeps one. Then follow chains to a line that stays.
    for x in list(cands):
        y = cands[x][0]
        if y in cands and cands[y][0] == x and okey[x] < okey[y]:
            del cands[x]
    for x, (y, info) in cands.items():
        seen = {x}
        while y in cands and y not in seen:
            seen.add(y)
            y = cands[y][0]
        if y in seen:
            continue
        out[x] = (y, info)
    say(f"single-track companions: {len(out)} register lines give their track to the line "
        f"whose route it is ({sum(i['km'] for _y, i in out.values()):.1f} km)")
    for x, (y, info) in sorted(out.items(), key=lambda kv: -kv[1][1]["km"]):
        say(f"    {byid[x].get('ref') or ''} {byid[x]['name']} ({info['km']:.1f} km) -> "
            f"{byid[y].get('ref') or ''} {byid[y]['name']}")
    return out


# ---------------------------------------------------------------- report

def nearest_name(stations, lon, lat):
    if not stations:
        return ""
    best, bd = "", math.inf
    k = math.cos(math.radians(lat)) ** 2
    for s in stations:
        d = (s[0] - lon) ** 2 * k + (s[1] - lat) ** 2
        if d < bd:
            best, bd = s[2], d
    return best


def places(js, W_nodes, W_geo=None, near_m=30.0):
    """Ways joined into places: by a shared end node, or (given W_geo) lying within near_m
    of each other, so both tracks of a pair are one place."""
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for j in js:
        ns = W_nodes[j]
        if len(ns) == 0:
            continue
        r = find(("w", j))
        for n in (int(ns[0]), int(ns[-1])):
            parent[find(("n", n))] = r
    if W_geo is not None and len(js) > 1:
        js_ = list(js)
        geo = [W_geo[j] for j in js_]
        tree = STRtree(geo)
        for a, g in enumerate(geo):
            y = g.centroid.y
            sc = scale_at(math.degrees(2 * math.atan(math.exp(y / R * math.pi / 180)) - math.pi / 2))
            for b in tree.query(g, predicate="dwithin", distance=near_m * sc):
                ra, rb = find(("w", js_[a])), find(("w", js_[int(b)]))
                if ra != rb:
                    parent[rb] = ra
    groups = defaultdict(list)
    for j in js:
        groups[find(("w", j))].append(j)
    return list(groups.values())


def log_report(rep, stations, say):
    lines, secs = rep["lines"], rep["secs"]
    W_len, W_geo, W_nodes = rep["W_len"], rep["W_geo"], rep["W_nodes"]
    status, owner, users = rep["status"], rep["owner"], rep["users"]
    st = [(s["lon"], s["lat"], s.get("name") or s.get("n") or "") for s in stations.values()
          if not s.get("junction")] if stations else []

    def lonlat(j):
        x, y = W_geo[j].interpolate(0.5, normalized=True).coords[0]
        return x / R, math.degrees(2 * math.atan(math.exp(y / R * math.pi / 180)) - math.pi / 2)

    def lab(li):
        l = lines[li]
        return f"{(l.get('ref') or '').strip()} {l['name']}".strip()

    def where(js):
        k = sum(W_len[j] for j in js)
        lon = sum(lonlat(j)[0] * W_len[j] for j in js) / max(k, 1e-9)
        lat = sum(lonlat(j)[1] * W_len[j] for j in js) / max(k, 1e-9)
        return nearest_name(st, lon, lat)

    # Track the fixed rule decided: several OSM lines over one way.
    by_set = defaultdict(list)
    for j, o, others in rep["osm_contest"]:
        by_set[(o, tuple(others))].append(j)
    rows = []
    for (o, others), js in by_set.items():
        for comp in places(js, W_nodes, W_geo):
            rows.append((sum(W_len[j] for j in comp) / 1000, o, others, comp))
    rows.sort(key=lambda r: -r[0])
    li_of = {l["id"]: i for i, l in enumerate(lines)}
    say(f"fixed rule (shared track no register covers): {len(rows)} places, "
        f"{sum(r[0] for r in rows):.1f} km of way length (both tracks of a pair counted)")
    for k, o, others, comp in rows[:40]:
        say(f"    {k:6.2f} km  {lab(li_of[o])} over "
            f"{', '.join(lab(li_of[x]) for x in others)[:80]}  near {where(comp)}")

    # Register lines drawn on the same rails, and what each lost (companions apart).
    comp_ = rep["companions"]
    lost = [((x, y), km) for (x, y), km in rep["lost_km"].items() if x not in comp_]
    lost.sort(key=lambda kv: -kv[1])
    say(f"register lines on the same rails: {len(rep['same_rails'])} ways two register lines "
        f"are drawn over, given to one by the fixed rule; {len(lost)} lines lose a stretch to "
        f"another, {sum(k for _p, k in lost):.1f} km in all (single-track companions apart):")
    small = [k for _p, k in lost if k < 0.2]
    for (x, y), km in lost:
        if km < 0.2:
            break
        pts = rep["lost_where"][(x, y)]
        lon = sum(p[0] * p[2] for p in pts) / max(sum(p[2] for p in pts), 1e-9)
        lat = sum(p[1] * p[2] for p in pts) / max(sum(p[2] for p in pts), 1e-9)
        say(f"    {km:6.2f} km of {lab(li_of[x])} counts as {lab(li_of[y])}  near "
            f"{nearest_name(st, lon, lat)}")
    if small:
        say(f"    and {len(small)} stretches under 0.2 km ({sum(small):.1f} km), mostly junctions")

    # Likely register gaps: track an operating pattern owns because no register line covers it.
    reg_share = rep["reg_share"]
    gaps = defaultdict(list)
    for j in np.flatnonzero(status == 4):
        o = lines[int(owner[j])]
        if reg_share.get(o["id"], 0.0) >= PATTERN_REG_SHARE:
            gaps[int(owner[j])].append(int(j))
    rows = []
    for li, js in gaps.items():
        for comp in places(js, W_nodes, W_geo):
            rows.append((sum(W_len[j] for j in comp) / 1000, li, comp))
    rows.sort(key=lambda r: -r[0])
    say(f"likely register gaps (track owned by an operating pattern that otherwise runs on "
        f"register lines): {sum(r[0] for r in rows):.1f} km of way length in {len(rows)} places")
    for k, li, comp in rows[:30]:
        if k < 0.5:
            break
        say(f"    {k:6.2f} km  {lab(li)}  near {where(comp)}")
    svc = [int(j) for j in np.flatnonzero(status == 5)]
    rows = sorted(((sum(W_len[j] for j in c) / 1000, c) for c in places(svc, W_nodes, W_geo)),
                  key=lambda r: -r[0])
    say(f"track only named trains run over (owned by nobody, counted nowhere): "
        f"{sum(r[0] for r in rows):.1f} km of way length")
    for k, comp in rows[:12]:
        if k < 0.5:
            break
        names = Counter(u for j in comp for u in users.get(j, ()))
        say(f"    {k:6.2f} km  near {where(comp)}: "
            f"{', '.join(lab(li_of[u]) for u, _ in names.most_common(2))[:80]}")
    comp = rep["comp"]
    say("OSM section km by the track under it: "
        + ", ".join(f"{k} {v:,.1f}" for k, v in comp.most_common()))


# ---------------------------------------------------------------- without a register

def drawn_ways(region):
    """The drawn ways when no register was merged (register_way_lines did not run)."""
    import pickle
    from build_tiles import KIND, rank_of
    d = ROOT / "data" / "proc" / region
    with open(d / "ways.pkl", "rb") as f:
        ways = pickle.load(f)
    with open(d / "rels.pkl", "rb") as f:
        rels = pickle.load(f)
    c = np.load(d / "coords.npz")
    cid, cx, cy = c["id"], c["x"], c["y"]
    on_route = {ref for tags, members in rels.values() if tags.get("type") == "route"
                for ty, ref, role in members
                if ty == "w" and (not role or role.startswith(("forward", "backward")))}
    wids, wkind = [], []
    for wid, (tags, nodes) in ways.items():
        kind = KIND.get(tags.get("railway"))
        if kind is None or (rank_of(kind, tags) >= 2 and wid not in on_route):
            continue
        pos = np.searchsorted(cid, nodes)
        np.clip(pos, 0, cid.size - 1, out=pos)
        if (cid[pos] == nodes).sum() < 2:
            continue
        wids.append(wid)
        wkind.append(kind)
    return {"ways": ways, "cid": cid, "cx": cx, "cy": cy, "wids": wids, "wkind": wkind}


SCALE = 10000             # foot.json writes fractions as integers in these units


def write(out, region, foot):
    """foot.json: {"region", "scale": SCALE, "foot": {section id: [entry, ...]}}, only for
    sections whose footprint is not simply themselves whole (a named train's always is not).
    Each entry is [owner section, from, to, a, b] with the fractions as integers in SCALE
    units, the entries in order of a. Shorter where it can be: an entry whose a is the last
    entry's b (0 for the first) leaves a out, [t, from, to, b]; one that also runs to the end
    of the section leaves b out too, [t, from, to]. Russia: 3.6 MB against 23.5 MB for the
    credits.json this replaces."""
    def enc(entries):
        rows, prev = [], 0
        for t, f, to, a, b in sorted(entries, key=lambda e: (e[3], e[4])):
            F, T, A, B = (int(round(x * SCALE)) for x in (f, to, a, b))
            if A != prev:
                rows.append([t, F, T, A, B])
            elif B == SCALE:
                rows.append([t, F, T])
            else:
                rows.append([t, F, T, B])
            prev = B
        return rows
    with open(out / "foot.json", "w", encoding="utf-8") as fh:
        json.dump({"region": region, "scale": SCALE,
                   "foot": {str(g): enc(f) for g, f in sorted(foot.items())}},
                  fh, separators=(",", ":"))


def read(path):
    """foot.json back into {section id: [[t, from, to, a, b], ...]} as fractions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scale = data.get("scale", 1)
    out = {}
    for k, rows in data["foot"].items():
        prev, got = 0, []
        for e in rows:
            a = e[3] if len(e) == 5 else prev
            b = e[4] if len(e) == 5 else e[3] if len(e) == 4 else scale
            prev = b
            got.append([e[0], e[1] / scale, e[2] / scale, a / scale, b / scale])
        out[int(k)] = got
    return out
