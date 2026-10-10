"""SECTIONS THAT RUN ALONGSIDE ONE ANOTHER (2026-10-07; handoff_notes/full_line_not_100.md).

    python along.py --region us                 # from dist/data/us, write dist/data/us/along.json
    python along.py --region us --out <dir>     # write elsewhere (a trial)

Anita: "the B train and Q train are not identified as the same thing ... we're probably
missing a lot of other cases around the world where lines are running on the same tracks".
Footprints (foot.json, ownership.py) say which OWNER track a section lies on, by OSM ways; two
services drawn over different ways of one corridor (the two directions' tracks, a local and an
express track side by side, a relation drawn a few metres off) share no way and never meet
there. This measures it by geometry instead: for every section S, every other section T of the
same kind of rail (kind_family: metro with metro, rail with rail) and not one high-speed and
the other conventional, the stretches of S lying within ALONG_M of T. The app counts S ridden
on its own line when a ride's sections run alongside nearly all of it (ALONG_SHARE in
index.html), and only on its own line: the owner track under S gets nothing more than the ride
gave it, so country and operator totals do not move.

along.json: {"region", "scale": SCALE, "d": ALONG_M, "along": {S: [[T, a, b], ...]}} with a, b
the stretch of S (fractions in SCALE units) within ALONG_M of T. Only stretches of at least
MIN_SHARE of S or MIN_M are kept.

High-speed or not, per section: its own line's register flag where it has one, else the flag
of the lines owning most of its footprint (a Shinkansen named train lies on the Shinkansen
register line); unknown matches either.
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely import LineString, STRtree

ROOT = Path(__file__).resolve().parent
SCALE = 1000
ALONG_M = 30.0        # within this of the other track: adjacent tracks are 4-15 m apart in OSM
STEP_M = 10.0         # sample spacing along a section
MAX_SAMPLES = 600
MIN_SHARE = 0.05
MIN_M = 150.0
GAP_M = 40.0          # a gap in the stretch shorter than this is closed (a crossover, a platform)
OWNER_SHARE = 0.3     # a register line owning this much of a section's footprint is its owner


def kind_family(k):
    return "rail" if k in ("train", "rail", "narrow_gauge", "heritage") else k


def compute(region, lines, geoms, foot, log, d_m=ALONG_M, sec_hs=None):
    """{section id: [[other section id, a, b], ...]} (a, b in SCALE units). `sec_hs`: section
    id -> high-speed or not where the build knows it from the track (the ways' highspeed
    tag), over the line flags and footprints used otherwise."""
    lats = [p[1] for g in geoms.values() for pts in g.values() for p in pts[:1]]
    if not lats:
        return {}
    lat0 = float(np.median(lats))
    kx, ky = math.cos(math.radians(lat0)) * 111320.0, 110570.0

    line_of, hs_line = {}, {}
    for l in lines:
        hs_line[l["id"]] = l.get("highspeed")
        for s in l["sections"]:
            line_of[s[3]] = l
    # High-speed or not per section (None: unknown).
    hs = {}
    for l in lines:
        own = l.get("highspeed")
        for s in l["sections"]:
            g = s[3]
            if own is not None:
                hs[g] = bool(own)
                continue
            w = defaultdict(float)
            for e in foot.get(g, ()):
                t = e[0]
                tl = line_of.get(t)
                if tl is None or tl.get("highspeed") is None:
                    continue
                w[bool(tl["highspeed"])] += abs(e[4] - e[3])
            hs[g] = (w[True] >= w[False]) if w else None

    t0 = time.time()
    if sec_hs:
        hs.update(sec_hs)
    # The register lines owning a section's track: itself for a register section that is its
    # own footprint, else the register lines under at least OWNER_SHARE of its footprint.
    # Two sections on DIFFERENT register lines are different tracks by the register's own
    # word, however close (Sanyo Electric beside JR's Sanyo Line, Nishitetsu beside JR
    # Kagoshima, Hankyu beside JR Kobe, the Gyeongbu high-speed line beside the old one), so
    # they never pair here; their shared ways, if any, already credit through footprints.
    reg_owner = {}
    for l in lines:
        for s in l["sections"]:
            g = s[3]
            f = foot.get(g)
            if not f:
                reg_owner[g] = frozenset([l["id"]]) if l.get("src") != "osm" else frozenset()
                continue
            w = defaultdict(float)
            for e in f:
                tl = line_of.get(e[0])
                if tl is not None and tl.get("src") != "osm":
                    w[tl["id"]] += abs(e[4] - e[3])
            reg_owner[g] = frozenset(k for k, v in w.items() if v >= OWNER_SHARE)

    gids, geos, fams, hss, lens, owns = [], [], [], [], [], []
    fam_code = {}
    for l in lines:
        g = geoms.get(l["id"], {})
        fam = fam_code.setdefault(kind_family(l.get("kind")), len(fam_code))
        for a, b, km, gid in (s[:4] for s in l["sections"]):
            if a == b:
                continue
            pts = g.get(f"{a}|{b}") or g.get(f"{b}|{a}")
            if not pts or len(pts) < 2:
                continue
            arr = np.asarray(pts, dtype=np.float64)[:, :2]
            xy = np.column_stack([arr[:, 0] * kx, arr[:, 1] * ky])
            ls = LineString(xy)
            if not (ls.length > 0):
                continue
            h = hs.get(gid)
            gids.append(gid); geos.append(ls); fams.append(fam)
            hss.append(-1 if h is None else int(h)); lens.append(ls.length)
            owns.append(reg_owner.get(gid, frozenset()))
    if not geos:
        return {}
    fams, hss, lens = np.array(fams), np.array(hss), np.array(lens)
    geos_a = shapely.simplify(np.array(geos, dtype=object), 2.0)

    # Every section as its segments, in one tree.
    coords, gi = shapely.get_coordinates(geos_a, return_index=True)
    same = gi[1:] == gi[:-1]
    seg_owner = gi[:-1][same]
    segs = shapely.linestrings(np.stack([coords[:-1][same], coords[1:][same]], axis=1))
    tree = STRtree(segs)

    # Samples along every section: every STEP_M, at most MAX_SAMPLES.
    ns = np.clip((lens / STEP_M + 1).astype(int), 3, MAX_SAMPLES)
    owner = np.repeat(np.arange(len(geos)), ns)
    fr = np.concatenate([np.linspace(0.0, 1.0, n) for n in ns])
    pts = shapely.line_interpolate_point(np.array(geos, dtype=object)[owner], fr, normalized=True)
    log(f"along: {len(geos)} sections, {len(segs)} segments, {len(pts)} samples")

    out = defaultdict(list)
    n_pairs = 0
    CH = 200000
    pairs_all = []
    for c0 in range(0, len(pts), CH):
        pi, si = tree.query(pts[c0:c0 + CH], predicate="dwithin", distance=d_m)
        pi = pi + c0
        S, T = owner[pi], seg_owner[si]
        ok = (S != T) & (fams[S] == fams[T]) & ~((hss[S] >= 0) & (hss[T] >= 0) & (hss[S] != hss[T]))
        pairs_all.append(np.unique(np.stack([pi[ok], T[ok]], axis=1), axis=0))
    P = np.concatenate(pairs_all) if pairs_all else np.zeros((0, 2), dtype=np.int64)
    if not len(P):
        return {}
    P = np.unique(P, axis=0)
    S = owner[P[:, 0]]
    order = np.lexsort((fr[P[:, 0]], P[:, 1], S))
    P, S = P[order], S[order]
    F = fr[P[:, 0]]
    T = P[:, 1]
    # group boundaries by (S, T)
    brk = np.flatnonzero((S[1:] != S[:-1]) | (T[1:] != T[:-1])) + 1
    starts = np.concatenate([[0], brk])
    stops = np.concatenate([brk, [len(S)]])
    n_reg = 0
    for g0, g1 in zip(starts, stops):
        i, j = int(S[g0]), int(T[g0])
        if owns[i] and owns[j] and not (owns[i] & owns[j]):
            n_reg += 1
            continue
        f = F[g0:g1]
        step = 1.0 / (ns[i] - 1)
        gap = max(GAP_M / lens[i], 1.5 * step)
        cut = np.flatnonzero(np.diff(f) > gap) + 1
        for a_i, b_i in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(f)]])):
            a, b = f[a_i], f[b_i - 1]
            share = b - a
            if share < MIN_SHARE and share * lens[i] < MIN_M:
                continue
            out[gids[i]].append([gids[j], int(round(a * SCALE)), int(round(b * SCALE))])
            n_pairs += 1
    log(f"along: {len(out)} of {len(geos)} sections have track within {d_m:.0f} m of another "
        f"section of the same kind ({n_pairs} stretches; {n_reg} pairs on two different register "
        f"lines left apart), {time.time() - t0:.0f} s")
    return dict(out)


def write(out_dir, region, along, d_m=ALONG_M):
    with open(Path(out_dir) / "along.json", "w", encoding="utf-8") as fh:
        json.dump({"region": region, "scale": SCALE, "d": d_m,
                   "along": {str(g): r for g, r in sorted(along.items())}},
                  fh, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--d", type=float, default=ALONG_M)
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

    src = ROOT / "dist" / "data" / args.region
    lines = json.loads((src / "lines.json").read_text(encoding="utf-8"))["lines"]
    geoms = {}
    for l in lines:
        p = src / "geom" / f"{l['id']}.json"
        if p.exists():
            geoms[l["id"]] = json.loads(p.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    import ownership
    foot = ownership.read(src / "foot.json") if (src / "foot.json").exists() else {}
    along = compute(args.region, lines, geoms, foot, log, args.d)
    out = Path(args.out) if args.out else src
    out.mkdir(parents=True, exist_ok=True)
    write(out, args.region, along, args.d)
    log(f"wrote {out / 'along.json'} ({(out / 'along.json').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
