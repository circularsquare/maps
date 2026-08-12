"""
Straighten HydroRIVERS basins into schematic top-down diagrams.

    y = DIST_DN_KM        exact along-channel distance to terminus; terminus at 0
    x = lateral offset from a W-km-smoothed version of the real path, x GAIN

So continental-scale routing is removed (the river points straight down) while
bends shorter than W survive at true amplitude. W is the one aesthetic knob:
    W -> 0    dead straight line
    W -> inf  the real river, merely rotated

Tributaries hang off the stem at their true confluence distance, on their real
bank (left/right), packed outward so subtrees don't collide.

    python build.py [--smooth 250] [--gain 1.0] [--min-dis 5]
"""
import argparse, json, math, sys, collections
import numpy as np
import pandas as pd

CACHE = 'reaches.pkl'
OUT = 'basins.json'


# ---------------------------------------------------------------- geometry ---

def to_km(coords, lat0):
    """lon/lat degrees -> local km, equirectangular about lat0. Geography is
    getting destroyed downstream of here anyway; we only need locally-correct
    distances so the smoothing window means something."""
    xy = np.asarray(coords, dtype=float)
    x = xy[:, 0] * 111.32 * math.cos(math.radians(lat0))
    y = xy[:, 1] * 110.57
    return np.column_stack([x, y])


def chain(geoms):
    """Concatenate reach LineStrings (given headwater->outlet) into one polyline,
    flipping each so consecutive reaches actually join."""
    parts = [np.asarray(g.coords, dtype=float) for g in geoms]
    if len(parts) == 1:
        return parts[0]
    # orient the first against the second
    a, b = parts[0], parts[1]
    if min(np.hypot(*(a[0] - b[0])), np.hypot(*(a[0] - b[-1]))) < \
       min(np.hypot(*(a[-1] - b[0])), np.hypot(*(a[-1] - b[-1]))):
        parts[0] = a[::-1]
    out = [parts[0]]
    tail = parts[0][-1]
    for p in parts[1:]:
        if np.hypot(*(p[-1] - tail)) < np.hypot(*(p[0] - tail)):
            p = p[::-1]
        out.append(p[1:] if np.allclose(p[0], tail, atol=1e-9) else p)
        tail = p[-1]
    return np.vstack(out)


def resample(pts, step_km):
    """Even arc-length resampling, so the smoothing window is in real km."""
    d = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total < step_km * 2:
        return pts, s
    n = max(int(total / step_km) + 1, 2)
    su = np.linspace(0, total, n)
    return np.column_stack([np.interp(su, s, pts[:, 0]),
                            np.interp(su, s, pts[:, 1])]), su


def smooth(a, win):
    """Box-smooth with edge padding (reflect keeps the ends from collapsing)."""
    if win < 3:
        return a.copy()
    win = int(win) | 1
    pad = win // 2
    p = np.pad(a, pad, mode='reflect')
    k = np.ones(win) / win
    return np.convolve(p, k, mode='valid')


def lateral_offset(pts, win_pts):
    """Signed perpendicular distance from each point to the smoothed path.

    This is the actual straightening: the smoothed path IS the continental
    routing, and what's left over is the local character we want to keep.
    Rotation-free, so it doesn't care which way the river happens to run.
    """
    sx = smooth(pts[:, 0], win_pts)
    sy = smooth(pts[:, 1], win_pts)
    # tangent of the smoothed path
    tx = np.gradient(sx)
    ty = np.gradient(sy)
    n = np.hypot(tx, ty)
    n[n == 0] = 1.0
    tx, ty = tx / n, ty / n
    # left normal
    nx, ny = -ty, tx
    return (pts[:, 0] - sx) * nx + (pts[:, 1] - sy) * ny


# ------------------------------------------------------------------- trees ---

def decompose(children, outlet, upland):
    """Split a basin tree into stems. A stem runs headwater->outlet, always
    following the largest-upland parent; every other inflow starts a new stem.
    Returns [(stem_reach_ids, parent_stem_index, join_reach_id), ...]."""
    stems = []
    queue = [(outlet, -1, None)]
    while queue:
        start, parent, join = queue.pop()
        seq, cur = [start], start
        while True:
            kids = children.get(cur)
            if not kids:
                break
            kids = sorted(kids, key=lambda k: upland[k], reverse=True)
            for k in kids[1:]:
                queue.append((k, len(stems), cur))
            cur = kids[0]
            seq.append(cur)
        stems.append((seq[::-1], parent, join))  # headwater -> outlet
    return stems


# ------------------------------------------------------------------- build ---

def build_basin(g, main_riv, smooth_km, gain, trib_frac, min_upland,
                spread=0.6, step_km=2.0):
    sub = g[g.MAIN_RIV == main_riv]
    if len(sub) < 2:
        return None
    basin_area = float(sub.UPLAND_SKM.max())

    # Prune to tributaries worth drawing. Threshold on upstream AREA, never on
    # discharge -- discharge is what collapses on the rivers this map is about,
    # so a discharge cut would eat exactly the tails we want to show. Area only
    # grows downstream, so the survivors stay a connected tree rooted at the
    # outlet and every terminus is preserved.
    cut = max(trib_frac * basin_area, min_upland)
    sub = sub[sub.UPLAND_SKM >= cut]
    if len(sub) < 2:
        return None

    rec = sub.set_index('HYRIV_ID')
    ids = set(rec.index)

    children = collections.defaultdict(list)
    for hid, nd in zip(sub.HYRIV_ID.values, sub.NEXT_DOWN.values):
        if nd in ids:
            children[nd].append(hid)
    upland = dict(zip(sub.HYRIV_ID.values, sub.UPLAND_SKM.values))

    # outlet = the surviving reach closest to the terminus
    outlet = rec.DIST_DN_KM.idxmin()
    stems = decompose(children, outlet, upland)

    lat0 = float(np.mean([g.y for g in sub.geometry.representative_point()]))

    built = []
    built_of = {}                       # stem index -> built index
    for si, (seq, parent, join) in enumerate(stems):
        geoms = [rec.at[i, 'geometry'] for i in seq]
        pts = chain(geoms)
        km = to_km(pts, lat0)
        km, s = resample(km, step_km)
        if len(km) < 3:
            continue
        win_pts = max(3, int(smooth_km / step_km))
        lat = lateral_offset(km, win_pts) * gain

        # y from the data, not from the geometry: exact, monotone, terminus at 0
        r = rec.loc[seq]
        y_nodes = r.DIST_DN_KM.values             # per reach, headwater->outlet
        s_nodes = np.linspace(0, s[-1], len(seq)) if len(seq) > 1 else np.array([0.0])
        y = np.interp(s, s_nodes, y_nodes)

        dis = np.interp(s, s_nodes, r.DIS_AV_CMS.values)
        up = np.interp(s, s_nodes, r.UPLAND_SKM.values)

        built_of[si] = len(built)
        built.append(dict(seq=seq, parent_stem=parent, join=join,
                          lat=lat, y=y, dis=dis, up=up,
                          km=km, s=s,
                          peak=float(r.DIS_AV_CMS.max()),
                          length=float(r.DIST_DN_KM.max() - r.DIST_DN_KM.min())))

    if not built:
        return None

    # Degenerate stems get skipped above, which shifts every later index, so
    # remap parents through built_of. A stem orphaned by a dropped parent
    # becomes its own root rather than pointing at whatever now sits there.
    for b in built:
        b['parent'] = built_of.get(b['parent_stem'], -1)

    # -------- horizontal packing -------------------------------------------
    # width each subtree demands, bottom-up
    kids_of = collections.defaultdict(list)
    for i, b in enumerate(built):
        if b['parent'] >= 0:
            kids_of[b['parent']].append(i)

    # Demand is how much horizontal room a subtree needs. Children sit on both
    # banks, so a subtree only needs room for its busier side, not the sum of
    # both -- summing is what made this compound out to absurd widths.
    demand = [0.0] * len(built)
    for i in range(len(built) - 1, -1, -1):
        own = float(np.abs(built[i]['lat']).max()) if len(built[i]['lat']) else 0.0
        kd = sorted((demand[k] for k in kids_of[i]), reverse=True)
        # busiest child in full, the rest heavily discounted: they interleave
        # along the stem rather than stacking at one height
        nest = kd[0] + 0.15 * sum(kd[1:]) if kd else 0.0
        demand[i] = own + nest + 6.0

    # Two horizontal quantities, kept apart all the way to the renderer:
    #   L  local bend, REAL km -- must stay locked to the vertical scale or the
    #      shapes that make a river recognisable get smeared sideways
    #   O  layout offset, INVENTED units -- pure tidy-tree spacing, free to be
    #      stretched to fill whatever cell it's drawn in
    # Merging them into one x was what forced a bad tradeoff between honest
    # bends and a readable tree.
    order = [i for i, b in enumerate(built) if b['parent'] < 0]
    for i in order:
        built[i]['O'] = np.zeros_like(built[i]['y'])
        built[i]['L'] = built[i]['lat'].copy()
        built[i]['join_y'] = float(built[i]['y'][-1])
        built[i]['join_O'] = 0.0
        built[i]['join_L'] = float(built[i]['lat'][-1])
    while order:
        i = order.pop()
        b = built[i]
        left = right = 0.0
        for k in sorted(kids_of[i], key=lambda k: -built[k]['peak']):
            kb = built[k]
            jy = float(rec.at[kb['join'], 'DIST_DN_KM'])
            idx = int(np.argmin(np.abs(b['y'] - jy)))
            # Real bank, in real geography: cross the stem's local downstream
            # tangent with the vector out to the tributary's body. Keeps a bit
            # of genuine information in an otherwise invented axis.
            j0 = max(idx - 3, 0)
            j1 = min(idx + 3, len(b['km']) - 1)
            tx, ty = b['km'][j0] - b['km'][j1]          # points downstream
            cx, cy = kb['km'].mean(axis=0) - b['km'][idx]
            side = -1.0 if (tx * cy - ty * cx) > 0 else 1.0

            if side < 0:
                left += demand[k] * spread
                sideoff = -left
            else:
                right += demand[k] * spread
                sideoff = right

            # Splay: no offset at the mouth, easing to the full side offset over
            # the lower third. Tributaries grow away from their confluence
            # instead of jumping sideways and being stitched back.
            rise = max(60.0, 0.35 * max(kb['y'][0] - jy, 1.0))
            t = np.clip((kb['y'] - jy) / rise, 0.0, 1.0)
            ramp = t * t * (3 - 2 * t)                       # smoothstep

            kb['O'] = b['O'][idx] + sideoff * ramp
            # slide the bend profile so the mouth lands on the parent's bank
            kb['L'] = kb['lat'] - kb['lat'][-1] + b['L'][idx]
            kb['join_y'] = jy
            kb['join_O'] = float(b['O'][idx])
            kb['join_L'] = float(b['L'][idx])
            order.append(k)

    for b in built:                       # unreachable stems, if any
        if 'O' not in b:
            b['O'] = np.zeros_like(b['y'])
            b['L'] = b['lat'].copy()
            b['join_y'] = float(b['y'][-1])
            b['join_O'] = 0.0
            b['join_L'] = float(b['lat'][-1])

    # Real projected shape, centred on the basin, kept alongside the straightened
    # coordinates so the viewer can show either without a rebuild.
    ax = np.concatenate([b['km'][:, 0] for b in built])
    ay = np.concatenate([b['km'][:, 1] for b in built])
    gcx, gcy = float(ax.mean()), float(ay.mean())

    term = rec.loc[outlet]
    tp = term.geometry.coords[-1] if term.geometry.coords else (0, 0)
    return dict(
        main_riv=int(main_riv),
        lon=round(float(tp[0]), 3), lat=round(float(tp[1]), 3),
        endorheic=int(term.ENDORHEIC),
        upland=float(term.UPLAND_SKM),
        length_km=float(rec.DIST_DN_KM.max()),
        peak_cms=float(rec.DIS_AV_CMS.max()),
        term_cms=float(term.DIS_AV_CMS),
        stems=[dict(
            x=np.round(b['L'], 2).tolist(),          # real km, locked to y
            ox=np.round(b['O'], 1).tolist(),         # invented layout offset
            y=np.round(b['y'], 2).tolist(),
            gx=np.round(b['km'][:, 0] - gcx, 1).tolist(),   # true shape, km east
            gy=np.round(b['km'][:, 1] - gcy, 1).tolist(),   # true shape, km north
            q=np.round(b['dis'], 3).tolist(),
            parent=b['parent'],
            join_x=round(b['join_L'], 2),
            join_ox=round(b['join_O'], 1),
            join_y=round(b['join_y'], 2),
        ) for b in built],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smooth', type=float, default=250.0,
                    help='straightening window in km (the aesthetic knob)')
    ap.add_argument('--gain', type=float, default=1.0,
                    help='multiply lateral wiggle')
    ap.add_argument('--trib-frac', type=float, default=0.004,
                    help='draw tributaries draining at least this fraction of '
                         'the basin (0.004 = 0.4%%)')
    ap.add_argument('--min-upland', type=float, default=200.0,
                    help='absolute floor on upstream area (km2)')
    ap.add_argument('--spread', type=float, default=0.6,
                    help='how far tributaries fan out sideways')
    ap.add_argument('--basins', type=int, default=24)
    a = ap.parse_args()

    g = pd.read_pickle(CACHE)
    print(f'{len(g):,} reaches', flush=True)

    outlets = g[g.NEXT_DOWN == 0].sort_values('UPLAND_SKM', ascending=False)
    targets = outlets.head(a.basins).MAIN_RIV.tolist()

    out = []
    for n, mr in enumerate(targets, 1):
        b = build_basin(g, mr, a.smooth, a.gain, a.trib_frac, a.min_upland,
                         a.spread)
        if b:
            out.append(b)
            print(f'  [{n}/{len(targets)}] {mr}: {len(b["stems"])} stems, '
                  f'{b["length_km"]:,.0f} km, peak {b["peak_cms"]:,.0f} -> '
                  f'term {b["term_cms"]:,.0f} cms'
                  f'{"  ENDORHEIC" if b["endorheic"] else ""}', flush=True)

    out.sort(key=lambda b: -b['upland'])
    with open(OUT, 'w') as f:
        json.dump(dict(smooth_km=a.smooth, gain=a.gain, basins=out), f)
    print(f'wrote {OUT} ({len(out)} basins)')


if __name__ == '__main__':
    sys.exit(main())
