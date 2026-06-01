"""
Audit which segment groups will have their displayed from/to "rescued" at
runtime by index.html's geomFromTo logic — i.e. where the upstream
gtfs-gis.jp 始点駅/終点駅 disagrees with the station at the geometric endpoint.

Output is a single sorted table: published label → derived label, with the
sibling count and a marker on whichever side got overridden. Useful for
identifying upstream mislabels worth fixing in build.py (rather than relying
on the runtime safety net).

Run: python audit_fromto.py
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SEGS = ROOT / 'data' / 'segments.geojson'
STNS = ROOT / 'data' / 'stations.geojson'

# Same thresholds as index.html geomFromTo.
KEEP_R = 0.05 * 0.05    # ~5.5 km — keep published name if same-named station is this close
FB     = 0.012 * 0.012  # ~1.3 km — fallback search radius
NEAR   = 0.012 * 0.012  # orientation check radius


def flatten_coords(geom):
    if geom['type'] == 'MultiLineString':
        return [c for line in geom['coordinates'] for c in line]
    return geom['coordinates']


def effective_endpoints(siblings, from_name, stns_by_name):
    cs = []
    for f in siblings:
        c = flatten_coords(f['geometry'])
        if len(c) >= 2:
            cs.append(c[0])
            cs.append(c[-1])
    if len(cs) < 2:
        return None
    best_d2 = -1.0
    ai, bi = 0, 1
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            dx = cs[i][0] - cs[j][0]
            dy = cs[i][1] - cs[j][1]
            d2 = dx * dx + dy * dy
            if d2 > best_d2:
                best_d2 = d2
                ai, bi = i, j
    A, B = cs[ai], cs[bi]
    # orient: if a same-named station sits at B but not A, flip
    at_a = same_name_within(from_name, A[0], A[1], NEAR, stns_by_name)
    at_b = same_name_within(from_name, B[0], B[1], NEAR, stns_by_name)
    if at_b and not at_a:
        A, B = B, A
    return A, B


def same_name_within(name, sx, sy, max_d2, stns_by_name):
    for f in stns_by_name.get(name, ()):
        cx, cy = f['geometry']['coordinates']
        d2 = (cx - sx) ** 2 + (cy - sy) ** 2
        if d2 < max_d2:
            return f
    return None


def nearest_same_name(name, sx, sy, stns_by_name):
    best, best_d2 = None, float('inf')
    for f in stns_by_name.get(name, ()):
        cx, cy = f['geometry']['coordinates']
        d2 = (cx - sx) ** 2 + (cy - sy) ** 2
        if d2 < best_d2:
            best, best_d2 = f, d2
    return best, best_d2


def nearest_to_point(sx, sy, max_d2, pref_op, stns):
    best_any, best_any_d2 = None, float('inf')
    best_pref, best_pref_d2 = None, float('inf')
    for f in stns:
        cx, cy = f['geometry']['coordinates']
        d2 = (cx - sx) ** 2 + (cy - sy) ** 2
        if d2 >= max_d2:
            continue
        if d2 < best_any_d2:
            best_any, best_any_d2 = f, d2
        if pref_op and f['properties']['op'] == pref_op and d2 < best_pref_d2:
            best_pref, best_pref_d2 = f, d2
    return best_pref or best_any


def geom_from_to(p, siblings, stns, stns_by_name):
    from_, from_op = p['from'], p['op']
    to, to_op = p['to'], p['op']
    from_rescued = to_rescued = False

    ends = effective_endpoints(siblings, p['from'], stns_by_name)
    if ends is None:
        return from_, from_op, from_rescued, to, to_op, to_rescued
    A, B = ends

    fns, fns_d2 = nearest_same_name(p['from'], A[0], A[1], stns_by_name)
    if fns and fns_d2 < KEEP_R:
        from_op = fns['properties']['op']
    else:
        ns = nearest_to_point(A[0], A[1], FB, p['op'], stns)
        if ns:
            from_ = ns['properties']['station']
            from_op = ns['properties']['op']
            from_rescued = True

    tns, tns_d2 = nearest_same_name(p['to'], B[0], B[1], stns_by_name)
    if tns and tns_d2 < KEEP_R:
        to_op = tns['properties']['op']
    else:
        ns = nearest_to_point(B[0], B[1], FB, p['op'], stns)
        if ns:
            to = ns['properties']['station']
            to_op = ns['properties']['op']
            to_rescued = True

    return from_, from_op, from_rescued, to, to_op, to_rescued


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    segs = json.loads(SEGS.read_text(encoding='utf-8'))['features']
    stns = json.loads(STNS.read_text(encoding='utf-8'))['features']
    stns_by_name = defaultdict(list)
    for f in stns:
        stns_by_name[f['properties']['station']].append(f)

    by_key = defaultdict(list)
    for f in segs:
        p = f['properties']
        by_key[(p['op'], p['line'], p['from'], p['to'])].append(f)

    rows = []
    for key, group in by_key.items():
        op, line, fr, to = key
        p = group[0]['properties']
        f_d, _, f_r, t_d, _, t_r = geom_from_to(p, group, stns, stns_by_name)
        if not (f_r or t_r):
            continue
        rows.append((op, line, fr, to, f_d, t_d, f_r, t_r, len(group)))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    print(f"Audited {len(by_key)} (op, line, from, to) groups.")
    print(f"Rescued (published label disagrees with geometric endpoint): {len(rows)}\n")
    print(f"  {'operator':<14}  {'line':<22}  {'published':<24}  {'derived':<24}  ×n")
    print(f"  {'-'*14}  {'-'*22}  {'-'*24}  {'-'*24}  --")
    for op, line, fr, to, f_d, t_d, f_r, t_r, n in rows:
        published = f"{fr} → {to}"
        f_mark = '*' if f_r else ''
        t_mark = '*' if t_r else ''
        derived = f"{f_d}{f_mark} → {t_d}{t_mark}"
        print(f"  {op:<14}  {line:<22}  {published:<24}  {derived:<24}  ×{n}")


if __name__ == '__main__':
    main()
