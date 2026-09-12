# -*- coding: utf-8 -*-
"""Turn the OSM city pull into a shape for each drawn segment.

The five intracity models draw straight hops between published station
coordinates, which reads as a diagram: Busan line 2 curves a long way round the
bay and the map put a chord across it. `fetch_city_track.py` has the metals;
this fits them to the stations each builder already has.

The method is `build.py`'s, scaled down. Assemble one polyline per line, project
each station onto it, sort by distance along, slice between neighbours. What is
different is the assembly: the intercity net needed a national graph and a
shortest path because its lines interlock, while a metro line is a single
corridor and its ways only have to be chained end to end.

Route relations are preferred where one covers the line, since a relation is
already the operator's own idea of the route and its members come ordered. The
named ways are the fallback and also the repair: a relation that arrives in two
pieces gets the gap closed from the way pool.

    python city_track.py            # report what matches what, draw nothing
"""

import io
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "osm_city_track.json")

# How far a station may sit from the metals and still be considered on them.
# Station coordinates are the national workbook's, the track is OSM's, and a
# platform is not a centreline -- but 400 m is far more than that gap and far
# less than the distance to a neighbouring line.
SNAP_M = 400.0

# Chaining tolerance: two way ends this close are the same place. OSM splits a
# way at every tagging change, so the pieces meet exactly; the slack is for
# rounding to six decimals.
JOIN_M = 25.0


def haversine(a, b):
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def _norm(name):
    """'부산 도시철도 1호선' and '부산 1호선' are the same line.

    OSM spells a city line four ways in the same city -- 대구 2호선, 대구 도시철도
    2호선, and the odd bare 1호선 -- so the key is the digits and nothing else,
    with the city supplied by which pull the element came from.
    """
    s = re.sub(r"\s+", "", name or "")
    s = s.replace("도시철도", "").replace("광역시", "")
    m = re.search(r"(\d+)호선", s)
    if m:
        return m.group(1)
    if "경전철" in s or "김해" in s:
        return "lrt"
    return None


def load():
    with io.open(SRC, encoding="utf-8") as f:
        return json.load(f)


def _chain(parts):
    """Join way geometries end to end; return the pieces, longest first.

    Greedy: take a piece, keep attaching whatever way starts or ends where it
    currently does. A line usually comes out in one piece; a branch or a gap in
    the mapping leaves more, and the caller decides what to do about that.
    """
    pool = [list(p) for p in parts if len(p) >= 2]
    out = []
    while pool:
        cur = pool.pop()
        grew = True
        while grew:
            grew = False
            for i, w in enumerate(pool):
                for cand in (w, w[::-1]):
                    if haversine(_ll(cur[-1]), _ll(cand[0])) * 1000 <= JOIN_M:
                        cur = cur + cand[1:]
                        pool.pop(i)
                        grew = True
                        break
                    if haversine(_ll(cur[0]), _ll(cand[-1])) * 1000 <= JOIN_M:
                        cur = cand[:-1] + cur
                        pool.pop(i)
                        grew = True
                        break
                if grew:
                    break
        out.append(cur)
    out.sort(key=lambda c: -_length(c))
    return out


def _ll(p):
    """OSM coordinates are [lon, lat]; the distance helpers want (lat, lon)."""
    return (p[1], p[0])


def _length(coords):
    return sum(haversine(_ll(a), _ll(b)) for a, b in zip(coords, coords[1:]))


def candidates(city_elements):
    """{line key: [polyline, ...]} for one city, best first is *not* implied.

    Several candidates per line, deliberately. A metro is double track and OSM
    maps each track as its own way, so chaining the whole name pool end to end
    produces one out-and-back polyline of twice the real length -- Busan line 1
    came out 79.8 km against a railway of 40.5. Route relations are the way out
    because the operator publishes one per direction, and either of them is the
    line: `부산 도시철도 1호선: 노포 → 다대포해수욕장` and its reverse.

    So each relation is chained on its own and offered separately, with the
    merged ways offered last as a fallback for a line no relation covers.
    Choosing between them needs the stations, which live in the builders, so
    that is `pick_for` below and not here.
    """
    rels, ways = {}, {}
    for el in city_elements:
        key = _norm(el.get("name"))
        if key is None:
            continue
        if el["type"] == "relation":
            # Each relation is its own candidate. Grouping by name looked
            # right and was not: Busan line 2 and Daegu line 2 each have two
            # relations under one name, one per direction, and merging them
            # rebuilt exactly the out-and-back this is here to avoid -- 90.7 km
            # against a railway of 45.2.
            rels.setdefault(key, []).append(el.get("parts", []))
        else:
            ways.setdefault(key, []).append(el.get("geometry", []))

    out = {}
    for key in set(rels) | set(ways):
        cands = []
        for parts in (rels.get(key) or []):
            pieces = _chain(parts)
            if pieces and _length(pieces[0]) > 1.0:
                cands.append(pieces[0])
        if ways.get(key):
            pieces = _chain(ways[key])
            if pieces and _length(pieces[0]) > 1.0:
                cands.append(pieces[0])
        if cands:
            out[key] = cands
    return out


def pick_for(cands, stations):
    """The shortest candidate that passes within SNAP_M of every station.

    Shortest is the point: an out-and-back covers the stations just as well as
    one direction does and is twice as long, so length is what separates them.
    Requiring full coverage is what stops a half-mapped relation winning on
    shortness alone.

    Returns (polyline, cum, worst station offset in metres) or None.
    """
    best = None
    for poly in sorted(cands, key=_length):
        cum = chainage(poly)
        worst = 0.0
        ok = True
        for pt in stations:
            _, off = project(pt, poly, cum)
            worst = max(worst, off)
            if off * 1000.0 > SNAP_M:
                ok = False
                break
        if ok:
            best = (poly, cum, worst * 1000.0)
            break
    return best


def chainage(poly):
    cum = [0.0]
    for a, b in zip(poly, poly[1:]):
        cum.append(cum[-1] + haversine(_ll(a), _ll(b)))
    return cum


def project(pt, poly, cum):
    """Distance along `poly` of the point on it nearest `pt`, and how far off.

    Vertex resolution only. OSM traces a metro at a few metres a node, so the
    nearest vertex is within a rounding error of the true foot of the
    perpendicular, and interpolating buys nothing here.
    """
    best_i, best_d = 0, None
    for i, q in enumerate(poly):
        d = haversine(_ll(q), pt)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return cum[best_i], best_d


def slice_between(poly, cum, a_km, b_km):
    """The stretch of `poly` between two chainages, ends included."""
    lo, hi = (a_km, b_km) if a_km <= b_km else (b_km, a_km)
    pts = [p for p, c in zip(poly, cum) if lo - 1e-9 <= c <= hi + 1e-9]
    if len(pts) < 2:
        return None
    return pts if a_km <= b_km else pts[::-1]


_CACHE = {}


def shapes_for(city, line_key, stations):
    """Per-segment track shapes for one line, or None if it cannot be fitted.

    `stations` is the builder's own drawn order as (lon, lat). Returns a list
    one shorter than it, each entry a [lon, lat] coordinate list, plus the
    worst station offset in metres so the caller can report how well it sat.

    Returns (shapes, worst_m) or None. A None means the fallback stands and the
    line keeps its straight hops -- better than a corridor the stations do not
    lie on.
    """
    if not _CACHE:
        _CACHE.update(load())
    els = _CACHE.get(city)
    if not els:
        return None
    cands = candidates(els).get(line_key)
    if not cands:
        return None
    pts = [(lat, lon) for lon, lat in stations]
    got = pick_for(cands, pts)
    if got is None:
        return None
    poly, cum, worst = got
    marks = [project(pt, poly, cum)[0] for pt in pts]
    # The stations have to run monotonically along the corridor or the slices
    # will double back; a line whose relation loops (Gwangju's line 2 is a
    # ring) fails this and keeps its straight hops.
    if not (all(x <= y for x, y in zip(marks, marks[1:]))
            or all(x >= y for x, y in zip(marks, marks[1:]))):
        return None
    shapes = []
    for a, b in zip(marks, marks[1:]):
        seg = slice_between(poly, cum, a, b)
        if seg is None or len(seg) < 2:
            return None
        shapes.append(seg)
    return shapes, worst


# Drawn line name -> which city pull holds it and under what key. The pulls
# overlap at the edges (Busan-Gimhae's box catches Busan lines 2 and 3), so the
# city is named rather than searched.
LINE_SOURCE = {
    "부산 1호선": ("busan", "1"), "부산 2호선": ("busan", "2"),
    "부산 3호선": ("busan", "3"), "부산 4호선": ("busan", "4"),
    "대구 1호선": ("daegu", "1"), "대구 2호선": ("daegu", "2"),
    "대구 3호선": ("daegu", "3"),
    "부산김해경전철": ("busan_gimhae", "lrt"),
    "대전 1호선": ("daejeon", "1"), "광주 1호선": ("gwangju", "1"),
}


def fit_edges(by_line):
    """{edge id: shape} for whatever can be fitted, and a per-line report.

    `by_line` is {drawn line name: [(edge id, a coord, b coord), ...]} in drawn
    order, coords as [lon, lat]. A line that cannot be fitted simply gets no
    entries and keeps its straight hops, which is the honest fallback: a
    corridor the stations do not sit on would be worse than a chord.
    """
    shapes, report = {}, {}
    for line, edges in by_line.items():
        src = LINE_SOURCE.get(line)
        if not src or not edges:
            report[line] = {"fitted": False, "reason": "no OSM source named"}
            continue
        stations = [e[1] for e in edges] + [edges[-1][2]]
        got = shapes_for(src[0], src[1], stations)
        if got is None:
            report[line] = {"fitted": False, "reason": "no candidate covers "
                                                       "every station in order"}
            continue
        segs, worst = got
        for (eid, _, _), sh in zip(edges, segs):
            shapes[eid] = sh
        report[line] = {"fitted": True, "worst_station_offset_m": round(worst),
                        "track_km": round(sum(_length(sh) for sh in segs), 3),
                        "source": "OSM %s %s" % src}
    return shapes, report


def main():
    data = load()
    print("%-14s %-6s %7s  %s" % ("city", "line", "cands", "lengths km"))
    print("-" * 62)
    for city, els in sorted(data.items()):
        cands = candidates(els)
        for key in sorted(cands):
            lens = sorted(_length(c) for c in cands[key])
            print("%-14s %-6s %7d  %s"
                  % (city, key, len(lens), " ".join("%.1f" % v for v in lens)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
