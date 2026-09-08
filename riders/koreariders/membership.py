# -*- coding: utf-8 -*-
"""Which lines physically serve each station.

The roster in `8. 시설` sheet 2 gives every station exactly one *home* line, which
is enough to spot a stranger snapping onto a corridor but useless for the
network solve: it says 익산 is 호남선's and will not admit that 전라선 and 장항선
also end there. Membership has to come from the track.

The test is physical -- a station belongs to line L if track named L runs within
`NEAR_M` of it. That distinguishes the two cases the roster confuses:

    익산      호남선, 전라선 and 장항선 metals all reach the platforms   -> all three
    천안아산   경부고속선 metals only; 아산 is a separate station 100 m off -> 경부고속선

    python membership.py           # print the shared stations it finds
"""

import collections
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
RAILWAYS = os.path.join(D, "osm_railways.json")
STATIONS = os.path.join(D, "osm_stations.json")

NEAR_M = 250.0
CELL = 0.02             # index cell, degrees -- about 2 km


def _key(lat, lon):
    return (int(math.floor(lat / CELL)), int(math.floor(lon / CELL)))


def build_index():
    """Grid of track segments, so a station only tests against nearby track.

    A station-by-edge scan is 2000 x 250k and far too slow in Python; bucketing
    the segments by a 2 km cell makes it a few dozen comparisons each.
    """
    with io.open(RAILWAYS, encoding="utf-8") as f:
        ways = json.load(f)["elements"]
    grid = collections.defaultdict(list)
    for w in ways:
        nm = (w.get("tags") or {}).get("name")
        if not nm:
            continue
        pts = [(p["lat"], p["lon"]) for p in w.get("geometry", [])]
        for a, b in zip(pts, pts[1:]):
            lo_la, hi_la = sorted((a[0], b[0]))
            lo_lo, hi_lo = sorted((a[1], b[1]))
            for i in range(_key(lo_la, lo_lo)[0], _key(hi_la, hi_lo)[0] + 1):
                for j in range(_key(lo_la, lo_lo)[1], _key(hi_la, hi_lo)[1] + 1):
                    grid[(i, j)].append((a, b, nm))
    return grid


def seg_dist_m(pt, a, b):
    kx = math.cos(math.radians(a[0])) * 111320.0
    ky = 110540.0
    ax, ay = a[1] * kx, a[0] * ky
    bx, by = b[1] * kx, b[0] * ky
    px, py = pt[1] * kx, pt[0] * ky
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def station_lines(grid, stations):
    """{station name: {osm line name: nearest distance in m}}."""
    out = {}
    for nm, pts in stations.items():
        best = {}
        for pt in pts:
            ci, cj = _key(pt[0], pt[1])
            for i in (ci - 1, ci, ci + 1):
                for j in (cj - 1, cj, cj + 1):
                    for a, b, line in grid.get((i, j), ()):
                        d = seg_dist_m(pt, a, b)
                        if d <= NEAR_M and d < best.get(line, 1e9):
                            best[line] = d
        if best:
            out[nm] = best
    return out


KORAIL = "한국철도공사"


def load_stations(LN=None):
    """{name: [(lat, lon), ...]} for every OSM station node.

    Pass `LN` to keep only the ones Korail actually serves. Snapping by distance
    alone drags in whatever railway happens to run alongside: the 부산 도시철도
    follows the old 경부선 alignment through 개금 and 주례, Seoul's line 4 sits
    over it at 신용산 and 삼각지, and 서울역 is a second node for 서울. Left in,
    they carry no traffic but split the corridor at places no train stops.

    The test is the operator, because nothing else separates them. A 광역전철
    station is still Korail's -- 노량진, 구로, 금천구청, all of 경춘선 -- and
    people really travel between those, so they stay even though the yearbook
    counts 일반열차 only and has no row for them. Dropped only if OSM knows the
    name, no node of that name is Korail's, and neither the 승하차 table nor any
    roster has heard of it.

    That name-wide test has a hole, and it is the one that lets a station be in
    two places. 좌천 is a Korail station on 동해선 at 35.312/129.245 **and** a
    부산교통공사 subway station 26 km away at 35.134/129.054, and because the
    name is in the 승하차 table the whole name survives -- so 경부선 snapped to
    the subway node and 동해선 to the Korail one, and `check.py` reported the two
    lines meeting 26 km apart. Worse than the drawing: `solve.py` keys junctions
    by name, so it was conserving through flow between two lines at a station
    neither shares.

    A name being in the 승하차 table says *a* station of that name has 일반열차
    traffic. It does not say every node of that name does. So the operator test
    is now per node as well as per name: **where a name has a Korail node, the
    non-Korail nodes of that name are dropped.** The yearbook's row belongs to
    the Korail one, and the other is a different railway that happens to share a
    name. Nothing that was kept before by having Korail *somewhere* under its
    name is affected, since that node is exactly the one this keeps.
    """
    with io.open(STATIONS, encoding="utf-8") as f:
        els = json.load(f)["elements"]

    ops = collections.defaultdict(set)
    for n in els:
        t = n.get("tags", {}) or {}
        if t.get("name"):
            ops[t["name"]].add(t.get("operator") or "")

    skip = set()
    if LN is not None:
        flows = LN.station_flows()
        roster = set()
        for _, sts in LN.rosters().items():
            roster |= set(sts)
        skip = {nm for nm, o in ops.items()
                if KORAIL not in o and nm not in flows and nm not in roster}

    named = collections.defaultdict(list)
    for n in els:
        t = n.get("tags", {}) or {}
        nm = t.get("name")
        if not nm or nm in skip:
            continue
        if KORAIL in ops[nm] and (t.get("operator") or "") != KORAIL:
            continue
        named[nm].append((n["lat"], n["lon"]))
    return named


def serves(LN):
    """{station: [canonical line, ...]} -- which lines actually call there.

    Proximity alone over-reports badly, because the 고속선 runs alongside the
    line it duplicates for hundreds of km: 지탄 and 좌천 are village halts that
    a KTX has never stopped at, but high-speed metals pass within 250 m of both.
    Requiring the line's own train types to show traffic at the station removes
    those, and also settles name collisions -- there are two 판교, and only the
    장항선 one has 무궁화 passengers.
    """
    grid = build_index()
    sl = station_lines(grid, load_stations(LN))
    by_type = LN.station_flows_by_type()

    wanted = collections.defaultdict(list)
    for canon, (_, _, _, osm) in LN.LINES.items():
        for o in osm:
            wanted[o].append(canon)

    out = {}
    for st, hits in sl.items():
        keep = set()
        for o in hits:
            for canon in wanted.get(o, ()):
                served = sum(by_type.get(t, {}).get(st, (0, 0, 0, 0))[i]
                             for t in LN.TYPES.get(canon, LN.ALL_TYPES)
                             for i in (0, 2))
                if served > 0:
                    keep.add(canon)
        # Where one name is two stations and both are Korail's, the operator
        # test above cannot help and lines.STATION_HOME says which one the
        # 승하차 row is. Only applied when that line is a claimant, so a stale
        # entry cannot invent membership -- it can only take it away.
        home = getattr(LN, "STATION_HOME", {}).get(st)
        if home is not None and home in keep:
            keep = {home}
        if keep:
            out[st] = sorted(keep)
    return out


def main():
    import lines as LN
    flows = LN.station_flows()
    srv = serves(LN)
    shared = {s: v for s, v in srv.items() if len(v) > 1 and s in flows}
    print("%d stations with 승하차 are called at by more than one mapped line:\n"
          % len(shared))
    for st in sorted(shared, key=lambda s: -(flows[s][0] + flows[s][2])):
        print("   %-10s %9.0f 명/년 승차   %s"
              % (st, flows[st][0] + flows[st][2], ", ".join(shared[st])))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
