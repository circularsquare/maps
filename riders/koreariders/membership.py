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


def load_stations():
    with io.open(STATIONS, encoding="utf-8") as f:
        named = collections.defaultdict(list)
        for n in json.load(f)["elements"]:
            nm = (n.get("tags", {}) or {}).get("name")
            if nm:
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
    sl = station_lines(grid, load_stations())
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
