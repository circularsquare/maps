# -*- coding: utf-8 -*-
"""Reconstruct per-segment passenger load for every intercity line, and write
data/segments.geojson.

The method and its validation are in README.md; prototype.py is the single-line
version with the workings printed. This is the same thing over the whole
network, driven by lines.py.

    python build.py                 # all lines, report + geojson
    python build.py --line 전라선    # one line, with its station table
"""

import argparse
import collections
import heapq
import io
import json
import math
import os
import sys

import lines as LN

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
RAILWAYS = os.path.join(D, "osm_railways.json")
STATIONS = os.path.join(D, "osm_stations.json")
OUT = os.path.join(D, "segments.geojson")

SNAP_KM = 0.30          # how close a station must be to count as on the line
PENALTY = 40.0          # cost multiplier for track belonging to another line


# ---------------------------------------------------------------- geometry


def haversine(a, b):
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def load_network():
    """One graph for the whole country: {node: [(node, km, line_name), ...]}.

    Built once and reused for every line, with the per-line cost applied at
    search time -- rebuilding it per line was most of the runtime.
    """
    with io.open(RAILWAYS, encoding="utf-8") as f:
        ways = json.load(f)["elements"]
    g = collections.defaultdict(list)
    for w in ways:
        nm = (w.get("tags") or {}).get("name")
        pts = [(p["lat"], p["lon"]) for p in w.get("geometry", [])]
        for a, b in zip(pts, pts[1:]):
            ka = (round(a[0], 6), round(a[1], 6))
            kb = (round(b[0], 6), round(b[1], 6))
            if ka == kb:
                continue
            d = haversine(a, b)
            g[ka].append((kb, d, nm))
            g[kb].append((ka, d, nm))
    return g


def shortest_path(g, src, dst, target):
    """Cheapest route from src to dst, preferring track named in `target`."""
    dist, prev, seen = {src: 0.0}, {}, set()
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, real, nm in g.get(u, ()):
            nd = d + real * (1.0 if nm in target else PENALTY)
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                prev[v] = (u, nm in target)
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, 0.0
    path, cur, foreign = [dst], dst, 0.0
    while cur != src:
        cur, mine = prev[cur]
        if not mine:
            foreign += 1
        path.append(cur)
    return path[::-1], foreign


def chainage(line):
    out = [0.0]
    for i in range(len(line) - 1):
        out.append(out[-1] + haversine(line[i], line[i + 1]))
    return out


def project(pt, line, cum):
    """Nearest point on the polyline: (distance_km, chainage_km)."""
    best = (1e9, 0.0)
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        kx = math.cos(math.radians(a[0]))
        ax, ay, bx, by = a[1] * kx, a[0], b[1] * kx, b[0]
        px, py = pt[1] * kx, pt[0]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0,
                                             ((px - ax) * dx + (py - ay) * dy) / L2))
        d = math.hypot(px - (ax + t * dx), py - (ay + t * dy)) * 111.32
        if d < best[0]:
            best = (d, cum[i] + t * (cum[i + 1] - cum[i]))
    return best


# ---------------------------------------------------------------- per line


def order_stations(spec, g, named, flows):
    """The line's stations, in running order, with km along the corridor."""
    target = set(spec["ways"])
    for end in ("first", "last"):
        if spec[end] not in named:
            return None, "no OSM station node named %s" % spec[end]

    # Anchor the search on the line's own track, so it cannot set off down a
    # crossing line at a junction station.
    own = [n for n in g if any(nm in target for _, _, nm in g[n])]
    if not own:
        return None, "no OSM track named %s" % "/".join(spec["ways"])

    # A name can carry several nodes -- 동대구 and 수서 have one per operator, and
    # they are not all on this line. Take whichever sits closest to this line's
    # own track.
    def pick(nm):
        best = min(((haversine(n, pt), n) for pt in named[nm] for n in own),
                   key=lambda x: x[0])
        return best[1]

    poly, _ = shortest_path(g, pick(spec["first"]), pick(spec["last"]), target)
    if poly is None:
        return None, "track not connected between %s and %s" % (spec["first"],
                                                                spec["last"])
    cum = chainage(poly)

    snapped = {}
    for nm, pts in named.items():
        for pt in pts:
            d, km = project(pt, poly, cum)
            if d <= SNAP_KM and (nm not in snapped or d < snapped[nm][0]):
                snapped[nm] = (d, km)

    # The corridor was routed *between* the two end stations, so they sit at its
    # ends by construction even when the nearest named track stops short of the
    # platform -- 경부고속선's mapped track begins south of 광명, 광주선's at the
    # 동송정 junction. Place them rather than requiring them to snap.
    snapped[spec["first"]] = (0.0, cum[0])
    snapped[spec["last"]] = (0.0, cum[-1])

    # Distance alone grabs whatever sits near the track -- 천안아산, a 경부고속선
    # station 100 m from 아산, would put its KTX arrivals into 장항선. A station
    # reporting traffic whose home line is another one is dropped, except at the
    # line's own two ends.
    ends = {spec["first"], spec["last"]}
    roster = spec["roster"]
    keep = [(km, nm) for nm, (d, km) in snapped.items()
            if nm in ends or not (nm in flows and roster and nm not in roster)]
    keep.sort()
    if len(keep) < 3:
        return None, "only %d stations snapped" % len(keep)

    k0 = dict((nm, km) for km, nm in keep)[spec["first"]]
    k1 = dict((nm, km) for km, nm in keep)[spec["last"]]
    if k0 > k1:
        keep = [(k0 - km, nm) for km, nm in keep][::-1]
    else:
        keep = [(km - k0, nm) for km, nm in keep]
    keep = [(km, nm) for km, nm in keep if -0.2 <= km <= abs(k1 - k0) + 0.2]

    scale = spec["length_km"] / keep[-1][0] if keep[-1][0] > 0 else 1.0
    return [(km * scale, nm) for km, nm in keep], None


def reconstruct(stops, flows, down=True, rev=False):
    """Cumulate inward from the anchor end; see README.md.

    `down` means travelling stop 0 -> stop n along the chain. That is 하행 only
    when the chain runs 기점 -> 종점; `rev` says it does not, and the 승하차
    column pair has to be swapped to match.
    """
    n = len(stops) - 1
    col = 0 if down != rev else 2
    bo = lambda nm: flows.get(nm, (0, 0, 0, 0))[col:col + 2]
    loads = [0.0] * n
    b_end, a_end = bo(stops[n][1])
    loads[n - 1] = a_end if down else b_end
    for i in range(n - 1, 0, -1):
        b, a = bo(stops[i][1])
        loads[i - 1] = (loads[i] - b + a) if down else (loads[i] + b - a)
    return loads


def run(canon, spec, g, named, flows):
    stops, err = order_stations(spec, g, named, flows)
    if err:
        return {"line": canon, "error": err}

    rev = spec.get("reversed", False)
    down = reconstruct(stops, flows, True, rev)
    up = reconstruct(stops, flows, False, rev)

    f = lambda nm: flows.get(nm, (0, 0, 0, 0))
    users = down[0] + up[-1] + sum(f(nm)[0] + f(nm)[2] for _, nm in stops[1:-1])

    # No clean terminus means the anchor read a junction's whole traffic and
    # lifted both profiles by a constant. 통과인원 shifts by 2d when they do, so
    # it pins the constant the anchor could not supply.
    shift = 0.0
    if not spec["clean_end"] and spec["passing"] > 0:
        shift = (spec["passing"] - users) / 2.0
        down = [x + shift for x in down]
        up = [x + shift for x in up]
        users = spec["passing"]

    gap = max(abs(down[i] - up[i]) / max(abs(down[i]), abs(up[i]), 1)
              for i in range(len(down)))
    pkm = sum((down[i] + up[i]) * (stops[i + 1][0] - stops[i][0])
              for i in range(len(down)))
    negative = min(min(down), min(up)) < 0

    # A verdict, so the report says which lines are usable rather than leaving
    # it to be eyeballed. A negative load is proof the cumulation is wrong --
    # trains cannot carry fewer than nobody. A wide mirror gap means the two
    # directions disagree, which the arithmetic alone cannot cause.
    #
    # The 통과인원 ratio only tests a line whose level was measured rather than
    # solved -- for a solved line it is an identity, since it is what was solved
    # for. It is also only a fair test where the line carries every train type:
    # 통과인원 counts everyone who touched the line's metals, so restricting a
    # line to conventional trains guarantees a low ratio rather than an error.
    ratio = users / spec["passing"] if spec["passing"] else 0.0
    testable = spec["clean_end"] and set(spec["types"]) == set(LN.ALL_TYPES)
    if negative:
        verdict = "broken"
    elif gap > 0.15 or (testable and not 0.6 < ratio < 1.5):
        verdict = "shaky"
    elif not spec["clean_end"]:
        verdict = "solved"
    elif not testable:
        verdict = "partial"
    else:
        verdict = "good"
    # reconstruct()'s `down` is the chain's own order, which on a reversed chain
    # is 상행 -- swap the pair back before anything is labelled 하행.
    if rev:
        down, up = up, down
    return {
        "line": canon, "stops": stops, "down": down, "up": up,
        "users": users, "passing": spec["passing"], "shift": shift,
        "mirror": gap, "length": stops[-1][0], "clean": spec["clean_end"],
        "density": pkm / stops[-1][0] / 365.0 if stops[-1][0] else 0.0,
        "negative": negative, "verdict": verdict,
        "types": "+".join(spec["types"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line")
    args = ap.parse_args()

    table, flows = LN.resolve()
    with io.open(STATIONS, encoding="utf-8") as f:
        named = collections.defaultdict(list)
        for n in json.load(f)["elements"]:
            nm = (n.get("tags", {}) or {}).get("name")
            if nm:
                named[nm].append((n["lat"], n["lon"]))
    print("loading the national track graph ...")
    g = load_network()
    print("   %d nodes\n" % len(g))

    todo = [args.line] if args.line else list(LN.LINES)
    results = []
    for canon in todo:
        spec = table[canon]
        if "error" in spec:
            print("%-11s %s" % (canon, spec["error"]))
            continue
        if spec["passing"] <= 0:
            print("%-11s no 통과인원 -- freight only, skipped" % canon)
            continue
        r = run(canon, spec, g, named, spec["flows"])
        results.append(r)
        if "error" in r:
            print("%-11s %s" % (canon, r["error"]))

    ok = [r for r in results if "error" not in r]
    order = {"good": 0, "solved": 1, "partial": 2, "shaky": 3, "broken": 4}
    print("\n%-11s %-7s %5s %7s %11s %11s %7s %9s"
          % ("line", "verdict", "stops", "km", "통과인원", "yearbook",
             "mirror", "수송밀도"))
    print("-" * 76)
    for r in sorted(ok, key=lambda x: (order[x["verdict"]], -x["density"])):
        print("%-11s %-7s %5d %7.1f %11.0f %11.0f %6.1f%% %9.0f"
              % (r["line"], r["verdict"], len(r["stops"]), r["length"],
                 r["users"], r["passing"], 100 * r["mirror"], r["density"]))
    tally = collections.Counter(r["verdict"] for r in ok)
    print("\n%s" % ", ".join("%d %s" % (n, k) for k, n in tally.most_common()))

    if args.line and ok:
        r = ok[0]
        print("\n%-13s %8s %10s %10s %10s"
              % ("segment", "km", "하행", "상행", "명/일"))
        print("-" * 54)
        for i in range(len(r["down"])):
            a, b = r["stops"][i], r["stops"][i + 1]
            print("%-13s %8.1f %10.0f %10.0f %10.0f"
                  % ((a[1] + "-" + b[1])[:13], b[0] - a[0], r["down"][i],
                     r["up"][i], (r["down"][i] + r["up"][i]) / 365.0))

    write_geojson(ok)


def write_geojson(results):
    feats = []
    for r in results:
        for i in range(len(r["down"])):
            a, b = r["stops"][i], r["stops"][i + 1]
            feats.append({
                "type": "Feature",
                "properties": {
                    "line": r["line"], "from": a[1], "to": b[1],
                    "km": round(b[0] - a[0], 3),
                    "down": round(r["down"][i]), "up": round(r["up"][i]),
                    "daily": round((r["down"][i] + r["up"][i]) / 365.0),
                    "anchored": "clean" if r["clean"] else "solved",
                    "verdict": r["verdict"],
                },
                "geometry": None,
            })
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f,
                  ensure_ascii=False)
    print("\nwrote %s (%d segments, geometry still to come)"
          % (os.path.relpath(OUT, HERE), len(feats)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
