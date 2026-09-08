# -*- coding: utf-8 -*-
"""역별 승하차 as map bubbles: data/stations.geojson.

The one number on this map that is *published* rather than reconstructed. Sheet
8 of `4. 수송(여객)` gives every station's boardings and alightings, split by
direction; summing all four columns and dividing by 365 is the whole
computation. The segment loads in segments.geojson are built out of exactly
these counts (see README), so the bubbles are the input and the lines are the
inference -- worth being able to see both at once.

Placement is the only real work. A station name can carry several OSM nodes
(동대구 and 수서 have one per operator), so each name takes the node closest to
the network that actually got drawn, which both locates it and confirms it is on
the map at all.

    python build_stations.py
"""

import collections
import io
import json
import math
import os
import sys

import lines as LN
import membership as M

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
SEGMENTS = os.path.join(D, "segments.geojson")
OUT = os.path.join(D, "stations.geojson")

CELL = 0.01             # ~1.1 km grid cell for the nearest-track lookup
FAR_KM = 3.0            # a station further than this from any drawn line is off-map

# 승하차 row name -> the name OSM and the drawn network use. Rows merge into the
# target, so a rename that left the yearbook carrying both spellings adds up
# rather than dropping one.
#
# 신경주 is the 2021 rename: OSM's 경주 node *is* 신경주, and the yearbook's own
# 경주 row is 164 passengers of residue from the old station closing that
# December. Not included: 서대구('22.3.31~), 506/일, which sits beside a plain
# 서대구 row of 2,185/일 -- a dated suffix usually marks an opening, but that
# would leave the undated row describing a station that did not yet exist, so
# what the pair means is unclear and merging them would invent a number.
ALIAS = {
    "김천구미": "김천(구미)",
    "영월": "영월역",
    "신경주": "경주",
}


def haversine(a, b):
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def load_segments():
    """The drawn network: a vertex grid, and which lines call at each station."""
    with io.open(SEGMENTS, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    grid = collections.defaultdict(list)
    calls = collections.defaultdict(dict)      # station -> {line: busiest daily}
    for ft in feats:
        p = ft["properties"]
        daily = p.get("daily", 0)
        for nm in (p["from"], p["to"]):
            cur = calls[nm].get(p["line"], 0)
            calls[nm][p["line"]] = max(cur, daily)
        g = ft.get("geometry")
        if not g:
            continue
        for lon, lat in g["coordinates"]:
            grid[(int(lat / CELL), int(lon / CELL))].append((lat, lon, p["line"]))
    return grid, calls, len(feats)


def nearest_drawn(grid, pt):
    """(km, line) of the closest drawn vertex to pt, or (inf, None)."""
    ci, cj = int(pt[0] / CELL), int(pt[1] / CELL)
    best, bestln = float("inf"), None
    r = 1
    while r <= 4:
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                for lat, lon, ln in grid.get((i, j), ()):
                    d = haversine(pt, (lat, lon))
                    if d < best:
                        best, bestln = d, ln
        if bestln is not None:
            break                              # found something in this ring
        r += 1
    return best, bestln


def main():
    raw = LN.station_flows()
    named = M.load_stations(LN)
    grid, calls, nseg = load_segments()
    print("%d segments drawn, %d stations in 역별 승하차, %d OSM station names"
          % (nseg, len(raw), len(named)))

    flows = collections.defaultdict(float)
    for st, v in raw.items():
        flows[ALIAS.get(st, st)] += sum(v)
    for a, b in sorted(ALIAS.items()):
        print("   merged %s -> %s" % (a, b))

    feats, far, unlocated = [], [], []
    for st, v in sorted(flows.items()):
        riders = v / 365.0
        if round(riders) < 1:
            # 미군기지 and 호계 report a handful of passengers a year. They draw
            # at radius zero but would still hold a finger-sized hit target over
            # whatever line runs beneath, so they are left out entirely.
            continue
        pts = named.get(st)
        if not pts:
            unlocated.append(st)
            continue
        # a name with several nodes: take whichever sits closest to drawn track
        d, ln, pt = float("inf"), None, None
        for q in pts:
            dq, lq = nearest_drawn(grid, q)
            if dq < d:
                d, ln, pt = dq, lq, q
        if d > FAR_KM:
            far.append((st, round(d, 1), round(riders)))
            continue
        # which lines call here, busiest first -- the endpoints in the drawn
        # network are exact, so prefer them over the nearest-vertex guess
        serving = sorted(calls.get(st, {}).items(), key=lambda kv: -kv[1])
        lns = [k for k, _ in serving] or ([ln] if ln else [])
        feats.append({
            "type": "Feature",
            "properties": {
                "station": st,
                "line": lns[0] if lns else "",
                "lines": lns,
                "riders": round(riders),
            },
            "geometry": {"type": "Point", "coordinates": [round(pt[1], 6),
                                                          round(pt[0], 6)]},
        })

    feats.sort(key=lambda f: f["properties"]["riders"])       # big drawn on top
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f,
                  ensure_ascii=False)

    print("\nwrote %s (%d stations)" % (os.path.relpath(OUT, HERE), len(feats)))
    if unlocated:
        print("\n%d with 승하차 but no OSM node: %s"
              % (len(unlocated), ", ".join(unlocated[:12])))
    if far:
        print("\n%d located but >%.0f km off the drawn network:" % (len(far), FAR_KM))
        for st, d, r in sorted(far, key=lambda x: -x[2])[:12]:
            print("   %-10s %5.1f km  %8d 명/일" % (st, d, r))

    print("\n%-10s %10s   %s" % ("station", "명/일", "lines"))
    print("-" * 60)
    for ft in sorted(feats, key=lambda f: -f["properties"]["riders"])[:15]:
        p = ft["properties"]
        print("%-10s %10d   %s" % (p["station"], p["riders"], " · ".join(p["lines"])))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
