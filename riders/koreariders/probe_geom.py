# -*- coding: utf-8 -*-
"""Is a route relation's way geometry contiguous enough to measure km along?

If it is, stations can be snapped to it and ordered by distance-along, which is
the only thing standing between the yearbook's alphabetical station lists and a
per-segment profile.

    python probe_geom.py "전라선 무궁화호"
"""

import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = os.path.join(HERE, "data", "osm_routes.json")


def haversine(a, b):
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def chain(ways, tol_km=0.05):
    """Greedily string way geometries end-to-end, flipping as needed."""
    pool = [[(p["lat"], p["lon"]) for p in w["geometry"]] for w in ways
            if w.get("geometry")]
    if not pool:
        return [], []
    line = pool.pop(0)
    breaks = []
    while pool:
        best, bd, bflip, bend = None, 1e9, False, True
        for i, w in enumerate(pool):
            for flip in (False, True):
                ww = w[::-1] if flip else w
                d_tail = haversine(line[-1], ww[0])
                if d_tail < bd:
                    best, bd, bflip, bend = i, d_tail, flip, True
                d_head = haversine(line[0], ww[-1])
                if d_head < bd:
                    best, bd, bflip, bend = i, d_head, flip, False
        w = pool.pop(best)
        ww = w[::-1] if bflip else w
        if bd > tol_km:
            breaks.append(bd)
        line = line + ww if bend else ww + line
    return line, breaks


def length(line):
    return sum(haversine(line[i], line[i + 1]) for i in range(len(line) - 1))


def main():
    needle = sys.argv[1] if len(sys.argv) > 1 else "전라선 무궁화호"
    with io.open(ROUTES, encoding="utf-8") as f:
        data = json.load(f)
    for rel in data["elements"]:
        name = rel.get("tags", {}).get("name", "")
        if needle not in name:
            continue
        ways = [m for m in rel["members"] if m["type"] == "way"]
        line, breaks = chain(ways)
        print("\n=== %s" % name)
        print("    %d ways, %d points, %.1f km" % (len(ways), len(line), length(line)))
        print("    %d joins over 50 m; worst %s"
              % (len(breaks),
                 ", ".join("%.2f km" % b for b in sorted(breaks, reverse=True)[:5])
                 or "none"))
        print("    ends: %.4f,%.4f -> %.4f,%.4f"
              % (line[0][0], line[0][1], line[-1][0], line[-1][1]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
