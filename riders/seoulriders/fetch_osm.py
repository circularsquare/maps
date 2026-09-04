# -*- coding: utf-8 -*-
"""Pull station nodes and route relations from OSM for every line we draw.

Replaces the two ad-hoc Overpass pulls that produced data/osm_stations.json and
data/osm_routes.json. Those filtered relation names on 호선, which is why the
network beyond lines 1-9 has no geometry: 경의중앙, 수인분당, 신분당, 공항철도
and the light rail lines do not have 호선 in their names.

The route relations are the useful half. Each carries `stop`-role nodes in
running order *and* way members with full geometry, so one pull gives ordered
stations and curved track together.

    python fetch_osm.py --survey    # list matching relation names, fetch nothing
    python fetch_osm.py             # write data/osm_routes.json, osm_stations.json
"""

import argparse
import collections
import io
import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Wide enough for every terminus we now carry: 신창 in the south, 연천 in the
# north, 인천공항 in the west, 춘천 and 여주 in the east.
BBOX = "36.60,126.20,38.30,127.95"

SURVEY_Q = """
[out:json][timeout:600];
(
  relation["type"="route"]["route"~"^(subway|light_rail|train|monorail)$"](%s);
);
out tags;
""" % BBOX

ROUTES_Q = """
[out:json][timeout:900];
(
  relation["type"="route"]["route"~"^(subway|light_rail|monorail)$"](%s);
  relation["type"="route"]["route"="train"]["network"~"수도권|광역|Seoul|Korail|한국철도"](%s);
  relation["type"="route"]["route"="train"]["name"~"수도권 전철|경의|중앙선|수인|분당|경춘|경강|서해|공항철도"](%s);
);
out geom;
""" % (BBOX, BBOX, BBOX)

STATIONS_Q = """
[out:json][timeout:600];
(
  node["railway"~"^(station|halt)$"](%s);
);
out body;
""" % BBOX


# Overpass turns away requests without one; kumi.systems says so outright and
# overpass-api.de answers 406.
UA = ("seoulriders/1.0 (map of Seoul subway ridership; "
      "https://github.com/ - contact via repo)")


def overpass(query, what):
    last = None
    for url in ENDPOINTS:
        for attempt in (1, 2):
            print("   %s via %s (try %d) ..." % (what, url.split("/")[2], attempt))
            try:
                r = requests.post(url, data={"data": query},
                                  headers={"User-Agent": UA}, timeout=1200)
                if r.status_code == 200:
                    return r.json()
                last = "HTTP %d: %s" % (r.status_code, r.text[:160])
            except Exception as e:      # noqa: BLE001 - report and try the next
                last = repr(e)
            print("      %s" % last)
            time.sleep(20)
    raise SystemExit("overpass failed for %s: %s" % (what, last))


def survey():
    data = overpass(SURVEY_Q, "relation tags")
    names = collections.Counter()
    for e in data.get("elements", []):
        t = e.get("tags", {})
        names[(t.get("route", "?"), t.get("name", "?").split(":")[0].strip())] += 1
    print("\n%d relations, %d distinct line names:\n" % (
        len(data.get("elements", [])), len(names)))
    for (route, name), c in sorted(names.items(), key=lambda kv: -kv[1]):
        print("   %-11s %-40s %3d" % (route, name[:40], c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true",
                    help="list the relation names Overpass returns, write nothing")
    ap.add_argument("--stations-only", action="store_true")
    ap.add_argument("--routes-only", action="store_true")
    args = ap.parse_args()

    if args.survey:
        survey()
        return

    if not args.stations_only:
        routes = overpass(ROUTES_Q, "route relations with geometry")
        out = os.path.join(D, "osm_routes.json")
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(routes, f, ensure_ascii=False)
        print("   wrote %s (%.1f MB, %d relations)"
              % (out, os.path.getsize(out) / 1e6, len(routes.get("elements", []))))

    if not args.routes_only:
        nodes = overpass(STATIONS_Q, "station nodes")
        out = os.path.join(D, "osm_stations.json")
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(nodes, f, ensure_ascii=False)
        print("   wrote %s (%.1f MB, %d nodes)"
              % (out, os.path.getsize(out) / 1e6, len(nodes.get("elements", []))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
