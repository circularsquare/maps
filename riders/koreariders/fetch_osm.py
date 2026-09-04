# -*- coding: utf-8 -*-
"""Pull intercity rail route relations from OSM for the whole of South Korea.

The 철도통계연보 lists each line's stations alphabetically and never says what
order they run in, so the running order has to come from somewhere else. OSM
route relations carry `stop`-role nodes in running order *and* way members with
full geometry, so one pull gives both.

    python fetch_osm.py --survey    # list matching relation names, fetch nothing
    python fetch_osm.py             # write data/osm_routes.json
"""

import argparse
import collections
import io
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# The whole of South Korea, 제주 included (no rail, but the box is free).
BBOX = "33.00,124.50,38.70,131.20"

SURVEY_Q = """
[out:json][timeout:600];
(
  relation["type"="route"]["route"="train"](%s);
);
out tags;
""" % BBOX

ROUTES_Q = """
[out:json][timeout:900];
(
  relation["type"="route"]["route"="train"](%s);
);
out geom;
""" % BBOX

# The route relations turned out to be uneven -- 경부선 and the KTX services have
# complete end-to-end relations, most lines have a stub covering the first 60 km.
# The track ways, though, carry the line name (64% of them, 167 distinct names,
# every line the yearbook reports on), so corridors get assembled from these.
RAILWAYS_Q = """
[out:json][timeout:900];
way["railway"="rail"]["name"](%s);
out geom;
""" % BBOX

# Korail's intercity stations are `railway=station`; the unstaffed 간이역 that
# still appear in the yearbook are often `halt`. `stop` is excluded on purpose --
# it is the platform-edge node, not the station.
STATIONS_Q = """
[out:json][timeout:600];
(
  node["railway"~"^(station|halt)$"](%s);
);
out body;
""" % BBOX

UA = ("koreariders/0.1 (national map of Korean rail throughput; "
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
        names[(t.get("name", "?").strip(), t.get("operator", "?"))] += 1
    print("\n%d relations, %d distinct (name, operator):\n"
          % (len(data.get("elements", [])), len(names)))
    for (name, op), c in sorted(names.items()):
        print("   %-46s %-18s %3d" % (name[:46], op[:18], c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true",
                    help="list the relation names Overpass returns, write nothing")
    args = ap.parse_args()

    if args.survey:
        survey()
        return

    if not os.path.exists(os.path.join(D, "osm_routes.json")):
        routes = overpass(ROUTES_Q, "route relations with geometry")
        out = os.path.join(D, "osm_routes.json")
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(routes, f, ensure_ascii=False)
        print("   wrote %s (%.1f MB, %d relations)"
              % (out, os.path.getsize(out) / 1e6, len(routes.get("elements", []))))

    if not os.path.exists(os.path.join(D, "osm_railways.json")):
        ways = overpass(RAILWAYS_Q, "named rail ways with geometry")
        out = os.path.join(D, "osm_railways.json")
        with io.open(out, "w", encoding="utf-8") as f:
            json.dump(ways, f, ensure_ascii=False)
        print("   wrote %s (%.1f MB, %d ways)"
              % (out, os.path.getsize(out) / 1e6, len(ways.get("elements", []))))

    nodes = overpass(STATIONS_Q, "station nodes")
    out = os.path.join(D, "osm_stations.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False)
    print("   wrote %s (%.1f MB, %d nodes)"
          % (out, os.path.getsize(out) / 1e6, len(nodes.get("elements", []))))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
