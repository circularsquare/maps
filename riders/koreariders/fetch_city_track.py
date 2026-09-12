# -*- coding: utf-8 -*-
"""Pull city metro track geometry from OSM, for the five intracity models.

Busan, Daegu, Daejeon, Gwangju and Busan-Gimhae are all drawn as straight hops
between published station coordinates, which reads as a diagram rather than a
railway -- Busan line 2 runs a long curve around the bay and the map draws a
chord across it.

`railway=subway|light_rail|monorail` ways carry the line name the way the
intercity
`railway=rail` ways do, so the same approach works: pull the named ways per
city, then match them to the drawn stations by proximity. Route relations are
pulled too and preferred where they are complete, since they give the running
order for free -- the same trade `fetch_osm.py` weighed for the intercity net,
where relations turned out to be stubs. Here they are mostly whole, because a
metro line is short enough for one mapper to finish.

    python fetch_city_track.py --survey    # list what OSM has, fetch nothing
    python fetch_city_track.py             # write data/osm_city_track.json
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
OUT = os.path.join(D, "osm_city_track.json")

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Both endpoints filter on the User-Agent, and neither says so in a way the
# status code alone reveals: overpass-api.de answers a default python-requests
# header with a bare Apache **406 Not Acceptable** -- the same 406 that made
# the earlier attempt at city geometry look like a query problem -- while
# kumi.systems returns 429 with the reason in the body, "Please include a
# meaningful User-Agent string with your requests to avoid rate-limiting".
# fetch_osm.py has always sent one, which is why it works and this did not.
UA = ("koreariders/0.1 (national map of Korean rail throughput; "
      "https://github.com/ - contact via repo)")

# One box per city rather than one for the country: these are small networks and
# a tight box keeps the reply to something Overpass will actually return. The
# earlier attempt at this pulled the whole peninsula and got HTTP 406.
CITIES = {
    # name: (south, west, north, east)
    "busan": (35.05, 128.75, 35.35, 129.30),
    "daegu": (35.72, 128.42, 36.00, 128.78),
    "daejeon": (36.24, 127.30, 36.42, 127.50),
    "gwangju": (35.09, 126.75, 35.24, 126.95),
    "busan_gimhae": (35.15, 128.78, 35.28, 129.00),
}

SURVEY_Q = """
[out:json][timeout:300];
(
  way["railway"~"^(subway|light_rail|monorail)$"]["name"](%s);
  relation["type"="route"]["route"~"^(subway|light_rail|monorail)$"](%s);
);
out tags;
"""

# Asking for ways and relations in one `out geom` reply times the server out
# with a 504: a route relation repeats every member way's full geometry, so the
# combined body is several times what either half costs. Two lighter queries
# come back where one heavy one does not.
WAYS_Q = """
[out:json][timeout:600];
way["railway"~"^(subway|light_rail|monorail)$"]["name"](%s);
out geom;
"""

RELS_Q = """
[out:json][timeout:600];
relation["type"="route"]["route"~"^(subway|light_rail|monorail)$"](%s);
out geom;
"""


def overpass(query, tries=3):
    """Post to Overpass, falling back to the mirror and backing off on 429.

    Overpass answers a rejected query with a 200 and an HTML body often enough
    that the status alone is not a check; a JSON reply that will not parse is
    the real signal.
    """
    last = None
    for attempt in range(tries):
        for url in ENDPOINTS:
            try:
                r = requests.post(url, data={"data": query},
                                  headers={"User-Agent": UA}, timeout=900)
            except requests.RequestException as e:
                last = "%s: %s" % (url, e)
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    last = "%s: 200 but not JSON (%s)" % (url, r.text[:120])
                    continue
            last = "%s: HTTP %d" % (url, r.status_code)
        wait = 20 * (attempt + 1)
        print("   retrying in %ds (%s)" % (wait, last))
        time.sleep(wait)
    raise SystemExit("Overpass failed: %s" % last)


def bbox(city):
    return "%.4f,%.4f,%.4f,%.4f" % CITIES[city]


def survey():
    for city in CITIES:
        b = bbox(city)
        data = overpass(SURVEY_Q % (b, b))
        ways = collections.Counter()
        rels = collections.Counter()
        for el in data.get("elements", []):
            nm = (el.get("tags") or {}).get("name")
            if not nm:
                continue
            (ways if el["type"] == "way" else rels)[nm] += 1
        print("\n=== %s" % city)
        print("   %d named ways over %d names, %d route relations"
              % (sum(ways.values()), len(ways), sum(rels.values())))
        for nm, n in ways.most_common(12):
            print("      way  %-38s %d" % (nm, n))
        for nm, n in rels.most_common(12):
            print("      rel  %-38s %d" % (nm, n))


def fetch():
    out = {}
    for city in CITIES:
        print("fetching %s ..." % city)
        b = bbox(city)
        els = list(overpass(WAYS_Q % b).get("elements", []))
        els += list(overpass(RELS_Q % b).get("elements", []))
        keep = []
        for el in els:
            tags = el.get("tags") or {}
            if not tags.get("name"):
                continue
            if el["type"] == "way" and el.get("geometry"):
                keep.append({"type": "way", "name": tags["name"],
                             "railway": tags.get("railway"),
                             "geometry": [[round(p["lon"], 6), round(p["lat"], 6)]
                                          for p in el["geometry"]]})
            elif el["type"] == "relation":
                parts = []
                for m in el.get("members", []):
                    if m.get("type") == "way" and m.get("geometry"):
                        parts.append([[round(p["lon"], 6), round(p["lat"], 6)]
                                      for p in m["geometry"]])
                if parts:
                    keep.append({"type": "relation", "name": tags["name"],
                                 "route": tags.get("route"), "parts": parts})
        out[city] = keep
        print("   %d named elements" % len(keep))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("\nwrote %s (%.1f MB)"
          % (os.path.relpath(OUT, HERE), os.path.getsize(OUT) / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true")
    args = ap.parse_args()
    if args.survey:
        survey()
    else:
        fetch()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
