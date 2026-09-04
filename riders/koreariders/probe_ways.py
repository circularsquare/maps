# -*- coding: utf-8 -*-
"""Do Korean rail tracks carry their line name in OSM?

The route relations turned out to be uneven -- some lines have a complete
end-to-end relation, most have a stub covering the first 60 km. If the
`railway=rail` ways themselves are named, corridors can be assembled by name
instead, which would cover the network uniformly.

    python probe_ways.py
"""

import collections
import io
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = os.path.join(D, "osm_rail_ways.json")

BBOX = "34.00,125.50,38.70,129.80"
UA = "koreariders/0.1 (national map of Korean rail throughput)"

Q = """
[out:json][timeout:600];
way["railway"="rail"]["usage"!="industrial"](%s);
out tags;
""" % BBOX


def main():
    if os.path.exists(OUT):
        with io.open(OUT, encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": Q}, headers={"User-Agent": UA}, timeout=900)
        r.raise_for_status()
        data = r.json()
        with io.open(OUT, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    ways = data["elements"]
    named = [w for w in ways if (w.get("tags") or {}).get("name")]
    print("%d rail ways, %d with a name (%.0f%%)"
          % (len(ways), len(named), 100.0 * len(named) / max(len(ways), 1)))

    keys = collections.Counter()
    for w in ways:
        for k in (w.get("tags") or {}):
            keys[k] += 1
    print("\ncommonest tags: %s"
          % ", ".join("%s(%d)" % (k, c) for k, c in keys.most_common(14)))

    names = collections.Counter((w["tags"]["name"]) for w in named)
    print("\n%d distinct names; top 40:" % len(names))
    for n, c in names.most_common(40):
        print("   %-28s %4d" % (n[:28], c))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
