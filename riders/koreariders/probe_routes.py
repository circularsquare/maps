# -*- coding: utf-8 -*-
"""Look at how one OSM route relation is put together, before trusting it.

    python probe_routes.py "전라선 무궁화호"
"""

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = os.path.join(HERE, "data", "osm_routes.json")


def main():
    needle = sys.argv[1] if len(sys.argv) > 1 else "전라선"
    with io.open(ROUTES, encoding="utf-8") as f:
        data = json.load(f)

    for rel in data["elements"]:
        name = rel.get("tags", {}).get("name", "")
        if needle not in name:
            continue
        members = rel.get("members", [])
        roles = {}
        for m in members:
            roles[(m["type"], m.get("role", ""))] = roles.get(
                (m["type"], m.get("role", "")), 0) + 1
        print("\n=== %s  (id %s)" % (name, rel["id"]))
        print("    tags: %s" % {k: v for k, v in rel["tags"].items()
                                if k in ("route", "operator", "from", "to", "ref")})
        print("    member roles: %s" % roles)
        stops = [m for m in members
                 if m["type"] == "node" and m.get("role", "").startswith("stop")]
        if not stops:
            stops = [m for m in members if m["type"] == "node"]
        print("    %d stop nodes" % len(stops))
        # Overpass `out geom` gives node members a lat/lon but not their tags,
        # so the name has to come from somewhere else -- check what is present.
        print("    first stop member keys: %s"
              % sorted(stops[0].keys()) if stops else "    (none)")
        ways = [m for m in members if m["type"] == "way"]
        withgeom = sum(1 for m in ways if m.get("geometry"))
        print("    %d way members, %d with geometry" % (len(ways), withgeom))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
