"""OSM water points for the "today" variant of the thirst factor (docs/hazard.md), into data/raw/osm_water/.

    python isochrone/notes/T-042/fetch_osm_water.py

Wells, springs and settlements (a place people live has water), one TSV per tag: lat, lon.
QLever's OSM planet endpoint, as tools/canoe/rivers_osm.py. Tags already on disk are skipped.
"""
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, "data", "raw", "osm_water")
URL = "https://qlever.dev/api/osm-planet"
UA = "isochrone-map-build/1.0 (water points for a travel-time map)"
QUERY = """PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
SELECT (geof:centroid(?wkt) AS ?c) WHERE {
  ?osm osmkey:%s "%s" .
  ?osm geo:hasGeometry/geo:asWKT ?wkt .
}
"""
TAGS = [("man_made", "water_well"), ("natural", "spring"), ("place", "city"), ("place", "town"),
        ("place", "village"), ("place", "hamlet")]
ROW = re.compile(r"POINT\(([-0-9.eE]+) ([-0-9.eE]+)\)")


def pull(key, value):
    path = os.path.join(OUT, "%s_%s.tsv" % (key, value))
    if os.path.exists(path):
        print(path, "on disk", flush=True)
        return
    for attempt in range(4):
        t0 = time.time()
        try:
            r = requests.post(URL, data={"query": QUERY % (key, value)},
                              headers={"Accept": "text/tab-separated-values", "User-Agent": UA}, timeout=1800)
        except requests.RequestException as ex:
            print("  %s=%s error %s" % (key, value, ex), flush=True)
            time.sleep(60 * (attempt + 1))
            continue
        if r.status_code != 200:
            print("  %s=%s failed %d: %s" % (key, value, r.status_code, r.text[:200]), flush=True)
            time.sleep(60 * (attempt + 1))
            continue
        rows = []
        for line in r.text.splitlines()[1:]:
            m = ROW.search(line)
            if m:
                rows.append(m.group(2) + "\t" + m.group(1))
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        os.replace(path + ".tmp", path)
        print("%s=%s: %d rows, %.0f s" % (key, value, len(rows), time.time() - t0), flush=True)
        return
    print("GAVE UP %s=%s" % (key, value), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for key, value in TAGS:
        pull(key, value)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
