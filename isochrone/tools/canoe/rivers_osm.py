"""Pull OSM waterfalls, dams, weirs and rapids into data/raw/osm_barriers/, one TSV per tag.

    python tools/canoe/rivers_osm.py

Uses QLever's OSM planet SPARQL endpoint (qlever.dev, osm2rdf of the whole planet), one query
per tag, centroids of ways and relations computed by the server. The public Overpass servers
were at capacity on 2026-10-26 and global per-box pulls kept timing out. Columns: type, id,
lat, lon, waterway, natural (as an Overpass CSV would give). Tags already on disk are skipped.
"""
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, "data", "raw", "osm_barriers")
URL = "https://qlever.dev/api/osm-planet"
UA = "isochrone-map-build/1.0 (rivers data for a travel-time map)"
QUERY = """PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
SELECT ?osm (geof:centroid(?wkt) AS ?c) WHERE {
  ?osm osmkey:%s "%s" .
  ?osm geo:hasGeometry/geo:asWKT ?wkt .
}
"""
TAGS = [("waterway", "waterfall"), ("natural", "waterfall"), ("waterway", "dam"), ("waterway", "weir"),
        ("waterway", "rapids")]
ROW = re.compile(r"<https://www\.openstreetmap\.org/(node|way|relation)/(\d+)>\tPOINT\(([-0-9.eE]+) ([-0-9.eE]+)\)")


def pull(key, value):
    path = os.path.join(OUT, "%s_%s.tsv" % (key, value))
    if os.path.exists(path):
        print(path, "on disk", flush=True)
        return
    for attempt in range(5):
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
        rows, bad = [], 0
        for line in r.text.splitlines()[1:]:
            m = ROW.match(line)
            if not m:
                bad += 1
                continue
            typ, oid, lon, lat = m.groups()
            ww = value if key == "waterway" else ""
            nat = value if key == "natural" else ""
            rows.append("\t".join((typ, oid, lat, lon, ww, nat)))
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        os.replace(path + ".tmp", path)
        print("%s=%s: %d rows, %d unparsed, %.0f s" % (key, value, len(rows), bad, time.time() - t0), flush=True)
        return
    print("GAVE UP %s=%s" % (key, value), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for key, value in TAGS:
        pull(key, value)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
