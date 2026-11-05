"""Download Geofabrik extracts into data/raw/osm/ (docs/roads.md).

    python isochrone/tools/roads/fetch.py europe/romania north-america/us/new-york ...

Each goes to data/raw/osm/<last part>.osm.pbf via a .part file, checked against Geofabrik's .md5.
An extract already there with a matching .md5 is skipped.
"""
import hashlib
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OSM = os.path.join(HERE, "data", "raw", "osm")
BASE = "https://download.geofabrik.de/"
UA = {"User-Agent": "isochrone-roads/1 (python urllib)"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch(region, name=None):
    name = name or region.rstrip("/").split("/")[-1]
    url = BASE + region + "-latest.osm.pbf"
    path = os.path.join(OSM, name + ".osm.pbf")
    want = get(url + ".md5").read().decode().split()[0]
    if os.path.exists(path) and md5_of(path) == want:
        print("%s: have it" % name, flush=True)
        return path
    t0 = time.time()
    part = path + ".part"
    with get(url) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got, last = 0, 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if time.time() - last > 15:
                last = time.time()
                print("  %s %.0f / %.0f MB, %.1f MB/s" % (name, got / 1e6, total / 1e6, got / 1e6 / (last - t0 + 1e-9)), flush=True)
    if md5_of(part) != want:
        raise SystemExit("%s: md5 mismatch" % name)
    os.replace(part, path)
    dt = time.time() - t0
    print("%s: %.0f MB in %.0f s (%.1f MB/s)" % (name, got / 1e6, dt, got / 1e6 / dt), flush=True)
    return path


if __name__ == "__main__":
    os.makedirs(OSM, exist_ok=True)
    for a in sys.argv[1:]:
        fetch(a)
