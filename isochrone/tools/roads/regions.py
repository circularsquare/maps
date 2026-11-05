"""The Geofabrik extracts a world roads pass reads (docs/roads.md "Fetching the world").

    python isochrone/tools/roads/regions.py            (writes data/roads/regions.json)

Reads Geofabrik's index (index-v1.json, every extract with its outline), keeps the leaves of its
tree, drops a leaf that is a grouping of other leaves (Alps, DACH, US census regions...) or that
lies inside another kept leaf, adds the parents in EXTRA, and asks Geofabrik for each one's size (HEAD). The list is ordered
west to east by the outline's centre, so neighbouring extracts run near each other.
"""
import json
import os
import time
import urllib.request

from shapely.geometry import shape

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, "data", "roads", "regions.json")
INDEX = os.path.join(HERE, "data", "raw", "osm", "index-v1.json")
UA = {"User-Agent": "isochrone-roads/1 (python urllib)"}
# parents read as well: no leaf covers the Lesser Antilles and Cayman (notes/T-010.md, coverage)
EXTRA = ["central-america"]


def get(url, method="GET"):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA, method=method), timeout=60)


def main():
    if not os.path.exists(INDEX):
        os.makedirs(os.path.dirname(INDEX), exist_ok=True)
        data = get("https://download.geofabrik.de/index-v1.json").read()
        with open(INDEX, "wb") as f:
            f.write(data)
    feats = json.load(open(INDEX, encoding="utf-8"))["features"]
    by = {f["properties"]["id"]: f for f in feats}
    parents = {f["properties"].get("parent") for f in feats}
    leaves = [f for f in feats if f["properties"]["id"] not in parents and "pbf" in f["properties"].get("urls", {})]
    geo = {f["properties"]["id"]: shape(f["geometry"]).buffer(0) for f in leaves}
    # drop a leaf whose outline is mostly covered by other leaves (a grouping or a duplicate)
    ids = sorted(geo, key=lambda i: -geo[i].area)
    kept = []
    for i in ids:
        g = geo[i]
        cover = 0.0
        for j in ids:
            if j == i or geo[j].area >= g.area:
                continue
            if g.intersects(geo[j]):
                cover += g.intersection(geo[j]).area
        if cover > 0.6 * g.area:
            print("drop %s (%.0f%% covered by smaller extracts)" % (i, 100 * cover / g.area))
            continue
        kept.append(i)
    for i in EXTRA:
        if i in by:
            geo[i] = shape(by[i]["geometry"]).buffer(0)
            kept.append(i)
    rows = []
    # sizes already known (an earlier list) are kept: Geofabrik can take 30 s a HEAD when busy
    known = {r["id"]: r["bytes"] for r in json.load(open(OUT))} if os.path.exists(OUT) else {}
    for i in kept:
        url = by[i]["properties"]["urls"]["pbf"]
        if known.get(i):
            size = known[i]
        else:
            try:
                size = int(get(url, "HEAD").headers.get("Content-Length", 0))
            except Exception as e:  # noqa: BLE001
                print("HEAD failed for %s: %s" % (i, e))
                size = 0
        c = geo[i].centroid
        b = geo[i].bounds
        rows.append({"id": i, "url": url, "bytes": size, "lon": round(c.x, 2), "lat": round(c.y, 2),
                     "bounds": [round(v, 2) for v in b]})
        if i not in known:
            time.sleep(0.2)
    rows.sort(key=lambda r: (r["lon"] // 30, -r["lat"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=0)
    tot = sum(r["bytes"] for r in rows)
    big = sorted(rows, key=lambda r: -r["bytes"])[:12]
    print("%d extracts, %.1f GB" % (len(rows), tot / 1e9))
    for r in big:
        print("  %-40s %.2f GB" % (r["id"], r["bytes"] / 1e9))


if __name__ == "__main__":
    main()
