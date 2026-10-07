"""OpenStreetMap district (admin_level=6) and tehsil (admin_level=7) relations for Pakistan.

Fetched one province at a time from Overpass (a single all-Pakistan `out geom` query timed out),
then assembled into polygons from their member ways.
"""

import json
import os
import time

import requests

UA = {"User-Agent": "helper1m-map-research/1.0"}
SERVERS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter"]
TAGS_Q = ('[out:json][timeout:180];area["ISO3166-1"="PK"][admin_level=2]->.pk;'
          'rel(area.pk)[boundary=administrative][admin_level~"^(4|5|6|7|8)$"];out tags;')


def _post(q):
    for srv in SERVERS:
        for _ in range(2):
            try:
                r = requests.post(srv, data={"data": q}, timeout=900, headers=UA)
            except requests.RequestException as ex:
                print("   ", srv, ex)
                continue
            if r.status_code == 200 and r.content.lstrip().startswith(b"{"):
                return r.content
            print("   ", srv, r.status_code)
            time.sleep(10)
    raise SystemExit("Overpass: every server refused")


def fetch(raw_dir):
    tags_path = os.path.join(raw_dir, "osm_pk_admin_tags.json")
    if not os.path.exists(tags_path):
        open(tags_path, "wb").write(_post(TAGS_Q))
    tags = json.load(open(tags_path, encoding="utf-8"))
    for e in tags["elements"]:
        if e["tags"].get("admin_level") != "4":
            continue
        out = os.path.join(raw_dir, f"osm_tehsils_{e['id']}.json")
        if os.path.exists(out) and os.path.getsize(out) > 200:
            continue
        q = (f'[out:json][timeout:600];rel({e["id"]});map_to_area->.p;'
             'rel(area.p)[boundary=administrative][admin_level~"^(6|7)$"];out geom;')
        open(out, "wb").write(_post(q))
        print(f"  fetched OSM admin 6/7 for {e['tags'].get('name:en')}")


def polygons(raw_dir):
    """GeoDataFrame of every admin 6/7 relation: osm_id, level, name, geometry (EPSG:4326)."""
    import glob

    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    rows, seen = [], set()
    for path in sorted(glob.glob(os.path.join(raw_dir, "osm_tehsils_*.json"))):
        for e in json.load(open(path, encoding="utf-8"))["elements"]:
            if e["type"] != "relation" or e["id"] in seen:
                continue
            seen.add(e["id"])
            outer, inner = [], []
            for m in e.get("members", []):
                if m["type"] != "way" or "geometry" not in m:
                    continue
                ls = LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                (inner if m.get("role") == "inner" else outer).append(ls)
            polys = list(polygonize(unary_union(outer))) if outer else []
            geom = unary_union(polys) if polys else None
            if geom is not None and inner:
                holes = list(polygonize(unary_union(inner)))
                if holes:
                    geom = geom.difference(unary_union(holes))
            t = e.get("tags", {})
            rows.append({"osm_id": e["id"], "level": int(t.get("admin_level", 0)),
                         "name": t.get("name:en") or t.get("name"), "name_local": t.get("name"),
                         "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
