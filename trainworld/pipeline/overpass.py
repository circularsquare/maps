"""One cached Overpass request (T-142: airport terminals, medical facilities). The answer is kept
in `data/raw/<name>.json`; delete it to refetch. Data (c) OpenStreetMap contributors, ODbL."""
import json
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
# Overpass answers a bare client with 406 or a false 429; a descriptive agent, no personal details.
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}
# tried in turn; 2026-10-12 the main instance answered 504 to every park tile while private.coffee
# answered in seconds (T-196), so it goes first
OVERPASS = ["https://overpass.private.coffee/api/interpreter", "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
STATUS = "https://overpass-api.de/api/status"


def bbox_of(geom, pad=0.0):
    """Overpass's (south, west, north, east) for a lon/lat geometry."""
    w, s, e, n = geom.bounds
    return (round(s - pad, 4), round(w - pad, 4), round(n + pad, 4), round(e + pad, 4))


def fetch(name, q):
    path = os.path.join(RAW, name + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    for attempt in range(6):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=600)
        except requests.RequestException as ex:
            print(f"  {name}: {url} failed ({ex}); retrying", flush=True)
            time.sleep(10)
            continue
        if r.status_code == 200:
            data = r.json()
            if data.get("remark", "").lower().startswith("runtime error"):
                print(f"  {name}: {data['remark']}", flush=True)
                time.sleep(30)
                continue
            tmp = path + ".tmp"
            os.makedirs(os.path.dirname(path), exist_ok=True)  # a sub-folder (t196/) deleted after use
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)
            return data
        print(f"  {name}: {url} said {r.status_code}", flush=True)
        if r.status_code == 429:
            try:
                print("  status: " + requests.get(STATUS, headers=UA, timeout=30).text.replace("\n", " | "), flush=True)
            except requests.RequestException:
                pass
        # a 504 is the server's gateway giving up, not a rate limit: try the next server soon
        time.sleep(30 if r.status_code == 429 else 10)
    raise SystemExit(f"{name}: Overpass failed")


def polygon(el):
    """A closed way's or a multipolygon relation's lon/lat polygon from `out geom`, else None.
    A relation needs `out geom` with body verbosity: `out geom tags` leaves out its members (T-196
    found this; enrich_medical.py's query has it, so its multipolygon hospitals are skipped)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union, polygonize
    from shapely.geometry import LineString
    from shapely import make_valid
    if el["type"] == "way":
        g = el.get("geometry") or []
        if len(g) >= 4 and g[0] == g[-1]:
            p = make_valid(Polygon([(v["lon"], v["lat"]) for v in g]))
            return p if p.area > 0 else None
        return None
    if el["type"] == "relation":
        # an empty role is an outer ring in older multipolygons
        outer = [LineString([(v["lon"], v["lat"]) for v in m["geometry"]])
                 for m in el.get("members", []) if m.get("role") in ("outer", "") and len(m.get("geometry") or []) >= 2]
        inner = [LineString([(v["lon"], v["lat"]) for v in m["geometry"]])
                 for m in el.get("members", []) if m.get("role") == "inner" and len(m.get("geometry") or []) >= 2]
        polys = list(polygonize(unary_union(outer))) if outer else []
        if not polys:
            return None
        p = unary_union(polys)
        holes = list(polygonize(unary_union(inner))) if inner else []
        if holes:
            p = p.difference(unary_union(holes))
        p = make_valid(p)
        return p if p.area > 0 else None
    return None


def area_m2(geom):
    """Area of a lon/lat polygon in square metres (local equirectangular; fine at city scale)."""
    import numpy as np
    from shapely.ops import transform
    lat0 = geom.centroid.y
    k = np.cos(np.radians(lat0))
    r = 6371008.8
    return transform(lambda x, y: (np.radians(x) * r * k, np.radians(y) * r), geom).area
