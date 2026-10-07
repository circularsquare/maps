"""OSM industrial land and labour camps in the six Gulf states, a placement input for
countries/_gulf_place.py (sources/gulf_place.md).

    python sources/gulf_osm.py --fetch      Overpass, per country, into data/raw/gulf/osm_<cc>_*.json
    python sources/gulf_osm.py              rebuild data/geo/gulf/<cc>_industrial.gpkg from them

What is taken (OSM, ODbL):
  * landuse=industrial ways and multipolygon relations (factories, workshops, oil and gas plants,
    the industrial areas where most labour camps stand: Mussafah/ICAD, Al Quoz, Jebel Ali, Doha's
    Industrial Area, Shuwaikh, Riyadh's 2nd industrial city, Jubail, Ruwais...)
  * anything named like a labour camp ("labour/labor camp", "workers accommodation / village /
    city / camp / housing / residence", "staff accommodation") or tagged
    residential=workers / building=dormitory with a camp-like name, buffered 150 m when a point.
Written as polygons, one layer per country, column `kind` (industrial / camp).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd  # noqa: E402
import requests  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import LineString, Point, Polygon  # noqa: E402
from shapely.ops import polygonize, unary_union  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "gulf"
OUT = ROOT / "data" / "geo" / "gulf"
CCS = ["SA", "AE", "QA", "KW", "OM", "BH"]
UA = "languagedots-research/0.1 (python-requests)"
EPS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
CAMP_RE = ("labou?r (camp|accommodation|village|city|housing)|workers?'? ?(accommodation|camp|"
           "village|city|housing|residence|residential)|staff (accommodation|camp|village)|"
           "construction camp|worker camp")

QUERIES = {
    "ind_way": '[out:json][timeout:600];area["ISO3166-1"="{cc}"][admin_level=2]->.a;'
               'way[landuse=industrial](area.a);out geom;',
    "ind_rel": '[out:json][timeout:600];area["ISO3166-1"="{cc}"][admin_level=2]->.a;'
               'rel[landuse=industrial][type=multipolygon](area.a);out geom;',
    "camp": '[out:json][timeout:600];area["ISO3166-1"="{cc}"][admin_level=2]->.a;'
            '(nwr["name"~"' + CAMP_RE + '",i](area.a);'
            'nwr["name:en"~"' + CAMP_RE + '",i](area.a);'
            'nwr[residential=workers](area.a););out geom;',
}


def fetch(cc):
    RAW.mkdir(parents=True, exist_ok=True)
    for key, q in QUERIES.items():
        path = RAW / f"osm_{cc.lower()}_{key}.json"
        if path.exists():
            print(f"  {path.name}: have it")
            continue
        for i in range(6):
            ep = EPS[i % 2]
            try:
                r = requests.post(ep, data={"data": q.format(cc=cc)}, headers={"User-Agent": UA},
                                  timeout=700)
            except requests.RequestException as e:
                print(f"  {ep}: {e}")
                time.sleep(20)
                continue
            print(f"  {cc} {key}: {ep} {r.status_code} {len(r.content):,} bytes")
            if r.status_code == 200 and r.content.lstrip()[:1] == b"{":
                js = r.json()
                if "remark" in js and "error" in js["remark"].lower():
                    print(f"    remark: {js['remark'][:200]}")
                    time.sleep(30)
                    continue
                path.write_bytes(r.content)
                break
            time.sleep(30)
        else:
            raise SystemExit(f"Overpass failed for {cc} {key}")


def _way_poly(el):
    pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
    if len(pts) >= 4 and pts[0] == pts[-1]:
        p = Polygon(pts)
        return p if p.is_valid else p.buffer(0)
    if len(pts) >= 2:
        return LineString(pts).buffer(0.0007)     # an unclosed outline, a narrow strip
    return None


def _rel_poly(el):
    outer, inner = [], []
    for m in el.get("members", []):
        g = m.get("geometry")
        if m.get("type") != "way" or not g:
            continue
        (outer if m.get("role", "outer") != "inner" else inner).append(
            LineString([(p["lon"], p["lat"]) for p in g]))
    polys = list(polygonize(unary_union(outer))) if outer else []
    if not polys:
        return None
    shape = unary_union(polys)
    if inner:
        holes = list(polygonize(unary_union(inner)))
        if holes:
            shape = shape.difference(unary_union(holes))
    return shape


def build(cc):
    geoms, kinds = [], []
    for key in QUERIES:
        path = RAW / f"osm_{cc.lower()}_{key}.json"
        if not path.exists():
            raise SystemExit(f"{path.name} missing: run with --fetch")
        els = json.loads(path.read_text(encoding="utf-8"))["elements"]
        n = 0
        for el in els:
            if el["type"] == "node":
                g = Point(el["lon"], el["lat"]).buffer(0.0014)    # ~150 m
            elif el["type"] == "way":
                g = _way_poly(el)
            else:
                g = _rel_poly(el)
            if g is None or g.is_empty:
                continue
            if key == "camp" and g.area < 2e-6:                  # a camp drawn as a building
                g = g.centroid.buffer(0.0014)
            geoms.append(g)
            kinds.append("camp" if key == "camp" else "industrial")
            n += 1
        print(f"  {cc} {key}: {len(els):,} elements, {n:,} shapes")
    gdf = gpd.GeoDataFrame({"kind": kinds}, geometry=geoms, crs=4326)
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    OUT.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT / f"{cc.lower()}_industrial.gpkg", driver="GPKG")
    km2 = (gdf.to_crs(6933).area / 1e6).groupby(gdf["kind"]).sum()
    print(f"  -> {cc.lower()}_industrial.gpkg: {len(gdf):,} shapes; "
          + ", ".join(f"{k} {v:,.0f} km2" for k, v in km2.items()))


# OSM admin_level 6 relations used to split or check units (sources/gulf_place.md):
# Abu Dhabi emirate's three regions (SCAD publishes Emiratis and non-Emiratis for each), and
# Riyadh and Ad Diriyah governorates (RCRC's census table, a witness for the placement fit).
ADMIN = {13249273: "Abu Dhabi Region", 13249272: "Al Ain Region", 13249274: "Al Dhafra Region",
         12423679: "Riyadh governorate", 12423664: "Ad Diriyah governorate"}


def fetch_admin():
    path = RAW / "osm_admin.json"
    if path.exists():
        print(f"  {path.name}: have it")
        return
    q = f"[out:json][timeout:600];rel(id:{','.join(map(str, ADMIN))});out geom;"
    for i in range(6):
        r = requests.post(EPS[i % 2], data={"data": q}, headers={"User-Agent": UA}, timeout=700)
        print(f"  admin: {EPS[i % 2]} {r.status_code} {len(r.content):,} bytes")
        if r.status_code == 200:
            path.write_bytes(r.content)
            return
        time.sleep(30)
    raise SystemExit("Overpass failed for the admin relations")


def build_admin():
    els = json.loads((RAW / "osm_admin.json").read_text(encoding="utf-8"))["elements"]
    rows = [(ADMIN[e["id"]], _rel_poly(e)) for e in els]
    if sorted(n for n, _ in rows) != sorted(ADMIN.values()) or any(g is None for _, g in rows):
        raise SystemExit(f"osm_admin.json: got {[n for n, g in rows if g is not None]}")
    gdf = gpd.GeoDataFrame({"name": [n for n, _ in rows]}, geometry=[g for _, g in rows], crs=4326)
    gdf.to_file(OUT / "admin.gpkg", driver="GPKG")
    print("  -> admin.gpkg: " + ", ".join(
        f"{n} {a:,.0f} km2" for n, a in zip(gdf["name"], gdf.to_crs(6933).area / 1e6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("cc", nargs="*", default=CCS)
    a = ap.parse_args()
    for cc in [c.upper() for c in a.cc]:
        if a.fetch:
            fetch(cc)
        build(cc)
    if a.fetch:
        fetch_admin()
    build_admin()


if __name__ == "__main__":
    main()
