"""T-3244: universities and colleges with their outlines from Overpass, for packs whose OSM education
cache was made before T-3214 (no `wkt` column), so a light pass needs no extract read.

    python pipeline/campus_outlines.py fetch     the world's, in longitude bands, one request at a time
    python pipeline/campus_outlines.py split [<ids>]   each pack's part, from the bands (default: every
                                                  registered pack with an education cache without outlines)
    python pipeline/campus_outlines.py status    bands fetched, packs split

A band's answer is kept in data/raw/tier0/campus/bands/ and a pack's part in
data/raw/tier0/campus/<id>.json.gz (Overpass's answer shape; delete either to make it again). Bands,
not a request a pack: a request costs Overpass a few minutes almost whatever its box (2026-11-12:
Vancouver's box 35-45 s, a 10 degree box 190 s, the world 200 s), so 3,900 pack requests would take
days. A pack's part is every element whose box meets the pack's boundary box (with PAD): the same rows
as a request with that box gives once tier0 keeps only sites inside the boundary.
`tier0.osm_education` reads a pack's part through `rows()`: a cache without `wkt` keeps its schools
and takes its universities and colleges from here, built as the extract path builds them.
Data (c) OpenStreetMap contributors, ODbL.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import shapely

import overpass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKS = os.path.join(ROOT, "data", "packs")
REGISTRY = os.path.join(ROOT, "app", "src", "game", "registry.json")
TIER0 = os.path.join(ROOT, "data", "raw", "tier0")
SUB = "tier0/campus/"
KINDS = ("university", "college")
BANDS = [(-180, -130), (-130, -110), (-110, -90), (-90, -75), (-75, -60), (-60, -40), (-40, -10), (-10, 0), (0, 10),
         (10, 20), (20, 30), (30, 45), (45, 65), (65, 80), (80, 95), (95, 105), (105, 115), (115, 125), (125, 140), (140, 180)]
PAD = 0.01        # degrees round a pack's boundary box
TIMEOUT_S = 600   # Overpass's own limit a band; the client waits a little longer
TRIES = 6


def name(city):
    return SUB + city


def band_name(kind, w, e):
    return f"{SUB}bands/{kind}_{w}_{e}"


def band_query(kind, w, e):
    bb = f"-90,{w},90,{e}"
    k = f'["amenity"="{kind}"]'
    # Overpass admits a query once its declared timeout and memory fit, so a big declaration waits
    # in its queue until the gateway gives up (504s, 2026-11-12): the defaults' memory, a modest timeout
    return (f'[out:json][timeout:{TIMEOUT_S}];(way{k}({bb});relation{k}["type"="multipolygon"]({bb});'
            f'node{k}({bb}););out geom;')


def get(name_, q):
    """overpass.fetch with a client wait to match a long query, and a longer backoff."""
    if overpass.cache_path(name_):
        return overpass.read_cache(name_)
    for attempt in range(TRIES):
        url = overpass.OVERPASS[[1, 0, 2, 3][attempt % 4]]  # the main instance first, then the others in turn
        t = time.time()
        try:
            r = requests.post(url, data={"data": q}, headers=overpass.UA, timeout=TIMEOUT_S + 120)
        except requests.RequestException as ex:
            print(f"  {name_}: {url} failed after {time.time() - t:.0f} s ({type(ex).__name__}); waiting", flush=True)
            time.sleep(60 * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            if data.get("remark", "").lower().startswith("runtime error"):
                print(f"  {name_}: {data['remark'][:200]}", flush=True)
                time.sleep(60 * (attempt + 1))
                continue
            overpass.write_cache(name_, data)
            print(f"  {name_}: {len(data.get('elements', [])):,} elements, {len(r.content) / 1e6:.0f} MB in "
                  f"{time.time() - t:.0f} s from {url}", flush=True)
            return data
        print(f"  {name_}: {url} said {r.status_code} after {time.time() - t:.0f} s; waiting", flush=True)
        time.sleep((120 if r.status_code == 429 else 60) * (attempt + 1))
    raise SystemExit(f"{name_}: Overpass failed {TRIES} times")


def fetch():
    t0, failed = time.time(), []
    todo = [(k, w, e) for w, e in BANDS for k in KINDS]
    for i, (k, w, e) in enumerate(todo, 1):
        try:
            get(band_name(k, w, e), band_query(k, w, e))
        except SystemExit as ex:
            failed.append(band_name(k, w, e))
            print(f"  {ex}", flush=True)
        print(f"[{i}/{len(todo)}] {time.time() - t0:.0f} s", flush=True)
        time.sleep(5)
    print(f"bands done in {time.time() - t0:.0f} s; {len(failed)} failed {' '.join(failed)}", flush=True)
    return failed


def box(el):
    if el["type"] == "node":
        return el["lon"], el["lat"], el["lon"], el["lat"]
    b = el.get("bounds")
    if b:
        return b["minlon"], b["minlat"], b["maxlon"], b["maxlat"]
    pts = [(g["lon"], g["lat"]) for g in el.get("geometry") or []] + \
          [(g["lon"], g["lat"]) for m in el.get("members", []) for g in (m.get("geometry") or [])]
    if not pts:
        return None
    a = np.array(pts)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def bounds_of(city, packs=PACKS):
    with open(os.path.join(packs, f"{city}.boundary.geojson")) as f:
        fc = json.load(f)
    feats = fc["features"] if "features" in fc else [fc]
    # Each part's own box, not a union: some boundaries are invalid (Kolkata's raises a GEOS side conflict).
    b = np.array([shapely.geometry.shape(x["geometry"]).bounds for x in feats])
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()


def split(ids):
    t0 = time.time()
    els, seen = [], set()
    for w, e in BANDS:
        for k in KINDS:
            n = band_name(k, w, e)
            if not overpass.cache_path(n):
                raise SystemExit(f"{n}: not fetched (campus_outlines.py fetch)")
            for el in overpass.read_cache(n).get("elements", []):
                key = (el["type"], el["id"])
                if key not in seen:
                    seen.add(key)
                    els.append(el)
    boxes = [box(el) for el in els]
    keep = [i for i, b_ in enumerate(boxes) if b_ is not None]
    tree = shapely.STRtree(shapely.box(*np.array([boxes[i] for i in keep]).T))
    print(f"{len(els):,} elements from {len(BANDS) * len(KINDS)} bands in {time.time() - t0:.0f} s", flush=True)
    stamp = overpass.read_cache(band_name(KINDS[0], *BANDS[0])).get("osm3s", {})
    n_done = 0
    for c in ids:
        try:
            w, s, e, n = bounds_of(c)
        except OSError:
            print(f"  {c}: no boundary, skipped", flush=True)
            continue
        hit = tree.query(shapely.box(w - PAD, s - PAD, e + PAD, n + PAD))
        overpass.write_cache(name(c), {"osm3s": stamp, "source": "campus_outlines.py bands",
                                       "elements": [els[keep[j]] for j in sorted(hit)]})
        n_done += 1
    print(f"split into {n_done} packs in {time.time() - t0:.0f} s", flush=True)


def rows(city):
    """(kind, name, lat, lon, m2, wikidata, wkt) of the pack's universities and colleges as
    osm_education's extract path makes them (an outline's representative point, area and outline at
    about 3 m; a point inside a mapped site of its kind left out), or None when nothing is staged."""
    if not overpass.cache_path(name(city)):
        return None
    data = overpass.read_cache(name(city))
    out, polys, pts = [], {k: [] for k in KINDS}, []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        kind = tags.get("amenity")
        if kind not in polys:
            continue
        if el["type"] == "node":
            pts.append((kind, tags.get("name", ""), el["lat"], el["lon"], tags.get("wikidata", "")))
            continue
        g = overpass.polygon(el)
        if g is None:
            continue
        try:
            g = shapely.make_valid(g)
            c = g.representative_point()
            a = overpass.area_m2(g)
        except shapely.errors.GEOSException:
            continue
        out.append((kind, tags.get("name", ""), c.y, c.x, a, tags.get("wikidata", ""),
                    shapely.to_wkt(g.simplify(0.00003), rounding_precision=6)))
        polys[kind].append(g)
    trees = {k: shapely.STRtree(v) for k, v in polys.items() if v}
    for kind, nm, lat, lon, wd in pts:
        if kind in trees and len(trees[kind].query(shapely.Point(lon, lat), predicate="within")):
            continue
        out.append((kind, nm, lat, lon, 0.0, wd, ""))
    return pd.DataFrame(out, columns=["kind", "name", "lat", "lon", "m2", "wikidata", "wkt"])


def needs(city):
    """A tier 0 education cache without outlines."""
    p = os.path.join(TIER0, f"osm_education_{city}.csv.gz")
    return os.path.exists(p) and "wkt" not in pd.read_csv(p, nrows=0).columns


def registered():
    with open(REGISTRY, encoding="utf-8") as f:
        return [c["id"] for c in json.load(f)["cities"]]


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in ("fetch", "split", "status"):
        sys.exit(__doc__)
    ids = [x for x in args[1:] if not x.startswith("--")]
    if args[0] == "fetch":
        sys.exit(1 if fetch() else 0)
    ids = ids or [c for c in registered() if needs(c)]
    if args[0] == "split":
        split(ids)
    else:
        bands = sum(1 for w, e in BANDS for k in KINDS if overpass.cache_path(band_name(k, w, e)))
        print(f"{bands} of {len(BANDS) * len(KINDS)} bands fetched; {sum(1 for c in ids if overpass.cache_path(name(c)))} "
              f"of {len(ids)} packs split")
