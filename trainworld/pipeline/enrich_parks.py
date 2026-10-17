"""T-196: parks as social/leisure destinations, added to a finished city pack in place.

    python pipeline/enrich_parks.py <city> [--dry]

Places are OpenStreetMap's (global), inside the pack's boundary:

    leisure=park, common, recreation_ground, beach_resort        factor 1
    leisure=garden with garden:type=botanical                     factor 1
    natural=beach with a name                                     factor 1
    leisure=nature_reserve named "... Park", "Recreation Area"    factor 1 (counted as a park)
    leisure=nature_reserve, any other name                        factor 0.25

Left out: playgrounds, dog parks and pitches (mostly inside parks, and a trip to one is a short
walk the model drops under 150 m anyway), golf courses, cemeteries, private gardens, unnamed
reserves and anything tagged access=private|no. Other reserves count at a quarter: in New England
most are land-trust, watershed or wetland parcels with few visitors (Hartford maps 1,300 of them),
while the visited ones are usually also mapped as parks or named as one (a feature inside a larger
kept one merges into it, so nothing counts twice).

Each place's draw:

    factor x sqrt(area in ha, at least 0.1) x D / (D + 2,000)

D is the residents (the pack's `pop`) per km² of land within 2 km of the place's edge, the place
itself left out. The square root gives the diminishing return Anita asked for (a park 100 times
bigger draws 10 times as much); the population term discounts parks few people live near, and
stops growing once a park is in built-up surroundings: the gravity already sends more trips to
parks with more homes around them, so the term does not multiply them a second time. A density,
not a count: a count within 2 km grows with a park's edge, so a 19,000 ha state forest collected
40,000 "neighbours" and outdrew Central Park five times. The constants are judgement
(notes/T-196.md).

The draw goes to the pack cells whose centres lie inside the place, split evenly; a place with
none is put on the cell under its representative point, or the nearest cell within 500 m. The pack
array `park_draw` is relative draw per cell, not a visitor count: the kernel (`everyday.rs`) mixes
it into the social/leisure destinations at PARK_SHARE of the attraction, so parks take a share of
the outings residents already make; they add none.
"""
import hashlib
import json
import os
import re
import sys
import time

import h3
import numpy as np
import shapely
from shapely import STRtree, points
from shapely.geometry import Point
from shapely.ops import transform

import build_city as b
import education_sources as education
import enrich_airports as air
import overpass
from read_pack import load_pack

SUB = "t196/"
KINDS = {"park": 1.0, "common": 1.0, "recreation_ground": 1.0, "beach_resort": 1.0, "garden": 1.0,
         "beach": 1.0, "nature_reserve": 0.25}
VISITED = re.compile(r"\b(park|recreation area|beach)\b", re.I)
MIN_HA = 0.1
NODE_HA = 0.25          # a park mapped as a point only
NEAR_M = 2000.0         # residents within this distance of the place's edge
DENSITY_HALF = 2000.0   # ... per km² of that ring's land, at which the population term is one half
SNAP_M = 500.0
TILE_DEG = 0.5          # Overpass request tile, degrees


def kind_of(tags):
    if tags.get("access") in ("private", "no") or "disused:leisure" in tags:
        return None
    lei = tags.get("leisure")
    if lei in ("park", "common", "recreation_ground", "beach_resort"):
        return lei
    if lei == "garden":
        return "garden" if tags.get("garden:type") == "botanical" else None
    if lei == "nature_reserve":
        name = tags.get("name", "")
        if not name:
            return None
        # a reserve named as a park is one people visit (Hartford's Keney and Goodwin Parks,
        # Osbornedale State Park are mapped as reserves); English names only for now
        return "park" if VISITED.search(name) else "nature_reserve"
    if tags.get("natural") == "beach" and tags.get("name"):
        return "beach"
    return None


def query(s, w, n, e, rel=False):
    """Ways and nodes (`out geom tags` gives a way's geometry but no relation members), or the
    multipolygon relations with their members' geometry (`out geom`)."""
    bb = f"({s},{w},{n},{e})"
    if rel:
        return (f'[out:json][timeout:900];(rel["leisure"~"^(park|common|recreation_ground|beach_resort|nature_reserve)$"]{bb};'
                f'rel["leisure"="garden"]["garden:type"="botanical"]{bb};rel["natural"="beach"]["name"]{bb};);out geom;')
    return (f'[out:json][timeout:900];(nwr["leisure"~"^(park|common|recreation_ground|beach_resort|nature_reserve)$"]{bb};'
            f'wr["leisure"="garden"]["garden:type"="botanical"]{bb};wr["natural"="beach"]["name"]{bb};);out geom tags;')


def fetch_all(city, boundary):
    """Relations first (with members), then ways and nodes; elements on two tiles kept once.
    Returns (elements, cache files)."""
    rels, rn = fetch_tiles(city, boundary, True)
    rest, wn = fetch_tiles(city, boundary, False)
    seen, out = set(), []
    for el in rels + rest:
        if (el["type"], el["id"]) not in seen:
            seen.add((el["type"], el["id"]))
            out.append(el)
    return out, rn + wn


def fetch_tiles(city, boundary, rel):
    """One kind of the pack's places from Overpass in tiles of at most TILE_DEG (the public servers
    time out on a whole metro with geometry), the tiles that miss the boundary skipped."""
    from shapely.geometry import box
    shapely.prepare(boundary)
    s, w, n, e = overpass.bbox_of(boundary)
    ny, nx = int(np.ceil((n - s) / TILE_DEG)), int(np.ceil((e - w) / TILE_DEG))
    stem = f"{SUB}osm_park{'rels' if rel else 's'}_{city}"
    if ny * nx == 1:
        return overpass.fetch(stem, query(s, w, n, e, rel)).get("elements", []), [stem]
    tiles = []
    for i in range(ny):
        for j in range(nx):
            ts, tw = round(s + i * (n - s) / ny, 4), round(w + j * (e - w) / nx, 4)
            tn, te = round(s + (i + 1) * (n - s) / ny, 4), round(w + (j + 1) * (e - w) / nx, 4)
            if boundary.intersects(box(tw, ts, te, tn)):
                tiles.append((f"{stem}_{i}_{j}", (ts, tw, tn, te)))

    def get(name, bb, depth=0):
        """A tile's answers (and cache names); a tile every server gives up on is split in four."""
        try:
            return [overpass.fetch(name, query(*bb, rel))], [name]
        except SystemExit:
            if depth >= 2:
                raise
        ts, tw, tn, te = bb
        ym, xm = round((ts + tn) / 2, 4), round((tw + te) / 2, 4)
        datas, names = [], []
        for k, sub in enumerate([(ts, tw, ym, xm), (ts, xm, ym, te), (ym, tw, tn, xm), (ym, xm, tn, te)]):
            if boundary.intersects(box(sub[1], sub[0], sub[3], sub[2])):
                d, n_ = get(f"{name}_{k}", sub, depth + 1)
                datas += d
                names += n_
        return datas, names

    # three requests at a time (each tile is its own cache file)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as pool:
        answers = list(pool.map(lambda t: get(*t), tiles))
    seen, out, names = set(), [], []
    for datas, nm in answers:
        names += nm
        for data in datas:
            for el in data.get("elements", []):
                if (el["type"], el["id"]) not in seen:
                    seen.add((el["type"], el["id"]))
                    out.append(el)
    return out, names


def places(city, boundary):
    elements, names = fetch_all(city, boundary)
    shapely.prepare(boundary)  # T-420: 62 ms a covers() on Greater Tokyo's outline unprepared
    rows = []
    for el in elements:
        tags = el.get("tags", {})
        kind = kind_of(tags)
        if not kind:
            continue
        if el["type"] == "node":
            if kind != "park":
                continue
            poly, pt = None, Point(el["lon"], el["lat"])
        else:
            poly = overpass.polygon(el)
            if poly is None:
                continue
            pt = poly.representative_point()
        if not boundary.covers(pt):
            continue
        rows.append(dict(osm=el["type"][0] + str(el["id"]), kind=kind, name=tags.get("name", ""), lat=pt.y, lon=pt.x,
                         poly=poly, ha=overpass.area_m2(poly) / 1e4 if poly is not None else NODE_HA))
    # a place inside a larger kept one (a lawn inside Central Park, a park inside a reserve) merges into it
    polys = sorted([r for r in rows if r["poly"] is not None], key=lambda r: -r["ha"])
    tree = STRtree([r["poly"] for r in polys])
    for r in rows:
        for i in sorted(tree.query(Point(r["lon"], r["lat"]), predicate="within")):
            big = polys[i]
            if big is r or big["ha"] <= r["ha"] or big.get("merged_into"):
                continue
            r["merged_into"] = big["osm"]
            break
    return [r for r in rows if not r.get("merged_into")], len(rows), names


def main(city="nyc", dry=False):
    t0 = time.time()
    cfg = b.CITIES[city]
    boundary = b.build_boundary(city, cfg)
    kept, found, names = places(city, boundary)
    header, a = load_pack(city)
    lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
    k = np.cos(np.radians(lat0))

    def xy(lon, lat):
        return (np.radians(np.asarray(lon) - lon0) * b.R_EARTH * k, np.radians(np.asarray(lat) - lat0) * b.R_EARTH)

    cells = a["h3"]
    ll = np.array([h3.cell_to_latlng(h3.int_to_str(int(c))) for c in cells])
    cx, cy = xy(ll[:, 1], ll[:, 0])
    pop = a["pop"].astype(np.float64)
    ctree = STRtree(points(cx, cy))
    # each place in metres: its polygon (a point park: a disc of its default area)
    geoms = []
    for r in kept:
        if r["poly"] is not None:
            geoms.append(transform(lambda x, y: xy(x, y), r["poly"]))
        else:
            px, py = xy(r["lon"], r["lat"])
            geoms.append(Point(float(px), float(py)).buffer(np.sqrt(NODE_HA * 1e4 / np.pi)))
    near = np.array([g.buffer(NEAR_M, 8) for g in geoms], dtype=object)
    pi, ci = ctree.query(near, predicate="contains")
    P = np.bincount(pi, weights=pop[ci], minlength=len(kept))
    # the density of the ring around the place: its residents over the ring's land (the ring
    # without the place itself, inside the pack's boundary, whose county shapes stop at the shore)
    land = transform(lambda x, y: xy(x, y), boundary)
    near_a, geoms_a = near, np.array(geoms, dtype=object)
    ring_m2 = np.empty(len(kept))
    for i in range(0, len(kept), 1000):
        s = slice(i, i + 1000)
        # T-425: only the land near the ring matters; an overlay with the whole outline costs tens of
        # ms a park on a long coastline, a rectangle clip about 4 ms. The 1 km margin keeps the clip's
        # cut edges clear of the ring, which makes the areas bit-identical to the unclipped overlay.
        box_land = np.array([shapely.clip_by_rect(land, g.bounds[0] - 1000, g.bounds[1] - 1000, g.bounds[2] + 1000, g.bounds[3] + 1000)
                             for g in near[s]], dtype=object)
        ring_m2[s] = shapely.area(shapely.intersection(shapely.difference(near_a[s], geoms_a[s]), box_land))
    ring_km2 = np.maximum(ring_m2 / 1e6, 0.5)
    D = P / ring_km2
    draw = np.zeros(len(cells), np.float64)
    ii, cc = ctree.query(geoms_a, predicate="contains")
    inside = {}
    for i, c in zip(ii, cc):
        inside.setdefault(int(i), []).append(int(c))
    index = {int(c): i for i, c in enumerate(cells)}
    audit, omitted = [], []
    for i, r in enumerate(kept):
        w = KINDS[r["kind"]] * np.sqrt(max(r["ha"], MIN_HA)) * D[i] / (D[i] + DENSITY_HALF)
        rec = {x: v for x, v in r.items() if x != "poly"} | dict(pop_near=float(P[i]), density_near=float(D[i]), draw=float(w))
        cs = inside.get(i)
        if cs:
            draw[cs] += w / len(cs)
            rec["cells"] = len(cs)
        else:
            cell, snap = air.cell_of(index, r["lat"], r["lon"], SNAP_M)
            if cell is None:
                omitted.append(rec)
                continue
            draw[cell] += w
            rec |= dict(cells=1, snap_m=snap)
        audit.append(rec)
    total = sum(r["draw"] for r in audit) + sum(r["draw"] for r in omitted)
    by_kind = {kd: dict(count=sum(r["kind"] == kd for r in audit), ha=round(sum(r["ha"] for r in audit if r["kind"] == kd), 1),
                        draw=round(sum(r["draw"] for r in audit if r["kind"] == kd), 1)) for kd in KINDS}
    stats = dict(found=found, kept=len(kept), located=len(audit), omitted=len(omitted),
                 located_share=float(draw.sum() / total) if total else 0.0, seconds=round(time.time() - t0, 1))
    print(city, json.dumps(stats), json.dumps(by_kind), flush=True)
    top = sorted(audit, key=lambda r: -r["draw"])
    for r in top[:12]:
        print(f"  {r['kind']:15} {r['ha']:9.1f} ha  pop near {r['pop_near']:>10,.0f} ({r['density_near']:>6,.0f}/km2)  draw {r['draw']:7.1f}  {r['name'][:50]}")
    if dry:
        return
    # T-579: a pack with no OSM parks (or none with people near) keeps a zero park_draw; the
    # kernel then leaves parks out of its social attraction, and the gate judges the pack
    if total > 0 and draw.sum() / total < 0.95:
        raise ValueError("park location coverage below 95%")
    stem = os.path.join(b.PACKS, city)
    with open(stem + ".json") as f:
        header = json.load(f)
    with open(stem + ".bin", "rb") as f:
        buf = f.read()
    buf = education.replace_cell_arrays(header, buf, {"park_draw": draw.astype("<f4")})
    header["parks"] = dict(
        coverage="OpenStreetMap parks, commons, recreation grounds, botanical gardens, named beaches, reserves named as parks, and (at a quarter) other named nature reserves in the pack's boundary",
        places=len(audit), omitted=len(omitted), by_kind=by_kind, draw_total=float(draw.sum()),
        draw="factor x sqrt(max(area ha, 0.1)) x D / (D + 2000), D = residents per km2 of land within 2 km of the edge; split over the cells inside",
        limitations="relative draw, not visitor counts; no season, opening hours or amenities",
        sources=["OpenStreetMap leisure=park|common|recreation_ground|beach_resort|nature_reserve, botanical gardens, named beaches (ODbL), Overpass " +
                 hashlib.sha256(b"".join(open(os.path.join(b.RAW, nm + ".json"), "rb").read() for nm in names)).hexdigest()[:16]])
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    with open(os.path.join(b.WORK, city + ".parks.audit.json"), "w") as f:
        json.dump(dict(metadata=header["parks"], places=audit, omitted=omitted), f, indent=1)
    print(f"{city}: {len(audit)} places, {len(omitted)} omitted -> park_draw", flush=True)


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(*args[:1], dry="--dry" in sys.argv)
