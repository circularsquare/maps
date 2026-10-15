"""T-142: medical visit destinations for a finished city pack, in place. `enrich_medical.py <city>`
(`--dry` prints the facilities without writing the pack).

Facilities are OpenStreetMap's (global): amenity=hospital|clinic|doctors|dentist, or the same as
healthcare=hospital|clinic|doctor|dentist, inside the pack's counties. Each gets a size in visits
a year, judgement scaled to national figures (notes/T-142.md):

    hospital  1,000 x beds   (OSM `beds`; else estimated from the floor area of the OSM buildings
                              on its site, footprint x levels, by the pack's own tagged hospitals'
                              beds per square metre of floor; else their median)
    clinic    10,000         (a community health centre or urgent care site)
    doctors   4,000          (a practice)
    dentist   3,000

A hospital mapped twice (a building or node inside a campus polygon) counts once, at the campus,
with the larger bed count; anything else mapped as a point inside a polygon of its own kind too.
The pack array `medical_visits` is these sizes per cell: relative draw, not a capacity. Trips
come from residents in the kernel (`everyday.rs`, kind 5: a flat rate per resident, destinations
by size and a 5 km distance decay). Patients and visitors; staff are already commuters.
"""
import hashlib
import json
import os
import re
import sys

import h3
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Point

import build_city as b
import education_sources as education
import enrich_airports as air
import overpass

SIZE = {"clinic": 10000.0, "doctors": 4000.0, "dentist": 3000.0}
VISITS_PER_BED = 1000.0
DEFAULT_BEDS_PER_M2 = 1 / 200.0   # if a pack has under ten tagged hospitals with buildings
DEFAULT_LEVELS = 3.0              # a building with neither levels nor height
BED_RANGE = (20.0, 1500.0)
SNAP_M = 500.0
HEALTHCARE = {"hospital": "hospital", "clinic": "clinic", "doctor": "doctors", "dentist": "dentist"}
SUB = "t142b/"


def kind_of(tags):
    a = tags.get("amenity")
    if a in ("hospital", "clinic", "doctors", "dentist"):
        return a
    return HEALTHCARE.get(tags.get("healthcare"))


def beds_of(tags):
    for k in ("beds", "capacity:beds", "healthcare:beds"):
        m = re.match(r"\s*(\d+)", tags.get(k, ""))
        if m and int(m.group(1)) > 0:
            return float(m.group(1))
    return None


def facilities(city, boundary):
    s, w, n, e = overpass.bbox_of(boundary)
    data = overpass.fetch(f"{SUB}osm_medical_{city}",
                          f'[out:json][timeout:600];(nwr["amenity"~"^(hospital|clinic|doctors|dentist)$"]({s},{w},{n},{e});'
                          f'nwr["healthcare"~"^(hospital|clinic|doctor|dentist)$"]({s},{w},{n},{e}););out geom tags;')
    rows = []
    shapely.prepare(boundary)  # T-425: an unprepared covers() is tens of ms on a long coastline
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        kind = kind_of(tags)
        if not kind or tags.get("disused") == "yes" or "disused:amenity" in tags:
            continue
        poly = overpass.polygon(el) if el["type"] != "node" else None
        if el["type"] == "node":
            pt = Point(el["lon"], el["lat"])
        elif poly is not None:
            pt = poly.representative_point()
        else:
            continue
        if not boundary.covers(pt):
            continue
        rows.append(dict(osm=el["type"][0] + str(el["id"]), kind=kind, name=tags.get("name", ""), lat=pt.y, lon=pt.x,
                         poly=poly, ha=overpass.area_m2(poly) / 1e4 if poly is not None else None, beds=beds_of(tags)))
    # one record per place: a point or smaller polygon inside a polygon of its own kind merges into it
    polys = sorted([r for r in rows if r["poly"] is not None], key=lambda r: -r["ha"])
    tree = STRtree([r["poly"] for r in polys])
    for r in rows:
        # biggest container first (polys are sorted by area, largest first)
        for i in sorted(tree.query(Point(r["lon"], r["lat"]), predicate="within")):
            big = polys[i]
            if big is r or big["kind"] != r["kind"] or (r["poly"] is not None and r["ha"] >= big["ha"]) or big.get("merged_into"):
                continue
            r["merged_into"] = big["osm"]
            if r["beds"] and (not big["beds"] or r["beds"] > big["beds"]):
                big["beds"] = r["beds"]
            break
    kept = [r for r in rows if not r.get("merged_into")]
    hosp = [r for r in kept if r["kind"] == "hospital"]
    floor_area(city, boundary, hosp)
    pairs = [r["beds"] / r["floor_m2"] for r in hosp if r["beds"] and r.get("floor_m2", 0) > 1000]
    per_m2 = float(np.median(pairs)) if len(pairs) >= 10 else DEFAULT_BEDS_PER_M2
    tagged = [r["beds"] for r in hosp if r["beds"]]
    median_beds = float(np.median(tagged)) if tagged else 150.0
    for r in kept:
        if r["kind"] == "hospital":
            if r["beds"]:
                r["beds_used"], r["beds_from"] = r["beds"], "tag"
            elif r.get("floor_m2", 0) > 0:
                r["beds_used"], r["beds_from"] = float(np.clip(per_m2 * r["floor_m2"], *BED_RANGE)), "floor area"
            else:
                r["beds_used"], r["beds_from"] = median_beds, "median"
            r["size"] = VISITS_PER_BED * r["beds_used"]
        else:
            r["size"] = SIZE[r["kind"]]
    stats = dict(beds_per_floor_m2=per_m2, beds_per_floor_m2_pairs=len(pairs), median_tagged_beds=median_beds,
                 tagged_hospitals=len(tagged), merged=len(rows) - len(kept),
                 from_floor_area=sum(r.get("beds_from") == "floor area" for r in hosp),
                 from_median=sum(r.get("beds_from") == "median" for r in hosp))
    return kept, stats


def levels_of(tags):
    m = re.match(r"\s*(\d+(\.\d+)?)", tags.get("building:levels", ""))
    if m and float(m.group(1)) > 0:
        return min(float(m.group(1)), 60.0)
    m = re.match(r"\s*(\d+(\.\d+)?)", tags.get("height", ""))
    if m and float(m.group(1)) > 0:
        return max(1.0, min(float(m.group(1)) / 3.5, 60.0))
    return DEFAULT_LEVELS


def floor_area(city, boundary, hosp):
    """Each hospital site's building floor area (footprint x levels of the OSM buildings inside
    its polygon; a hospital mapped as a building counts itself). Sets r["floor_m2"]."""
    sites = [r for r in hosp if r["poly"] is not None]
    if not sites:
        return
    # buildings in each site's bounding box (a map_to_area query over the whole pack timed out
    # on the public servers); only those inside a site polygon count below
    boxes = "".join(f'way["building"]({b[1]:.5f},{b[0]:.5f},{b[3]:.5f},{b[2]:.5f});' for b in (r["poly"].bounds for r in sites))
    data = overpass.fetch(f"{SUB}osm_hospital_buildings_{city}", f'[out:json][timeout:600];({boxes});out geom tags;')
    tree = STRtree([r["poly"] for r in sites])
    seen = set()
    for el in data.get("elements", []):
        key = (el["type"], el["id"])
        if key in seen:
            continue
        seen.add(key)
        poly = overpass.polygon(el)
        if poly is None:
            continue
        m2 = overpass.area_m2(poly) * levels_of(el.get("tags", {}))
        for i in tree.query(poly.representative_point(), predicate="within"):
            sites[i]["floor_m2"] = sites[i].get("floor_m2", 0.0) + m2
            break


def main(city="nyc", dry=False):
    cfg = b.CITIES[city]
    boundary = b.build_boundary(city, cfg)
    kept, stats = facilities(city, boundary)
    by_kind = {k: dict(count=sum(r["kind"] == k for r in kept), size=sum(r["size"] for r in kept if r["kind"] == k))
               for k in ("hospital", "clinic", "doctors", "dentist")}
    print(city, json.dumps(stats), json.dumps(by_kind), flush=True)
    if dry:
        for r in sorted(kept, key=lambda r: -r["size"])[:25]:
            print(f"  {r['kind']:8} {r['size']:>9,.0f} {r.get('beds_from', ''):6} {r['name'][:60]}")
        return
    stem = os.path.join(b.PACKS, city)
    with open(stem + ".json") as f:
        header = json.load(f)
    with open(stem + ".bin", "rb") as f:
        buf = f.read()
    a = next(a for a in header["arrays"] if a["name"] == "h3")
    cells = np.frombuffer(buf, dtype="<u8", count=a["count"], offset=a["offset"])
    index = {int(c): i for i, c in enumerate(cells)}
    visits = np.zeros(len(cells), dtype="<f4")
    audit, omitted = [], []
    for r in kept:
        cell, snap = air.cell_of(index, r["lat"], r["lon"], SNAP_M)
        rec = {k: v for k, v in r.items() if k != "poly"}
        if cell is None:
            omitted.append(rec)
            continue
        visits[cell] += r["size"]
        audit.append(rec | dict(cell=int(cell), snap_m=snap))
    total = sum(r["size"] for r in kept)
    if total <= 0 or visits.sum(dtype=np.float64) / total < 0.95:
        raise ValueError("medical location coverage below 95%")
    buf = education.replace_cell_arrays(header, buf, {"medical_visits": visits})
    header["medical"] = dict(
        coverage="OpenStreetMap hospitals, clinics, doctors and dentists in the pack's counties; patients and visitors",
        facilities=len(audit), by_kind=by_kind, size_total=float(visits.sum(dtype=np.float64)),
        located_share=float(visits.sum(dtype=np.float64) / total), omitted=len(omitted),
        sizes="hospital 1,000 x beds (tag, else the site's building floor area x the pack's tagged beds per floor square metre, else their median); clinic 10,000; doctors 4,000; dentist 3,000",
        beds=stats, rate_per_resident=0.04, decay_m=5000,
        limitations="OSM coverage of doctors and dentists is uneven; sizes are judgement, not visit counts",
        sources=["OpenStreetMap amenity=hospital|clinic|doctors|dentist, healthcare=* (ODbL), Overpass " +
                 hashlib.sha256(open(os.path.join(b.RAW, f"{SUB}osm_medical_{city}.json"), "rb").read()).hexdigest()[:16]])
    with open(stem + ".bin.part", "wb") as f:
        f.write(buf)
    with open(stem + ".json.part", "w") as f:
        json.dump(header, f, indent=2)
    os.replace(stem + ".bin.part", stem + ".bin")
    os.replace(stem + ".json.part", stem + ".json")
    with open(os.path.join(b.WORK, city + ".medical.audit.json"), "w") as f:
        json.dump(dict(metadata=header["medical"], facilities=audit, omitted=omitted), f, indent=1)
    print(f"{city}: {len(audit)} facilities, {len(omitted)} omitted", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*args[:1], dry="--dry" in sys.argv)
