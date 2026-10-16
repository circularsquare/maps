"""T-142: medical visit destinations for a finished city pack, in place. `enrich_medical.py <city>`
(`--dry` prints the facilities without writing the pack).

Facilities are OpenStreetMap's (global): amenity=hospital|clinic|doctors|dentist, or the same as
healthcare=hospital|clinic|doctor|dentist, inside the pack's counties. Each gets a size in visits
a year, judgement scaled to national figures (notes/T-142.md):

    hospital  1,000 x beds   (T-387, the rule below)
    clinic    10,000         (a community health centre or urgent care site)
    doctors   4,000          (a practice)
    dentist   3,000

Hospital beds (T-387, the same everywhere; pooled numbers in `hospital_beds.json`, made by
`hospital_beds.py pool`): OSM `beds` where tagged; else k x the square root of the site's floor
(its buildings' footprint x levels, a pooled typical number of levels where a building has neither
levels nor height; or the site's area x a pooled floor per site square metre where that is more),
between 20 and 1,500; else, a hospital mapped as a point, the pooled median of tagged points.
The pack's hospitals are then scaled together to the World Bank's national beds per 1,000 people
(SH.MED.BEDS.ZS, the country's latest year) times the pack's population, and no site keeps more
than the cap (what it loses goes to the others in proportion).

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
BED_RANGE = (20.0, 1500.0)    # a floor-area estimate before the national scale
SNAP_M = 500.0
HEALTHCARE = {"hospital": "hospital", "clinic": "clinic", "doctor": "doctors", "dentist": "dentist"}
SUB = "t142b/"
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hospital_beds.json")


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


def facilities(city, boundary=None, keep=None):
    """The pack's medical facilities, one record per place, hospitals with their building floor
    parts. `boundary` filters; without it, `keep` (OSM ids from an audit) does, from the caches."""
    q = ""
    if boundary is not None:
        s, w, n, e = overpass.bbox_of(boundary)
        q = (f'[out:json][timeout:600];(nwr["amenity"~"^(hospital|clinic|doctors|dentist)$"]({s},{w},{n},{e});'
             f'nwr["healthcare"~"^(hospital|clinic|doctor|dentist)$"]({s},{w},{n},{e}););out geom tags;')
        shapely.prepare(boundary)  # T-425: an unprepared covers() is tens of ms on a long coastline
    data = overpass.fetch(f"{SUB}osm_medical_{city}", q)
    rows = []
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
        osm = el["type"][0] + str(el["id"])
        if (boundary is not None and not boundary.covers(pt)) or (boundary is None and osm not in keep):
            continue
        rows.append(dict(osm=osm, kind=kind, name=tags.get("name", ""), lat=pt.y, lon=pt.x,
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
    floor_area(city, [r for r in kept if r["kind"] == "hospital"])
    return kept, len(rows) - len(kept)


def levels_of(tags):
    """A building's levels from `building:levels`, else `height` / 3.5 m; None with neither."""
    m = re.match(r"\s*(\d+(\.\d+)?)", tags.get("building:levels", ""))
    if m and float(m.group(1)) > 0:
        return min(float(m.group(1)), 60.0)
    m = re.match(r"\s*(\d+(\.\d+)?)", tags.get("height", ""))
    if m and float(m.group(1)) > 0:
        return max(1.0, min(float(m.group(1)) / 3.5, 60.0))
    return None


def floor_area(city, hosp):
    """Each hospital site's OSM buildings inside its polygon (a hospital mapped as a building
    counts itself): r["floor_tagged_m2"] and r["footprint_tagged_m2"], floor (footprint x levels)
    and footprint of those with levels or height, and r["footprint_bare_m2"] of those with neither."""
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
        m2, lv = overpass.area_m2(poly), levels_of(el.get("tags", {}))
        for i in tree.query(poly.representative_point(), predicate="within"):
            s = sites[i]
            if lv is None:
                s["footprint_bare_m2"] = s.get("footprint_bare_m2", 0.0) + m2
            else:
                s["floor_tagged_m2"] = s.get("floor_tagged_m2", 0.0) + m2 * lv
                s["footprint_tagged_m2"] = s.get("footprint_tagged_m2", 0.0) + m2
            break


def floor_m2(r, levels):
    return r.get("floor_tagged_m2", 0.0) + levels * r.get("footprint_bare_m2", 0.0)


def load_table():
    with open(TABLE) as f:
        return json.load(f)


def country_of(city, header):
    """ISO alpha-2, the config's first: older tier 0 packs' headers lack `country`."""
    cfg = b.CITIES.get(city, {})
    if cfg.get("iso2"):
        return cfg["iso2"]
    if cfg.get("country") == "ca":
        return "CA"
    return header.get("country", "US")


def national_rate(table, country):
    """(beds per 1,000 people, year, where from): the country's World Bank figure, else the world's."""
    row = table["beds_per_1000"].get(country)
    if row:
        return row[1], row[0], country
    row = table["beds_per_1000"]["WLD"]
    return row[1], row[0], "world"


def size_m2(r, table):
    """The floor a hospital is judged by: its buildings' floor area, or its site's area times the
    pooled floor per site square metre where that is more (buildings are often mapped in part)."""
    return max(floor_m2(r, table["typical_levels"]), table["floor_per_site_m2"] * (r.get("ha") or 0.0) * 1e4)


def raw_beds(r, table):
    """A hospital's beds before the national scale, and where they came from."""
    if r.get("beds"):
        return r["beds"], "tag"
    s = size_m2(r, table)
    if s > 0:
        return float(np.clip(table["beds_per_sqrt_m2"] * s ** table["size_power"], *BED_RANGE)), "floor area"
    return table["median_beds_no_site"], "median"


def site_cap(total, table):
    return min(max(table["cap_share"] * total, table["cap_min_beds"]), table["cap_max_beds"])


def size_hospitals(hosp, pop, country, table):
    """Sets r["beds_used"] and r["beds_from"] on each hospital; returns the scale's figures."""
    if not hosp:
        return {}
    raw = np.array([raw_beds(r, table)[0] for r in hosp])
    rate, year, src = national_rate(table, country)
    target = rate * pop / 1000.0
    beds = raw * target / raw.sum()
    cap = site_cap(target, table)
    capped = np.zeros(len(beds), bool)
    for _ in range(50):
        over = (beds > cap) & ~capped
        if not over.any():
            break
        capped |= over
        beds[capped] = cap
        free = beds[~capped].sum()
        if free <= 0:
            break
        beds[~capped] *= (target - cap * capped.sum()) / free
    for r, x, b0 in zip(hosp, beds, raw):
        r["size_m2"] = size_m2(r, table)
        r["beds_raw"], r["beds_from"] = raw_beds(r, table)
        r["beds_used"] = float(x)
    return dict(beds_per_1000=rate, year=year, rate_from=src, population=float(pop), national_beds=float(target),
                raw_beds=float(raw.sum()), scale=float(target / raw.sum()), cap_beds=float(cap), capped_sites=int(capped.sum()),
                beds=float(beds.sum()))


def main(city="nyc", dry=False):
    cfg = b.CITIES[city]
    boundary = b.build_boundary(city, cfg)
    kept, merged = facilities(city, boundary)
    stem = os.path.join(b.PACKS, city)
    with open(stem + ".json") as f:
        header = json.load(f)
    with open(stem + ".bin", "rb") as f:
        buf = f.read()
    arr = {a["name"]: a for a in header["arrays"]}
    pop = np.frombuffer(buf, dtype="<f4", count=arr["pop"]["count"], offset=arr["pop"]["offset"]).sum(dtype=np.float64)
    table = load_table()
    hosp = [r for r in kept if r["kind"] == "hospital"]
    scale = size_hospitals(hosp, pop, country_of(city, header), table)
    for r in kept:
        r["size"] = VISITS_PER_BED * r["beds_used"] if r["kind"] == "hospital" else SIZE[r["kind"]]
    stats = dict(scale, **{k: table[k] for k in ("typical_levels", "floor_per_site_m2", "beds_per_sqrt_m2", "median_beds_no_site")},
                 merged=merged,
                 tagged_hospitals=sum(r["beds_from"] == "tag" for r in hosp),
                 from_floor_area=sum(r["beds_from"] == "floor area" for r in hosp),
                 from_median=sum(r["beds_from"] == "median" for r in hosp))
    by_kind = {k: dict(count=sum(r["kind"] == k for r in kept), size=sum(r["size"] for r in kept if r["kind"] == k))
               for k in ("hospital", "clinic", "doctors", "dentist")}
    print(city, json.dumps(stats), json.dumps(by_kind), flush=True)
    if dry:
        for r in sorted(kept, key=lambda r: -r["size"])[:25]:
            print(f"  {r['kind']:8} {r['size']:>9,.0f} {r.get('beds_from', ''):10} {r['name'][:60]}")
        return
    a = arr["h3"]
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
    rate_src = "World Bank SH.MED.BEDS.ZS (hospital beds per 1,000 people)"
    header["medical"] = dict(
        coverage="OpenStreetMap hospitals, clinics, doctors and dentists in the pack's counties; patients and visitors",
        facilities=len(audit), by_kind=by_kind, size_total=float(visits.sum(dtype=np.float64)),
        located_share=float(visits.sum(dtype=np.float64) / total), omitted=len(omitted),
        sizes="hospital 1,000 x beds (tag, else a pooled k x the square root of the site's floor area, "
              "else a pooled median; all scaled to the national beds per 1,000 x the pack's population, one site at most "
              "the cap); clinic 10,000; doctors 4,000; dentist 3,000 (T-387)",
        beds=stats, rate_per_resident=0.04, decay_m=5000,
        limitations="OSM coverage of doctors and dentists is uneven; sizes are judgement, not visit counts",
        sources=["OpenStreetMap amenity=hospital|clinic|doctors|dentist, healthcare=* (ODbL), Overpass " +
                 hashlib.sha256(open(os.path.join(b.RAW, f"{SUB}osm_medical_{city}.json"), "rb").read()).hexdigest()[:16],
                 f"{rate_src}, {stats.get('rate_from')} {stats.get('year')} (CC BY 4.0)"])
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
