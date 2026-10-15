"""T-292: India's inputs for a tier 0 city (Mumbai first), through tier0's hooks.

- **Outline and units**: OSM boundary=administrative relations read from the Geofabrik extract by
  name (districts at admin_level 5, talukas/subdistricts at 6), clipped to land by the OSM
  coastline polygons, cached as a small GeoJSON (the .pbf is deleted after routes.py).
- **Population**: Census 2011 Primary Census Abstract population by sub-district (UNITS below),
  grown to 2020 by each district's 2001-2011 growth (Census 2011 table A-2) carried forward,
  spread inside each unit by GHSL.
- **Curve**: Census 2011 table B-28, 'other workers' by distance from residence to place of work,
  for the districts the pack covers (B28 below), fitted in tier0.fit_curve_bands.

    python pipeline/india.py mum      (writes the units file and prints what it holds)
"""
import json
import os
import sys

import shapely
from shapely import make_valid, union_all
from shapely.geometry import mapping, shape

import build_city as b
import england

# OSM name -> (admin_level, label, 2011 district code, Census 2011 population). Source: Census 2011
# PCA, DDW_PCA2721..2724_2011_MDDS (censusindia.gov.in/nada catalog 6708-6711), sub-district rows;
# Mumbai City and Mumbai Suburban have no sub-districts in the PCA, so they are whole districts.
MUMBAI = {
    "Mumbai City District": ("5", "Mumbai City", "519", 3085411),
    "Mumbai Suburban District": ("5", "Mumbai Suburban", "518", 9356962),
    "Thane Subdistrict": ("6", "Thane", "517", 3787036),
    "Kalyan Subdistrict": ("6", "Kalyan", "517", 1565417),
    "Bhiwandi Subdistrict": ("6", "Bhiwandi", "517", 1141386),
    "Ulhasnagar Subdistrict": ("6", "Ulhasnagar", "517", 506098),
    "Ambernath Subdistrict": ("6", "Ambarnath", "517", 565340),
    "Murbad Subdistrict": ("6", "Murbad", "517", 190652),
    "Shahapur Subdistrict": ("6", "Shahapur", "517", 314103),
    "Vasai Subdistrict": ("6", "Vasai", "517", 1343402),
    "Palghar Subdistrict": ("6", "Palghar", "517", 550166),
    "Uran Subdistrict": ("6", "Uran", "520", 160303),
    "Panvel Subdistrict": ("6", "Panvel", "520", 750236),
    "Karjat Taluka": ("6", "Karjat", "520", 212051),
    "Khalapur Taluka": ("6", "Khalapur", "520", 207464),
    "Pen Taluka": ("6", "Pen", "520", 195454),
    "Alibag Taluka": ("6", "Alibag", "520", 236167),
}
UNITS = {"mum": MUMBAI}
# Units scaled as one: OSM's lines between them do not follow the 2011 talukas (GHSL in OSM's
# Ulhasnagar is 2.6 times its census population, Bhiwandi's half), while their sum is within 2%.
GROUPS = {"mum": {n: "Kalyan, Bhiwandi, Ulhasnagar, Ambarnath" for n in
                  ("Kalyan Subdistrict", "Bhiwandi Subdistrict", "Ulhasnagar Subdistrict", "Ambernath Subdistrict")}}
# District population 2001 and 2011 (2011 jurisdiction), Census 2011 table A-2 for Maharashtra
# (censusindia.gov.in/nada catalog 43360). Palghar's talukas were in Thane district until 2014.
A2 = {"517": (8131849, 11060148), "518": (8640419, 9356962), "519": (3338031, 3085411), "520": (2207929, 2634200)}
CENSUS_DATE, TARGET_YEAR = 2011.17, 2020.5  # census reference date 2011-03-01; GHSL E2020
# Census 2011 B-28 (DDW-2700B-28.xlsx, censusindia.gov.in/nada catalog 13981), 'other workers', all
# modes, persons, district Total rows: no travel, then the bands 0-1, 2-5, 6-10, 11-20, 21-30,
# 31-50, 51+ km. 'Distance not stated' left out.
B28_EDGES_KM = [0.0, 1.5, 5.5, 10.5, 20.5, 30.5, 50.5, 1e9]  # reported whole km
B28 = {
    "517": (650478, [604215, 682951, 429827, 270652, 251581, 373766, 222121]),
    "518": (649502, [593000, 677247, 619029, 455770, 247302, 161102, 43272]),
    "519": (293602, [215783, 229702, 229120, 127347, 46135, 28176, 10757]),
    "520": (164980, [103549, 127835, 87553, 51737, 30043, 36787, 24654]),
}
CENSUS_URL = "https://censusindia.gov.in/nada/index.php/catalog/6708"
B28_URL = "https://censusindia.gov.in/nada/index.php/catalog/13981"


def raw(name):
    return os.path.join(b.RAW, "tier0", "india", name)


def grown(code, pop2011):
    p01, p11 = A2[code]
    return pop2011 * (p11 / p01) ** ((TARGET_YEAR - CENSUS_DATE) / 10.0)


def b28_bands(city):
    """(edges in reported km, travellers per band, no-travel share) summed over the pack's districts."""
    codes = sorted({u[2] for u in UNITS[city].values()})
    counts = [sum(B28[c][1][i] for c in codes) for i in range(7)]
    none = sum(B28[c][0] for c in codes)
    return B28_EDGES_KM, counts, none / (none + sum(counts))


def units_file(city):
    """The units (OSM relations by name and admin level) clipped to land, as a GeoJSON with name,
    label, district, 2011 and 2020 population; its path."""
    extract, wanted = b.CITIES[city]["extract"], UNITS[city]
    out = raw(f"{city}_units.geojson")
    if os.path.exists(out):
        return out
    import pyogrio
    import routes
    import osm_water
    pbf = routes.download(extract)
    df = pyogrio.read_dataframe(pbf, layer="multipolygons", columns=["osm_id", "name", "admin_level", "boundary"],
                                where="boundary = 'administrative' AND admin_level IN ('5', '6')")
    geoms = {}
    for name, (lvl, label, _, _) in wanted.items():
        rows = df[(df.admin_level == lvl) & (df["name"] == name)]
        if len(rows) != 1:
            raise SystemExit(f"{city}: {name}: {len(rows)} OSM relations at admin_level {lvl}")
        geoms[name] = (make_valid(rows.geometry.iloc[0]).buffer(0), f"OSM r{rows.osm_id.iloc[0]}")
    w, s, e, n = union_all([g for g, _ in geoms.values()]).bounds
    sea = union_all(osm_water.sea((w - 0.01, s - 0.01, e + 0.01, n + 0.01)))
    feats = []
    for name, (lvl, label, code, pop) in wanted.items():
        g = geoms[name][0].difference(sea).buffer(0)
        if g.geom_type == "MultiPolygon":  # slivers left along the clipped shore
            g = shapely.MultiPolygon([p for p in g.geoms if p.area > 1e-6])
        feats.append({"type": "Feature", "properties": {"name": name, "label": label, "district": code, "pop2011": pop,
                                                        "pop": round(grown(code, pop)), "source": geoms[name][1]},
                      "geometry": mapping(g)})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    os.replace(out + ".part", out)
    return out


def units(city):
    with open(units_file(city), encoding="utf-8") as f:
        return [(ft["properties"], shape(ft["geometry"])) for ft in json.load(f)["features"]]


def boundary_file(city):
    """The units dissolved, for tier0's `boundary_file`; its path."""
    out = raw(f"{city}_outline.geojson")
    if not os.path.exists(out):
        g = union_all([u[1] for u in units(city)]).buffer(0)
        with open(out + ".part", "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"city": city},
                                                                   "geometry": mapping(g)}]}, f)
        os.replace(out + ".part", out)
    return out


def scaled_units(city):
    """The units with GROUPS merged: [(properties, geometry)], pop and pop2011 summed."""
    merged = {}
    for p, g in units(city):
        key = GROUPS.get(city, {}).get(p["name"], p["name"])
        if key in merged:
            q, h = merged[key]
            merged[key] = ({**q, "pop": q["pop"] + p["pop"], "pop2011": q["pop2011"] + p["pop2011"]}, union_all([h, g]).buffer(0))
        else:
            merged[key] = ({**p, "name": key, "label": key if key != p["name"] else p["label"]}, g)
    return list(merged.values())


def census_population(city, boundary, sig):
    """Census 2011 population by sub-district grown to 2020 by district growth, spread inside each
    unit by GHSL; tier0's `population` hook."""
    us = scaled_units(city)
    shapes = [(p["name"], g) for p, g in us]
    out = england.spread_totals(sig["lat"], sig["lon"], shapes, {p["name"]: p["pop"] for p, _ in us}, sig["pop_ghsl"])
    idx, _ = england.cell_units(sig["lat"], sig["lon"], shapes)
    for i, (p, _) in enumerate(us):
        have = sig["pop_ghsl"][idx == i].sum()
        print(f"  census {p['label']}: 2011 {p['pop2011']:,}, grown {p['pop']:,}; GHSL {have:,.0f} ({have / p['pop'] - 1:+.1%} on grown, "
              f"{have / p['pop2011'] - 1:+.1%} on 2011)", flush=True)
    g, c11 = float(sig["pop_ghsl"][idx >= 0].sum()), sum(p["pop2011"] for p, _ in us)
    print(f"  census by sub-district: {len(us)} units, 2011 {c11:,}, grown {out.sum():,.0f}; GHSL {g:,.0f} "
          f"({g / out.sum() - 1:+.1%} on grown, {g / c11 - 1:+.1%} on 2011)", flush=True)
    return out, ("Census 2011 population by sub-district (Primary Census Abstract), grown to 2020 by each district's "
                 "2001-2011 growth (table A-2), spread inside each by GHSL GHS-POP E2020")


if __name__ == "__main__":
    city = sys.argv[1]
    us = units(city)
    g = union_all([u[1] for u in us])
    km2 = 111.32 * 110.57 * 0.945  # degrees squared to km2 near 19 N
    print(f"{city}: {len(us)} units, 2011 {sum(p['pop2011'] for p, _ in us):,}, grown {sum(p['pop'] for p, _ in us):,}, "
          f"outline {g.geom_type}, {len(getattr(g, 'geoms', [g]))} parts, about {g.area * km2:,.0f} km2, "
          f"bounds {tuple(round(v, 3) for v in g.bounds)}")
    for p, geom in us:
        parts = list(getattr(geom, "geoms", [geom]))
        print(f"  {p['label']}: {len(parts)} parts, {sum(len(q.interiors) for q in parts)} holes, about {geom.area * km2:,.1f} km2")
    edges, counts, none = b28_bands(city)
    print(f"B-28: {sum(counts):,} travelling other workers, no travel {none:.1%}; shares "
          + " ".join(f"{c / sum(counts):.3f}" for c in counts))
