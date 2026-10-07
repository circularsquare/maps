"""Kuwait: the 2021 census's 157 areas on OpenStreetMap's area polygons.

Writes data/geo/kw/kw_units.gpkg and data/geo/kw/kw_lookup.csv.

  * **boundaries**: OpenStreetMap's `boundary=administrative` relations in Kuwait at
    `admin_level=6` (the areas, 192 relations; 169 of them tagged `source=www.q8maps.com`) and
    `admin_level=4` (the six governorates), through Overpass. ODbL. geoBoundaries' KWT ADM2 is
    OpenStreetMap as of 2011 (137 areas) and lacks the cities built since (Sabah Al-Ahmad, Jaber
    Al-Ahmad, Abdullah Al-Mubarak); there is no COD-AB below the governorate.
  * **the areas**: Table 52 of the 2021 census (`sources/kw.py`), which prints each area's name
    in Arabic and English.

## THE JOIN

On the Arabic name, after folding hamza forms, taa marbuta, alif maqsura, diacritics and spaces
(`fold`), which matches 115 of the 157 areas to exactly one polygon. The other 42 are named in
`BY_HAND`, area by area, to OSM's English names. Where two census areas share one polygon (Sabah
Al-Ahmad's five census "cities" are one OSM area; Anjafa sits in Coast Strip A with Al-Bida;
Al-Misila in Al Masayel; Mubarakiya Camps beside Shuwaikh Industrial 3, where OSM draws no area),
or one area takes several polygons (Shuwaikh Industrial 1-3; each governorate's desert), the
areas and polygons that touch form one drawn unit (union-find over the pairs).

Each governorate's desert row (Ahmadi 1,475 people, Jahra 2,669) takes every rural polygon of its
governorate that no other area claims (`AHMADI_DESERT`, `JAHRA_DESERT`); Kontur decides where in
them people are.

## CHECKS

  1. 192 area relations and 6 governorates, every one with a polygon;
  2. every census area joins, and every polygon named in `BY_HAND` exists with the pinned count;
  3. a name-matched area lies in its census governorate (the governorate is read off Table 52's
     order in `sources/kw.py`), except `GOV_PINNED`, two places with under 20 people;
  4. drawn units do not overlap by more than `MAX_OVERLAP_KM2`, once nested units are cut out;
  5. **witness**: the OSM areas that carry a `population` tag (most cite PACI, 2020 to 2024) are
     within `TAG_BAND` of the census area joined to them, which neither key decides;
  6. polygons no census area claims are printed with their size; Kontur's people in them are
     printed by `sources/kw_grid.py`.

Usage:
    python sources/kw_geo.py --fetch    two Overpass queries (2 MB)
    python sources/kw_geo.py            rebuild from data/raw/kw/
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "6")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "kw")
OUT_DIR = os.path.join(ROOT, "data", "geo", "kw")
OUT_UNITS = os.path.join(OUT_DIR, "kw_units.gpkg")
LOOKUP = os.path.join(OUT_DIR, "kw_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "kw.csv")
OSM_TAGS = os.path.join(RAW, "osm_kw_admin_tags.json")
OSM_GEOM = os.path.join(RAW, "osm_kw_admin_geom.json")

OVERPASS = "https://overpass-api.de/api/interpreter"
UA = "religiondots-map-build/1.0"
Q_TAGS = ('[out:json][timeout:120];area["ISO3166-1"="KW"][admin_level=2]->.a;'
          'rel(area.a)[boundary=administrative];out tags;')
Q_GEOM = "[out:json][timeout:240];rel(id:{ids});out geom;"

EXPECTED_AREAS, EXPECTED_GOVS = 192, 6
METRIC = "EPSG:32638"
MAX_OVERLAP_KM2 = 0.05
TAG_BAND = (0.7, 1.5)      # OSM population tag over the 2021 census, per area
TAG_MIN = 10

OSM_GOV = {"Capital": "Capital Governorate", "Hawalli": "Hawalli Governorate",
           "Ahmadi": "Ahmadi Governorate", "Jahra": "Jahra Governorate",
           "Farwaniya": "Farwaniya Governorate",
           "Mubarak Al-Kabeer": "Mubarak al-Kabir Governorate"}
# matched by name, and in another governorate in OSM: 16 and 8 people
GOV_PINNED = {"AL-DOHA CHALETS", "AL-SULAIBIYA INDUSTRAIL 3"}

AHMADI_DESERT = ["Um Hegoul", "Abu Khurjeen - Al Subayhiyah", "Qaradan - Al Hafirah - Al Fawwar",
                 "PAFER - Farms", "Al-Nuwaiseeb", "South of Sabah Al-Ahmad City"]
JAHRA_DESERT = ["Abdily - Sekherbriyat", "Al Metlaa - Jal Al Atraf", "Al Retqah - Al Heraigah",
                "As Syer - Um Al Medfaa", "Al Abaireq - Al Anaem - Al Laiyah",
                "Al Shejaya - Al Dubdiba - Al Mityahah", "Qulmat Shayea-Al Manaqeesh",
                "Kabd - Shiqq - Dabah", "Al Abdiliya", "Um Qudeer", "Sabryia", "Bhaith",
                "Ar Ruhayya - Um Tawaynij", "Rural Areas - Al-Jahra",
                "South of Saad Al-Abdullah City"]

# census area (Table 52's English) -> OSM name:en, for the areas the Arabic name does not match
BY_HAND = {
    "AL-SHARQ": ["Sharq"],
    "AL-SHUWAIKH INDUSTRIAL": ["Shuwaikh Industrial 1", "Shuwaikh Industrial 2",
                               "Shuwaikh Industrial 3"],
    # OSM draws no area where geoBoundaries' 2011 Mubarakiya_Camps was (4.1 km2, 0.13 of it in
    # Shuwaikh Industrial 3, the rest in no area); the nearest area
    "AL-MUBARAKIYA CAMPS": ["Shuwaikh Industrial 3"],
    "AL-SULAIBIKHAT": ["Sulaibikhat"],
    "AL-DOHA RESIDENTIAL": ["Doha"],
    "NORTH WEST SULIBIKHAT": ["Northwest Sulaibikhat"],
    "JABER AL-AHMED CITY": ["Jaber Al-Ahmad"],
    "MINISTRIES AREA": ["Ministries Area"],
    "ANJAFA": ["Coast Strip A"],
    # geoBoundaries' 2011 Elahmdi (23.7 km2) is 19.0 km2 of OSM's East Ahmadi (25.1)
    "AL-AHMADI CITY": ["East Ahmadi"],
    "AL-SHUAIBA INDUSTRIAL": ["Shuaiba Industrial Western"],
    "ABDULLAH PORT": ["Mina Abdulla"],
    # geoBoundaries' 2011 El_Kehran_Loloet_Elkheran is 55.4 of its 62.4 km2 in this polygon
    "AL-KHAIRAN CHALETS": ["Sabah Al Ahmad Marine City"],
    "SABAH AL-AHMAD SEA CITY": ["Sabah Al Ahmad Marine City"],
    "AL-ZOOR": ["Az Zour - Sulah"],
    "AL-WAFRA": ["Wafra Residential"],
    "AL-AHMADI GOVERNORATE DESERT": AHMADI_DESERT,
    "SOUTH JAWAKHER": AHMADI_DESERT,
    "RAJM KHASHMAN": AHMADI_DESERT,
    "KABAD AGRICULTURAL": AHMADI_DESERT,
    "FAHAD AL-AHMAD AL-JABER": ["Fahad Al-Ahmad"],
    "ALI SABAH AL-SALIM": ["Umm Al Hayman"],
    "SABAH AL-AHMAD CITY 1": ["Sabah Al-Ahmad"],
    "SABAH AL-AHMAD CITY 2": ["Sabah Al-Ahmad"],
    "SABAH AL-AHMAD CITY 3": ["Sabah Al-Ahmad"],
    "SABAH AL-AHMAD CITY 4": ["Sabah Al-Ahmad"],
    "SABAH AL-AHMAD CITY 5": ["Sabah Al-Ahmad"],
    "AL-KHAIRAN RESIDENTIAL": ["Khiran City"],
    "AL-JAHRA INDUSTRIAL 1": ["Jahra Industrial Crafts"],
    "AL-SULAIBIYA RESIDENTIAL": ["Sulaibiya Residential"],
    "AL-SULAIBIYA AGRICULTURAL": ["Sulaibiya Agricultural"],
    "AL-JAHRA GOVERNORATE DESERT": JAHRA_DESERT,
    "SOUTH AL-MITLAE": JAHRA_DESERT,
    "AL-NAAYIM INDUSTRIAL": ["Naayem"],
    "SOUTH AMGHARA": ["South Amgara"],
    # geoBoundaries' 2011 Ardhiya_Hkomeya (government) is 2.35 of 2.4 km2 in Ardhiya 4, and
    # Ardhiya_Makhzen (stores) 1.46 of 2.7 in Ardhiya 6
    "AL-ARDIYA GOVERNORATE USE": ["Ardhiya 4"],
    "AL-ARDIYA STORES": ["Ardhiya 6"],
    "THE AIRPORT": ["International Airport"],
    "ABDULLAH AL-MUBARAK": ["Abdullah Mubarak Al-Sabah"],
    "WEST ABDULLAH AL-MUBARAK": ["West of Abdullah Al Mubarak Al Sabah"],
    # no OSM area; geoBoundaries' 2011 Messila is 1.86 of 4.0 km2 in Al Masayel
    "AL-MISILA": ["Al Masayel"],
    "WEST ABU FUTAIRA": ["Abu Fatira"],
}
# polygons that BY_HAND names more than once by design (several OSM features of one name)
POLY_COUNT = {"Sulaibiya Agricultural": 2}


def fetch():
    os.makedirs(RAW, exist_ok=True)

    def post(q, dst):
        data = urllib.parse.urlencode({"data": q}).encode()
        req = urllib.request.Request(OVERPASS, data=data, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
        if not body.lstrip().startswith(b"{") or b'"elements"' not in body:
            raise SystemExit(f"Overpass did not return JSON: {body[:300]!r}")
        with open(dst + ".part", "wb") as fh:
            fh.write(body)
        os.replace(dst + ".part", dst)
        print(f"  {dst} ({len(body):,} bytes)")

    post(Q_TAGS, OSM_TAGS)
    with open(OSM_TAGS, encoding="utf-8") as fh:
        ids = [str(e["id"]) for e in json.load(fh)["elements"]
               if e["tags"].get("admin_level") in ("4", "6")]
    post(Q_GEOM.format(ids=",".join(ids)), OSM_GEOM)


def fold(s):
    s = re.sub("[ً-ْـ]", "", str(s))
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه")
          .replace("ى", "ي"))
    return re.sub(r"\s+", "", s)


def relation_polygon(rel):
    """Outer and inner rings polygonized separately, the inners subtracted (sources/md_geo.py)."""
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    outer, inner = [], []
    for m in rel.get("members", []):
        if m.get("type") != "way" or "geometry" not in m:
            continue
        pts = [(p["lon"], p["lat"]) for p in m["geometry"]]
        if len(pts) >= 2:
            (inner if m.get("role") == "inner" else outer).append(LineString(pts))
    faces = list(polygonize(unary_union(outer))) if outer else []
    if not faces:
        return None
    g = unary_union(faces)
    if inner:
        holes = list(polygonize(unary_union(inner)))
        if holes:
            g = g.difference(unary_union(holes))
    return g if g.is_valid else g.buffer(0)


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv or not os.path.exists(OSM_GEOM):
        fetch()
    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM}; run sources/kw.py first")

    with open(OSM_GEOM, encoding="utf-8") as fh:
        els = json.load(fh)["elements"]
    rows = []
    for e in els:
        t = e["tags"]
        rows.append(dict(osm_id=e["id"], level=t.get("admin_level"), en=t.get("name:en"),
                         ar=t.get("name:ar") or t.get("name"), osm_pop=t.get("population"),
                         osm_pop_date=t.get("population:date"), geometry=relation_polygon(e)))
    g = gpd.GeoDataFrame(rows, crs=4326)
    if g.geometry.isna().any():
        raise SystemExit(f"relations with no polygon: {g.loc[g.geometry.isna(), 'osm_id'].tolist()}")
    govs, areas = g[g["level"] == "4"], g[g["level"] == "6"].copy()
    if len(govs) != EXPECTED_GOVS or len(areas) != EXPECTED_AREAS:
        raise SystemExit(f"{len(govs)} governorates and {len(areas)} areas in OSM, expected "
                         f"{EXPECTED_GOVS} and {EXPECTED_AREAS}")
    pts = areas.copy()
    pts["geometry"] = areas.representative_point()
    j = gpd.sjoin(pts, govs[["en", "geometry"]].rename(columns={"en": "osm_gov"}), how="left",
                  predicate="within")
    areas["osm_gov"] = j.loc[~j.index.duplicated(), "osm_gov"]
    areas["km2"] = areas.to_crs(METRIC).area / 1e6
    print(f"OSM: {len(areas)} areas, {len(govs)} governorates")

    # ---- the census side ----
    kw = pd.read_csv(NORM)
    pop = kw.groupby("geo_id")["count"].sum()
    gov_of = kw.drop_duplicates("geo_id").set_index("geo_id")["governorate"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("kw_src", os.path.join(HERE, "kw.py"))
    src = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(src)
    t52 = src.xl("t52")
    ar_of = {}
    for _i, r in t52.iterrows():
        if pd.notna(r[10]) and str(r[10]).strip() in pop.index:
            ar_of[str(r[10]).strip()] = str(r[0]).strip()
    census = list(pop.index)
    if len(census) != 157 or set(ar_of) != set(census):
        raise SystemExit(f"{len(census)} areas in kw.csv, {len(ar_of)} Arabic names")

    # ---- the join ----
    by_fold = {}
    for i, r in areas.iterrows():
        by_fold.setdefault(fold(r["ar"]), []).append(i)
    by_en = {}
    for i, r in areas.iterrows():
        by_en.setdefault(r["en"], []).append(i)
    pairs, how = [], {}
    for a in census:
        if a in BY_HAND:
            for en in BY_HAND[a]:
                hits = by_en.get(en, [])
                if len(hits) != POLY_COUNT.get(en, 1):
                    raise SystemExit(f"BY_HAND {a}: OSM has {len(hits)} areas named {en!r}")
                pairs += [(a, i) for i in hits]
            how[a] = "by hand"
            continue
        hits = by_fold.get(fold(ar_of[a]), [])
        if len(hits) != 1:
            raise SystemExit(f"{a} ({ar_of[a]}): {len(hits)} OSM areas by Arabic name and not in BY_HAND")
        i = hits[0]
        if areas.at[i, "osm_gov"] != OSM_GOV[gov_of[a]] and a not in GOV_PINNED:
            raise SystemExit(f"{a}: census governorate {gov_of[a]}, OSM {areas.at[i, 'osm_gov']}")
        pairs.append((a, i))
        how[a] = "Arabic name"
    n_ar = sum(1 for v in how.values() if v == "Arabic name")
    print(f"  join: {n_ar} areas on the Arabic name, {len(how) - n_ar} by hand; all {len(census)} joined")

    # ---- units: connected groups of census areas and polygons ----
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, i in pairs:
        parent[find(("a", a))] = find(("p", i))
    groups = {}
    for a, i in pairs:
        groups.setdefault(find(("a", a)), (set(), set()))
        groups[find(("a", a))][0].add(a)
        groups[find(("a", a))][1].add(i)
    units, lut = [], []
    for _root, (anames, polys) in groups.items():
        name = max(anames, key=lambda x: (pop[x], x))
        geom = areas.loc[sorted(polys)].geometry.union_all()
        units.append(dict(unit=name, governorate=gov_of[name], pop=int(pop[list(anames)].sum()),
                          n_areas=len(anames), osm=", ".join(sorted(str(areas.at[i, "en"]) for i in polys)),
                          geometry=geom))
        lut += [dict(geo_id=a, unit=name) for a in sorted(anames)]
    u = gpd.GeoDataFrame(units, crs=4326)
    # OSM nests some areas inside others (Sabah Al-Ahmad inside "South of Sabah Al-Ahmad City",
    # Al Mitla inside "Al Metlaa - Jal Al Atraf", Wafra Residential inside Wafra Farms, Sabhan
    # Industrial inside Wista). Smaller units are cut out of larger ones, smallest first.
    um = u.to_crs(METRIC)
    order = um.area.sort_values().index
    done = None
    for i in order:
        g0 = um.at[i, "geometry"]
        g1 = g0 if done is None else g0.difference(done)
        if g1.is_empty or g1.area < 0.5 * g0.area and u.at[i, "pop"] > 0 and g0.area < 1e8:
            raise SystemExit(f"{u.at[i, 'unit']}: cutting nested units leaves "
                             f"{g1.area / 1e6:.2f} of {g0.area / 1e6:.2f} km2")
        if g1.area < g0.area - 1e4:
            print(f"      {u.at[i, 'unit']:<28} {g0.area / 1e6:9.1f} km2, {g1.area / 1e6:9.1f} once the "
                  f"units inside it are cut out")
        um.at[i, "geometry"] = g1
        done = g0 if done is None else done.union(g0)
    u = um.to_crs(4326)
    if u["pop"].sum() != pop.sum():
        raise SystemExit("units do not hold every person in kw.csv")
    merged = u[u["n_areas"] > 1].sort_values("pop", ascending=False)
    print(f"  {len(u)} drawn units from {len(census)} census areas; merged:")
    for _i, r in merged.iterrows():
        print(f"      {r['unit']:<28} {r['pop']:>8,}  {r['n_areas']} areas  [{r['osm'][:70]}]")

    # ---- witness: OSM's own population tags, most of them PACI's, against the census ----
    rows_w = []
    for _root, (anames, polys) in groups.items():
        if len(polys) == 1 and len(anames) == 1:
            i = next(iter(polys))
            v = areas.at[i, "osm_pop"]
            if v is not None and str(v).isdigit():
                a = next(iter(anames))
                rows_w.append((a, int(pop[a]), int(v), areas.at[i, "osm_pop_date"]))
    print(f"  witness: {len(rows_w)} OSM areas carry a population tag (PACI, 2020-2024, as tagged):")
    bad = []
    for a, c, v, d in sorted(rows_w, key=lambda x: -x[1]):
        print(f"      {a:<24} census 2021 {c:>8,}   OSM {v:>8,} ({d})   {v / c:5.2f}")
        if not TAG_BAND[0] <= v / c <= TAG_BAND[1]:
            bad.append(a)
    if len(rows_w) < TAG_MIN or bad:
        raise SystemExit(f"OSM population tags: {len(rows_w)} found, outside {TAG_BAND}: {bad}")

    um = u.to_crs(METRIC)
    ov = gpd.overlay(um[["unit", "geometry"]], um[["unit", "geometry"]].rename(columns={"unit": "u2"}),
                     how="intersection", keep_geom_type=True)
    ov = ov[ov["unit"] < ov["u2"]]
    ov["km2"] = ov.area / 1e6
    big = ov[ov["km2"] > MAX_OVERLAP_KM2]
    print(f"  overlap between units: {ov['km2'].sum():.3f} km2 in total")
    if len(big):
        raise SystemExit(f"units overlap: {big[['unit', 'u2', 'km2']].to_string()}")

    used = {i for _a, i in pairs}
    spare = areas.loc[[i for i in areas.index if i not in used]].sort_values("km2", ascending=False)
    print(f"  OSM areas no census area claims ({len(spare)}, {spare['km2'].sum():,.0f} km2):")
    for _i, r in spare.iterrows():
        print(f"      {str(r['en']):<40} {str(r['osm_gov']):<30} {r['km2']:8.1f} km2")

    os.makedirs(OUT_DIR, exist_ok=True)
    u[["unit", "governorate", "pop", "n_areas", "osm", "geometry"]].to_file(
        OUT_UNITS, layer="units", driver="GPKG")
    pd.DataFrame(lut).to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_UNITS} ({len(u)} units) and {LOOKUP} ({len(lut)} areas)")


if __name__ == "__main__":
    main()
