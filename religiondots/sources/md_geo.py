"""Moldova — boundaries for the 901 UATs the 2024 census publishes religion for.

Writes data/geo/md/md_uat.gpkg and data/geo/md/md_uat_lookup.csv.

**THE POLYGONS COME FROM THE OFFICE THAT PUBLISHED THE COUNTS, KEYED BY THE SAME CODE, SO
THERE IS NO NAME JOIN.** That is worth saying because it was nearly not the case. Three of
the usual boundary sources stop at the raion: geoBoundaries has MDA at ADM0 and ADM1 only,
the HDX COD-AB `mda_admin_boundaries` has `admin0` and `admin1` only, and Kontur's MD
extract has 287 units at its finest level against the 901 wanted. A name join to
OpenStreetMap was built and then thrown away when BNS's own GIS server turned up.

**THE LAYER IS `comune_p_distrib_2024_view` ON `gis.statistica.md`**, 897 polygons carrying
`code_com` — the CUATM *cod statistic* with the trailing `00` dropped, so `14220` is
Drepcăuți's `1422000`. Its code set is the census table's own: 896 of the 897 are exactly
the 896 non-sector UATs of `data/normalized/md.csv`, and the 897th is `01010`, oraşul
Chişinău, which the census splits into five sectors instead.

**AND THE LAYER CARRIES ITS OWN POPULATION, WHICH IS THE CHECK.** `p_distrib` is BNS's 2024
census population for the polygon. It equals the census total this project computed from
table 5.31, to the person, for **all 896** units, and `01010`'s 567,038 equals the sum of
the five sector rows. A code join can still be a join to the wrong vintage of a boundary;
agreement on a published population as well is what rules that out.

BNS's parallel ArcGIS Online organisation publishes the same tier as `lau2_2024`, 982
polygons, which is this layer plus the 85 units of Transnistria and Bender and the four
named right-bank places the census does not cover. It is not used: `comune_p_distrib` is
already the census's universe, so taking it means the "who is missing" question is answered
by BNS rather than by this file.

**THE FIVE CHIŞINĂU SECTORS ARE THE ONE THING BNS DOES NOT DRAW.** Botanica, Buiucani,
Centru, Ciocana and Rîşcani are 567,038 people, 23.5% of the country, and leaving them as
one polygon would make Chişinău by far the worst capital case on this map. The census does
publish them, so the geometry is taken from OpenStreetMap's five `admin_level=7` relations
and **clipped to `01010`**. The clip is not cosmetic: OSM's sectors are the administrative
sectors of the *municipality*, which reach out over the suburban towns and communes that
the census counts separately, so unclipped they would overlap eighteen other UATs. After
clipping, the five pieces are asserted to cover `01010` and not to overlap each other.

Usage:
    python sources/md_geo.py --fetch    ~7.5 MB, two servers
    python sources/md_geo.py            rebuild from data/raw/md/
"""

import json
import os
import sys
import unicodedata
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "md")
OUT_DIR = os.path.join(ROOT, "data", "geo", "md")
OUT = os.path.join(OUT_DIR, "md_uat.gpkg")
LOOKUP = os.path.join(OUT_DIR, "md_uat_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "md.csv")

COMUNE_JSON = os.path.join(RAW, "comune_p_distrib_2024.geojson")
COMUNE_URL = (
    "https://gis.statistica.md/server/rest/services/Hosted/"
    "comune_p_distrib_2024_view/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326"
    "&resultRecordCount=2000&f=geojson")
COMUNE_MIN = 6_000_000

SECTORS_JSON = os.path.join(RAW, "md_osm_sectors.json")
OVERPASS = "https://overpass-api.de/api/interpreter"
# [[reference_overpass_user_agent]]: a bare UA gets a 406 that reads as a rate limit.
UA = "religiondots/1.0 (research map; contact anitaxinchen@gmail.com)"
Q_SECTORS = """[out:json][timeout:300];
area["ISO3166-1"="MD"][admin_level=2]->.md;
(relation(area.md)["boundary"="administrative"]["admin_level"="7"];);
out geom;"""
SECTORS_MIN = 200_000

EXPECTED_UATS = 901
EXPECTED_POLYGONS = 897           # the 896 non-sector UATs plus oraşul Chişinău
CHISINAU_CITY = "01010"           # code_com; the census splits it into the five sectors
SECTOR_CODES = {                  # census geo_id -> the OSM relation's folded name
    "0110000": "botanica",
    "0120000": "buiucani",
    "0130000": "centru",
    "0140000": "ciocana",
    "0150000": "riscani",
}
SECTOR_TOTAL = 567_038

# Moldova's whole territory, 33,846 km², which is what the 982-polygon `lau2_2024` layer
# covers.  These 897 are that minus the left-bank units and Bender, so they must come in
# WELL under it and not near it.  The band is measured here rather than copied (§9u's
# rule): the drawn polygons are 30,323 km², 89.6% of the country, which is consistent with
# the ~3,570 km² usually given for the administrative units on the left bank, and NOT with
# Transnistria's larger claimed territory of 4,163 km² — some left-bank villages of raionul
# Dubăsari are Moldova-administered and were enumerated.
MOLDOVA_KM2 = 33_846
MIN_COVERAGE, MAX_COVERAGE = 0.87, 0.92
# The clipped sectors must actually fill oraşul Chişinău, and must not stack.
SECTOR_COVER_MIN = 0.995
MAX_SECTOR_OVERLAP = 0.005


def fold(s):
    """Fold a Moldovan name.  ș/ş and ț/ţ are cedilla-vs-comma pairs at different
    codepoints and both are in use; â and î are the same sound and the same place, so
    `Rîşcani` and `Râșcani` must fold together, which NFKD alone would not do."""
    s = " ".join(str(s).split()).split(" / ")[0]
    s = (s.replace("ş", "s").replace("ș", "s").replace("Ş", "S").replace("Ș", "S")
          .replace("ţ", "t").replace("ț", "t").replace("Ţ", "T").replace("Ț", "T")
          .replace("â", "i").replace("î", "i").replace("Â", "I").replace("Î", "I"))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    if s.startswith("sectorul "):
        s = s[len("sectorul "):]
    return " ".join(s.split())


def _save(dest, body, min_bytes, what):
    # §5a: a 200 is not a download.  An ArcGIS error is a 200 with {"error": ...} in it.
    if len(body) < min_bytes:
        raise SystemExit(f"{what} returned {len(body):,} bytes, expected at least "
                         f"{min_bytes:,}: {body[:300]!r}")
    with open(dest, "wb") as fh:
        fh.write(body)
    print(f"  {dest} ({len(body):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(COMUNE_JSON) and os.path.getsize(COMUNE_JSON) >= COMUNE_MIN:
        print("already have", COMUNE_JSON)
    else:
        print("downloading", COMUNE_URL.split("?")[0])
        req = urllib.request.Request(COMUNE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            _save(COMUNE_JSON, r.read(), COMUNE_MIN, "gis.statistica.md")

    if os.path.exists(SECTORS_JSON) and os.path.getsize(SECTORS_JSON) >= SECTORS_MIN:
        print("already have", SECTORS_JSON)
        return
    print("querying Overpass for the Chişinău sectors")
    data = urllib.parse.urlencode({"data": Q_SECTORS}).encode()
    req = urllib.request.Request(OVERPASS, data=data, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        _save(SECTORS_JSON, r.read(), SECTORS_MIN, "Overpass")


def _relation_polygon(rel):
    """Assemble one OSM boundary relation into a polygon.

    Outer and inner rings are polygonized separately and the inners subtracted; doing it in
    one pass would turn an enclave into a second outer face and silently double the area.
    """
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
    if g.is_empty:
        return None
    return g if g.is_valid else g.buffer(0)


def main():
    import geopandas as gpd
    import pandas as pd
    from shapely.ops import unary_union

    if "--fetch" in sys.argv:
        fetch()
    for p in (COMUNE_JSON, SECTORS_JSON, NORM):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run sources/md.py, then "
                             "sources/md_geo.py --fetch")

    # ---- the census side ----
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    uat = df[df["geo_level"] == "uat"]
    leaves = dict(zip(uat["geo_id"], uat["geo_name"]))
    raion_of = dict(zip(uat["geo_id"],
                        uat["note"].str.extract(r"raion=([^;]+)")[0].str.strip()))
    census_pop = uat.groupby("geo_id")["count"].sum().to_dict()
    if len(leaves) != EXPECTED_UATS:
        raise SystemExit(f"{len(leaves)} UATs in md.csv, expected {EXPECTED_UATS}")

    # ---- BNS's commune layer ----
    g = gpd.read_file(COMUNE_JSON)
    if len(g) != EXPECTED_POLYGONS:
        raise SystemExit(f"{len(g)} polygons in the BNS layer, expected "
                         f"{EXPECTED_POLYGONS} -- the layer was republished")
    if g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"the BNS layer came back in {g.crs}, expected EPSG:4326")
    g["code"] = g["code_com"].astype(str).str.zfill(5)
    if g["code"].duplicated().any():
        raise SystemExit("code_com is not unique in the BNS layer")
    if g.geometry.isna().any() or g.geometry.is_empty.any():
        raise SystemExit("the BNS layer has a feature with no geometry")
    print(f"BNS comune_p_distrib_2024_view: {len(g):,} polygons, EPSG:4326")

    # ---- 1. the code join, both ways ----
    want = {c[:5] for c in leaves if c not in SECTOR_CODES}
    have = set(g["code"])
    missing, spare = sorted(want - have), sorted(have - want)
    print("\n  the join, both ways (§12):")
    print(f"    census UATs excluding the 5 Chişinău sectors  {len(want):>5,}")
    print(f"    polygons in the BNS layer                     {len(have):>5,}")
    print(f"    census rows with no polygon                   {len(missing):>5} {missing}")
    print(f"    polygons with no census row                   {len(spare):>5} {spare}")
    if missing or spare != [CHISINAU_CITY]:
        raise SystemExit("join FAILED -- the only polygon without a census row should be "
                         f"{CHISINAU_CITY}, oraşul Chişinău")

    # ---- 2. the layer's own population against the counts read from table 5.31 ----
    g["p_distrib"] = g["p_distrib"].astype(int)
    pop = dict(zip(g["code"], g["p_distrib"]))
    off = [(c, leaves[c], census_pop[c], pop[c[:5]])
           for c in sorted(leaves) if c not in SECTOR_CODES
           and census_pop[c] != pop[c[:5]]]
    print(f"\n  BNS's own p_distrib against the totals read from table 5.31:")
    print(f"    units compared {len(want):,}; units differing {len(off)}")
    for row in off[:6]:
        print(f"      {row[0]} {row[1]}: table {row[2]:,} vs layer {row[3]:,}")
    if off:
        raise SystemExit(f"{len(off)} polygons carry a different population from the "
                         "census table -- the layer is a different vintage of the units")
    if pop[CHISINAU_CITY] != SECTOR_TOTAL:
        raise SystemExit(f"oraşul Chişinău's polygon says {pop[CHISINAU_CITY]:,}, the five "
                         f"census sectors say {SECTOR_TOTAL:,}")
    print("    Exact, to the person, on every unit. The polygons and the counts are the "
          "same\n    units of the same census (§12).")

    # ---- 3. the five Chişinău sectors, clipped to oraşul Chişinău ----
    with open(SECTORS_JSON, encoding="utf-8") as fh:
        rels = json.load(fh)["elements"]
    by_name = {}
    for e in rels:
        poly = _relation_polygon(e)
        if poly is not None:
            by_name[fold(e["tags"].get("name", ""))] = poly
    city = g.loc[g["code"] == CHISINAU_CITY, "geometry"].iloc[0]
    sectors = {}
    for code, nm in SECTOR_CODES.items():
        if nm not in by_name:
            raise SystemExit(f"OpenStreetMap has no admin_level=7 relation folding to "
                             f"'{nm}' -- the sectors were renamed")
        clipped = by_name[nm].intersection(city)
        if clipped.is_empty:
            raise SystemExit(f"sector {nm} does not intersect oraşul Chişinău")
        sectors[code] = clipped if clipped.is_valid else clipped.buffer(0)

    cover = unary_union(list(sectors.values())).area / city.area
    stacked = 1 - cover * city.area / sum(s.area for s in sectors.values())
    print(f"\n  the five sectors clipped to oraşul Chişinău: they cover {cover:.3%} of it, "
          f"and {stacked:.3%} of their summed area is double-covered")
    if cover < SECTOR_COVER_MIN:
        raise SystemExit("the clipped sectors do not fill oraşul Chişinău -- part of the "
                         "city would have no polygon and its people would not be drawn")
    if stacked > MAX_SECTOR_OVERLAP:
        raise SystemExit("the clipped sectors overlap each other")

    # ---- 4. build the layer ----
    keep = g[g["code"] != CHISINAU_CITY].copy()
    code_to_uat = {c[:5]: c for c in leaves if c not in SECTOR_CODES}
    keep["unit"] = keep["code"].map(code_to_uat)
    recs = gpd.GeoDataFrame(
        {"unit": list(keep["unit"]) + list(sectors),
         "name": [leaves[u] for u in keep["unit"]] + [leaves[c] for c in sectors],
         "raion": [raion_of[u] for u in keep["unit"]] + [raion_of[c] for c in sectors]},
        geometry=list(keep.geometry) + list(sectors.values()), crs=4326)
    if len(recs) != EXPECTED_UATS or recs["unit"].duplicated().any():
        raise SystemExit(f"{len(recs)} output polygons, expected {EXPECTED_UATS} distinct")
    invalid = ~recs.geometry.is_valid
    if invalid.any():
        print(f"  repairing {int(invalid.sum())} invalid polygons with buffer(0)")
        recs.loc[invalid, "geometry"] = recs.loc[invalid, "geometry"].buffer(0)

    # ---- 5. they tile the enumerated country, and they do not stack ----
    m = recs.to_crs(6933)
    km2 = m.area.sum() / 1e6
    cov = km2 / MOLDOVA_KM2
    print(f"\n  total area {km2:,.0f} km² against all of Moldova's {MOLDOVA_KM2:,}"
          f" — {cov:.1%}, the rest being the left bank and Bender")
    if not (MIN_COVERAGE <= cov <= MAX_COVERAGE):
        raise SystemExit(f"the UAT polygons cover {cov:.1%} of Moldova, outside the "
                         f"{MIN_COVERAGE:.0%}-{MAX_COVERAGE:.0%} band the enumerated area "
                         "should fall in")
    overlap = 1 - unary_union(list(m.geometry)).area / 1e6 / km2
    print(f"  summed area minus dissolved area: {overlap:.3%}")
    if overlap > 0.01:
        raise SystemExit(f"{overlap:.1%} of the summed area is double-covered")
    qs = (m.area / 1e6).quantile([0.05, 0.5, 0.95, 1.0])
    print("  area per UAT, km²: "
          + "  ".join(f"p{int(q * 100)}={v:,.1f}" for q, v in qs.items()))
    ppl = pd.Series([census_pop[u] for u in recs["unit"]])
    print(f"  people per UAT: median {ppl.median():,.0f}, "
          f"p95 {ppl.quantile(0.95):,.0f}, max {ppl.max():,.0f} "
          f"({recs['name'].iloc[int(ppl.idxmax())]})")

    os.makedirs(OUT_DIR, exist_ok=True)
    recs.to_file(OUT, layer="uat", driver="GPKG")
    print(f"\nwrote {OUT} ({len(recs):,} polygons, EPSG:4326)")

    lut = pd.DataFrame({"geo_id": sorted(leaves)})
    lut["kod"] = lut["geo_id"]
    lut.to_csv(LOOKUP, index=False)
    print(f"wrote {LOOKUP} ({len(lut):,} rows)")


if __name__ == "__main__":
    main()
