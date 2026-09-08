"""Mauritius — boundaries for the 182 drawn units (Village Council Areas and Municipal
Council Wards), from OpenStreetMap.

Writes data/geo/mu/mu_units.gpkg and data/geo/mu/mu_lookup.csv.

**NO HUMANITARIAN SOURCE HAS THIS TIER AND OSM DOES.** COD-AB Mauritius stops at ADM1 — 12
districts, outer islands included — and geoBoundaries has only ADM0 and ADM1. Statistics
Mauritius publishes religion at ward and VCA and ships no geography at all. So the only
boundaries that exist for the drawn tier are OSM's, and they are unusually complete for a
small island state: **164 relations at `admin_level=8`** (the VCAs, plus the five towns as
whole units) and **35 at `admin_level=9`** (the municipal wards). All 199 assemble into
valid polygons with no failures, and the level-8 set covers 1,942 km² against Mauritius's
2,040 km² — the shortfall is coastal generalisation.

**THE FIVE TOWN RELATIONS ARE PARENTS AND ARE DROPPED**, exactly as `sources/mu.py` drops
their equivalents in the table: `Vacoas-Phoenix` at level 8 contains `Town of
Vacoas-Phoenix, Ward 1`…`Ward 6` at level 9. Keeping both would double 40% of the country.

THE JOIN IS BY NAME IN THREE STAGES, each narrower and each reported:

  1. **folded exact** — accents, case, punctuation, the `VCA`/`Town of` noise words, the
     census's cross-district parenthetical, and its abbreviations (`Riv.`→`Rivière`,
     `Vac `→`Vacoas`, `B-Bassin/R-Hill`→`Beau Bassin/Rose Hill`, `St`→`Saint`). 177 units.
  2. **audited fuzzy at 0.86** — three units, every pairing printed with its ratio so it can
     be read rather than trusted. They are `Bois Chéri`/`Bois Chérie` (the census and OSM
     spell it differently), and two ward rows where the census records a cross-district
     split OSM does not.
  3. **two structural repairs**, named below, because they are facts about the data rather
     than about spelling.

**AND THEN EVERY PAIRING IS CHECKED SPATIALLY, WHICH IS THE PART THAT MATTERS.** D6 has no
code column of any kind — a unit's district is its POSITION in the printed table and nothing
else — so `sources/mu.py` records that position and this module asserts that each matched
polygon actually lies in the district the census put it in, against COD-AB's ADM1. A name
join on 183 French place names on a small island is exactly where §12's shape 2 lives (a key
matches almost everything and pairs some units with the wrong polygon, leaving every total
intact), and nothing else here would catch it.

Usage:
    python sources/mu_geo.py --fetch    one Overpass query, ~5.7 MB, and COD-AB from HDX
    python sources/mu_geo.py            rebuild from data/raw/mu/
"""

import difflib
import json
import os
import re
import sys
import unicodedata
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mu")
OUT_DIR = os.path.join(ROOT, "data", "geo", "mu")
OUT = os.path.join(OUT_DIR, "mu_units.gpkg")
LOOKUP = os.path.join(OUT_DIR, "mu_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "mu.csv")

OSM_NAME = "mu_osm_adm89.json"
COD_ZIP = "mus_adm_2020_v2_shp.zip"
COD_URL = ("https://data.humdata.org/dataset/30c13830-6f14-46db-acd9-29eae51c540a/"
           "resource/2832fe0a-e89e-44df-84e1-a2c12117cc60/download/"
           "mus_adm_2020_v2_shp.zip")

OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# Two bboxes: the main island, and Rodrigues 600 km east. An ISO3166-1 area lookup 504s.
QUERY = """
[out:json][timeout:600];
(
  relation["boundary"="administrative"]["admin_level"~"^(8|9)$"](-20.60,57.25,-19.95,57.90);
  relation["boundary"="administrative"]["admin_level"~"^(8|9)$"](-19.80,63.30,-19.60,63.55);
);
out geom;
"""

# The level-8 relations that are PARENTS of level-9 wards and must be dropped, or 40% of
# the country is counted twice.
#
# **THERE ARE FOUR, NOT FIVE, AND PORT LOUIS IS THE ONE MISSING** — which is not an OSM gap
# but the same fact `sources/mu.py` records: Port Louis is a district *and* a town, so its
# wards hang off `PORT LOUIS DISTRICT-Wholly Urban` one level up and there is no town row
# beside them in either the table or OSM. The other four towns sit inside Plaines Wilhems.
TOWN_PARENTS = {"Beau Bassin / Rose Hill", "Curepipe", "Quatre Bornes", "Vacoas-Phoenix"}

# ---- the two structural repairs, and why each is a fact about the data ----
#
# 1. Dubreuil. The census prints ONE row, `Dubreuil VCA (East in Flacq & West in P/W)`,
#    under Moka; OSM splits the same VCA into three relations, `East`, `West` and `Part`.
#    The census row is the whole VCA (its name carries no part-word, unlike every other
#    split unit, which is named `-East` or `-West`), so the three are unioned.
DUBREUIL = ["Dubreuil VCA, East", "Dubreuil VCA, West", "Dubreuil VCA, Part"]
DUBREUIL_CENSUS = "Dubreuil VCA (East in Flacq & West in P/W)"
#
# 2. Vacoas-Phoenix Wards 5 and 6-West. **OSM has no polygon for either** — it carries
#    Vacoas wards 1, 2, 3, 4 and `Ward 6, East`, and stops. That is a gap in OSM, not a
#    naming difference. The two are drawn TOGETHER on the remainder of the town polygon
#    after the wards that do exist are removed, and they are therefore ONE drawn unit
#    rather than two. 35,664 people, 2.89% of the country, and the cost is one internal
#    boundary inside one town.
VACOAS_MISSING = ["Town of Vacoas/Phoenix-Ward 5",
                  "Town of Vacoas/Phoenix-Ward 6-West (East in Moka)"]
VACOAS_TOWN = "Vacoas-Phoenix"
#
# 3. Rivière du Poste. **THE SPATIAL CHECK FOUND THIS ONE AND NOTHING ELSE WOULD HAVE.**
#    Both sides split the VCA in two and call the pieces East and West, the names matched
#    cleanly at stage 1, and the totals reconcile either way — but they are not the same
#    split. The census cuts it on the Grand Port/Savanne district line; OSM cuts it
#    somewhere else, and *both* OSM pieces sit mostly in Grand Port (`West` is 71.4% Grand
#    Port / 28.6% Savanne, `East` is 99.8% Grand Port). So OSM's `West` is not the census's
#    `-West`, and pairing them by name puts every Savanne dot in Grand Port.
#    The repair is to rebuild the census's own split: union the two OSM pieces back into the
#    whole VCA and intersect with each row's census district.
POSTE_OSM = ["Rivière du Poste VCA, East", "Rivière du Poste VCA, West"]
POSTE_CENSUS = {"Riv. du Poste VCA-East (West in Savanne)": "GRAND PORT",
                "Riv. du Poste VCA-West (East in G/P)": "SAVANNE"}

FUZZY_CUTOFF = 0.86

# Statistics Mauritius's district names against COD-AB's ADM1_EN. Three differ, all of them
# abbreviation or transliteration rather than a different place.
DISTRICT_ALIAS = {
    "R. DU REMPART": "Riviere du Rempart",
    "PLAINES WILHEMS": "Plaine Wilhems",
    "RODRIGUES": "Rodriguez Island",
}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)

    dest = os.path.join(RAW, OSM_NAME)
    if not (os.path.exists(dest) and os.path.getsize(dest) > 1_000_000):
        last = None
        for ep in OVERPASS:
            print("POST", ep)
            try:
                r = requests.post(ep, data={"data": QUERY}, timeout=900,
                                  headers={"User-Agent": "religiondots"})
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                print("   ", last)
                continue
            # §5a: Overpass answers a timeout with HTTP 200 and an HTML error page.
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                open(dest, "w", encoding="utf-8").write(r.text)
                print(f"  {os.path.getsize(dest):,} bytes")
                break
            last = f"{r.status_code}, body starts {r.text[:80]!r}"
            print("   ", last)
        else:
            raise SystemExit(f"no Overpass mirror answered with JSON -- last: {last}")
    else:
        print("already have", dest)

    dest = os.path.join(RAW, COD_ZIP)
    if os.path.exists(dest) and zipfile.is_zipfile(dest):
        print("already have", dest)
        return
    print("GET", COD_URL)
    r = requests.get(COD_URL, timeout=900, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a zip -- HDX 302s and the redirect must be "
                         "followed; got {os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def fold(s):
    """Census name -> OSM name, as far as spelling goes. See the module docstring."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'").replace("‘", "'").replace("–", "-")
    s = re.sub(r"\(.*?\)", " ", s)                      # the cross-district note
    s = re.sub(r"^\s*region\s*\d+\s*-\s*", " ", s)      # Rodrigues `Region 3 - St. Gabriel`
    s = s.replace("town of", " ")
    s = re.sub(r"\bvca\b|\bvillage council area\b", " ", s)
    s = re.sub(r"\briv\.?\b", "riviere", s)
    s = re.sub(r"\bvac\b", "vacoas", s)
    s = re.sub(r"\bb-bassin\b", "beaubassin", s)
    s = re.sub(r"\br-hill\b", "rosehill", s)
    s = re.sub(r"\bst\.?\b", "saint", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _polygon(el):
    """One OSM relation -> a shapely polygon, outer rings minus inner."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge, polygonize, unary_union

    outer, inner = [], []
    for m in el.get("members", []):
        if m["type"] != "way" or "geometry" not in m:
            continue
        pts = [(g["lon"], g["lat"]) for g in m["geometry"]]
        if len(pts) < 2:
            continue
        (inner if m.get("role") == "inner" else outer).append(LineString(pts))
    if not outer:
        return None
    rings = list(polygonize(linemerge(outer)))
    if not rings:
        return None
    poly = unary_union(rings)
    if inner:
        holes = list(polygonize(linemerge(inner)))
        if holes:
            poly = poly.difference(unary_union(holes))
    return poly if not poly.is_empty else None


def _read_osm():
    p = os.path.join(RAW, OSM_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    d = json.load(open(p, encoding="utf-8"))
    out = {}
    failed = []
    for el in d["elements"]:
        name = el["tags"].get("name", "")
        poly = _polygon(el)
        if poly is None:
            failed.append(name)
            continue
        out[el["id"]] = (name, el["tags"].get("admin_level"), poly)
    # §12: assert the feature count, never the absence of an exception.
    if failed:
        raise SystemExit(f"{len(failed)} OSM relations would not close into polygons: "
                         f"{failed[:5]}")
    if len(out) < 150:
        raise SystemExit(f"only {len(out)} OSM relations -- the bbox or the tagging has "
                         "changed")
    return out


def _read_districts():
    import geopandas as gpd

    src = os.path.join(RAW, COD_ZIP)
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run with --fetch first")
    g = gpd.read_file(f"zip://{src}!mus_admbnda_adm1_2020_v2.shp")
    if len(g) != 12:
        raise SystemExit(f"COD ADM1 has {len(g)} features, expected 12")
    return g


def main():
    import geopandas as gpd
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()

    osm = _read_osm()
    n8 = sum(1 for _, lv, _ in osm.values() if lv == "8")
    n9 = sum(1 for _, lv, _ in osm.values() if lv == "9")
    print(f"OSM: {len(osm)} relations ({n8} at admin_level=8, {n9} at 9), all closed")

    # Drop the five town parents (see the module docstring).
    drop = {i for i, (nm, lv, _) in osm.items() if lv == "8" and nm in TOWN_PARENTS}
    if len(drop) != len(TOWN_PARENTS):
        got = sorted(osm[i][0] for i in drop)
        raise SystemExit(f"expected {len(TOWN_PARENTS)} town parents at level 8, found "
                         f"{got} -- OSM's tagging has changed")
    towns = {osm[i][0]: osm[i][2] for i in drop}
    pool = {i: v for i, v in osm.items() if i not in drop}
    print(f"  dropped {len(drop)} town parents; {len(pool)} candidate polygons")

    if not os.path.exists(NORM):
        raise SystemExit(f"missing {NORM} -- run sources/mu.py first")
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    cen = df[(df["geo_level"] == "unit") & (df["source_category"] == "Total")]
    census = {}
    for _, r in cen.iterrows():
        m = re.search(r"district=(.+?)\s*$", str(r["note"]))
        census[r["geo_name"]] = (r["geo_id"], m.group(1) if m else None)
    print(f"census drawn units: {len(census)}")

    by_fold = {}
    for i, (nm, lv, _) in pool.items():
        by_fold.setdefault(fold(nm), []).append(i)

    pairs, todo = {}, []
    for name in census:
        k = fold(name)
        hits = by_fold.get(k, [])
        if len(hits) == 1:
            pairs[name] = [hits[0]]
        elif len(hits) > 1:
            raise SystemExit(f"census {name!r} matches {len(hits)} OSM relations by name")
        else:
            todo.append(name)
    print(f"\n  stage 1, folded exact:      {len(pairs):>4}")

    used = {i for v in pairs.values() for i in v}
    free = {}
    for i, (nm, lv, _) in pool.items():
        if i not in used:
            free.setdefault(fold(nm), i)

    fuzzy = 0
    for name in list(todo):
        if name in VACOAS_MISSING or name == DUBREUIL_CENSUS:
            continue
        hit = difflib.get_close_matches(fold(name), list(free), n=1, cutoff=FUZZY_CUTOFF)
        if not hit:
            continue
        i = free.pop(hit[0])
        ratio = difflib.SequenceMatcher(None, fold(name), hit[0]).ratio()
        print(f"      fuzzy {ratio:.3f}  {name!r}\n              -> {pool[i][0]!r}")
        pairs[name] = [i]
        todo.remove(name)
        fuzzy += 1
    print(f"  stage 2, audited fuzzy:     {fuzzy:>4}")

    # ---- stage 3: the two structural repairs ----
    ids = {nm: i for i, (nm, lv, _) in pool.items()}
    missing_ids = [n for n in DUBREUIL if n not in ids]
    if missing_ids:
        raise SystemExit(f"OSM no longer has {missing_ids} -- the Dubreuil repair is stale")
    pairs[DUBREUIL_CENSUS] = [ids[n] for n in DUBREUIL]
    if DUBREUIL_CENSUS in todo:
        todo.remove(DUBREUIL_CENSUS)

    from shapely.ops import unary_union
    if VACOAS_TOWN not in towns:
        raise SystemExit(f"OSM no longer has the {VACOAS_TOWN!r} town relation")
    have = [p for i, (nm, lv, p) in pool.items()
            if nm.startswith("Town of Vacoas-Phoenix, Ward")]
    if len(have) != 5:
        raise SystemExit(f"expected 5 Vacoas ward polygons in OSM, found {len(have)} -- "
                         "the remainder repair is stale; check whether OSM has gained the "
                         "two missing wards, in which case delete this repair")
    remainder = towns[VACOAS_TOWN].difference(unary_union(have)).buffer(0)
    if remainder.is_empty or remainder.area <= 0:
        raise SystemExit("the Vacoas remainder is empty -- OSM's town polygon no longer "
                         "exceeds its wards")
    for n in VACOAS_MISSING:
        if n in todo:
            todo.remove(n)

    # Rivière du Poste: rebuild the census's own district split. See the note above.
    missing_ids = [n for n in POSTE_OSM if n not in ids]
    if missing_ids:
        raise SystemExit(f"OSM no longer has {missing_ids} -- the Poste repair is stale")
    poste_whole = unary_union([pool[ids[n]][2] for n in POSTE_OSM])
    print(f"  stage 3, structural repairs:   3")
    print(f"      Dubreuil        = 3 OSM polygons unioned into the census's single row")
    print(f"      Vacoas 5+6-West = {remainder.area * 111.32 * 111.32 * 0.94:.1f} km2 "
          "remainder of the town, ONE drawn unit for two census rows")
    print(f"      Riv. du Poste   = 2 OSM polygons unioned, then re-split on the "
          "Grand Port/Savanne line")

    if todo:
        raise SystemExit(f"{len(todo)} census units still unmatched: {todo}")

    # ---- geometry per DRAWN unit ----
    geoms, members = {}, {}
    for name, rel_ids in pairs.items():
        polys = [pool[i][2] for i in rel_ids]
        geoms[name] = polys[0] if len(polys) == 1 else unary_union(polys)
        members[name] = [pool[i][0] for i in rel_ids]
    vac_unit = " + ".join(VACOAS_MISSING)
    geoms[vac_unit] = remainder
    members[vac_unit] = [f"{VACOAS_TOWN} minus its five mapped wards"]

    # ---- THE CHECK THAT MATTERS: does each polygon lie in the census's own district? ----
    dist = _read_districts()
    dist = dist.set_index(dist["ADM1_EN"].map(fold))

    for name, want in POSTE_CENSUS.items():
        piece = poste_whole.intersection(
            dist.loc[fold(DISTRICT_ALIAS.get(want, want)), "geometry"]).buffer(0)
        if piece.is_empty or piece.area / poste_whole.area < 0.05:
            raise SystemExit(f"the Poste repair produced nothing for {name!r} in {want} -- "
                             "the district boundary no longer crosses this VCA")
        geoms[name] = piece
        members[name] = [f"{' + '.join(POSTE_OSM)}, clipped to {want}"]
        print(f"      {name[:34]:<34} -> {100 * piece.area / poste_whole.area:4.1f}% of "
              "the VCA")
    checked = failed = 0
    for name, poly in geoms.items():
        if name == vac_unit:
            want = "PLAINES WILHEMS"
        else:
            want = census[name][1]
        key = fold(DISTRICT_ALIAS.get(want, want))
        if key not in dist.index:
            raise SystemExit(f"census district {want!r} has no COD ADM1 polygon "
                             f"(folded {key!r}; COD has {sorted(dist.index)})")
        checked += 1
        if not dist.loc[key, "geometry"].buffer(1e-4).contains(poly.representative_point()):
            failed += 1
            print(f"      OUT OF DISTRICT: {name!r} pairs with {members[name]} "
                  f"but its centre is not in {want}")
    print(f"\n  spatial check — {checked - failed}/{checked} polygons lie in the district "
          "the census puts them in")
    if failed:
        raise SystemExit(f"{failed} units pair with a polygon in the wrong district -- "
                         "the name join has made a confident wrong pairing (§12 shape 2)")

    # ---- write ----
    rows, lut = [], []
    for k, (name, poly) in enumerate(sorted(geoms.items()), start=1):
        unit = f"MU{k:03d}"
        rows.append({"unit": unit, "name": name,
                     "osm": "; ".join(members[name]), "geometry": poly})
        if name == vac_unit:
            for n in VACOAS_MISSING:
                lut.append({"geo_id": census[n][0], "unit": unit})
        else:
            lut.append({"geo_id": census[name][0], "unit": unit})

    g = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    os.makedirs(OUT_DIR, exist_ok=True)
    g.to_file(OUT, layer="units", driver="GPKG")
    print(f"\nwrote {OUT} ({len(g)} polygons for {len(lut)} census rows)")

    pd.DataFrame(lut).to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")


if __name__ == "__main__":
    main()
