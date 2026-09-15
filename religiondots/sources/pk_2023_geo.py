"""Pakistan 2023 — district boundaries on the 2023 district set, and the placement grid.

Writes:
    data/geo/pk2023/pk_districts.gpkg    the 136 census districts as drawn (`units`)
    data/geo/pk2023/pk_hexes.gpkg        Kontur H3 r8 hexes with `unit` and `pop` (`place`)
    data/geo/pk2023/pk_lookup.csv        unit -> names, what the polygon was built from,
                                         census population, Kontur population

Usage:
    python sources/pk_2023_geo.py --fetch   COD-AB zip (25 MB), one Overpass query, Kontur gz
    python sources/pk_2023_geo.py           rebuild from data/raw/pk2023/

THE 2023 DISTRICT SET IS NOT IN ANY ONE BOUNDARY FILE (spec §8.1). The census tabulates 136
districts, which is PBS's own list "as on 01-03-2023". OCHA's COD-AB v01 for Pakistan
(valid_on 2022-09-09, reviewed 2024-09-27) has 160 ADM2 units, of which 24 are Azad Kashmir
and Gilgit-Baltistan, outside the census. The other 136 are NOT the census's 136:

  * **Lehri** is a COD district and not a census one. It was folded back in, and not into one
    parent: the census prints its LEHRI sub-division under Sibi and its BHAG sub-division under
    Kachhi. COD's ADM3 layer has both tehsils under Lehri, so the polygons are reassigned
    TEHSIL BY TEHSIL and nothing is dissolved.
  * **Keamari** is a census district (notified 2020) and not a COD one, and Karachi's seven
    2023 districts do not rebuild out of COD's six at all: COD's ADM3 under Karachi is the
    2001 towns, and the 2023 Karachi West includes Manghopir (1.08m people), which COD files
    inside Gadap Town under Malir. So Karachi alone is cut from **OpenStreetMap**'s seven
    admin_level=6 district relations, which carry the 2020 layout, clipped to COD's Karachi
    footprint so the city's outer border stays COD's. Hexes in the footprint but outside every
    clipped OSM district go to the nearest one. OSM data is (c) OpenStreetMap contributors, ODbL.
  * **Surab** is COD's `Shaheed Sikandarabad` under its old name, and ten other names differ
    by spelling or word order; ALIAS lists every one.

So the join has four parts and every one is asserted: census name -> COD ADM2 within the
province, both directions, 1:1; Lehri's two tehsils found and moved; Karachi's seven census
districts -> seven OSM relations, confirmed by the OSM town names under each against the
census's own sub-division names; and a second, independent key on every COD pair, which is
the COD tehsil names against the census tehsil names printed under the same district
([[reference_name_join_wrong_neighbour]]: a name join can pair the wrong twin and every total
still adds up). Then §12's geography witness: religion shares are spatially smooth, so each
district's Hindu and Christian shares are correlated with their nearest neighbours' and
compared against shuffles of the same shares.

PLACEMENT is Kontur 2023-11-01 again, the same file the 2017 build used, joined on hex
centroids. The census and the grid are now the same year, so the national ratio should sit
near 1 and the band is tight.
"""

import csv
import difflib
import gzip
import json
import os
import re
import shutil
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "6")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pk2023")
GEO = os.path.join(ROOT, "data", "geo", "pk2023")
KONTUR = os.path.join(ROOT, "data", "geo", "kontur")
NORM = os.path.join(ROOT, "data", "normalized", "pk.csv")      # written by sources/pk_2023.py

COD_ZIP = os.path.join(RAW, "pak_admin_boundaries.shp.zip")
COD_DIR = os.path.join(RAW, "cod")
COD_URL = ("https://data.humdata.org/dataset/a64d1ff2-7158-48c7-887d-6af69ce21906/resource/"
           "d2752403-3c34-4e03-8b4e-3e55500ded10/download/pak_admin_boundaries.shp.zip")
OSM_JSON = os.path.join(RAW, "osm_karachi_districts.json")

GZ_URL = ("https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
          "kontur_population_PK_20231101.gpkg.gz")
GZ_NAME = "kontur_population_PK_20231101.gpkg.gz"
GPKG_NAME = "kontur_population_PK_20231101.gpkg"

OUT_UNITS = os.path.join(GEO, "pk_districts.gpkg")
OUT_HEXES = os.path.join(GEO, "pk_hexes.gpkg")
OUT_LOOKUP = os.path.join(GEO, "pk_lookup.csv")

DISTRICTS = 136
COD_ADM2 = 160
COD_OUTSIDE_CENSUS = {"Azad Kashmir", "Gilgit Baltistan"}

# census province (as pk.csv's note names it) -> COD adm1_name
PROVINCE = {"Khyber Pakhtunkhwa": "Khyber Pakhtunkhwa", "Punjab": "Punjab", "Sindh": "Sindh",
            "Balochistan": "Balochistan", "Islamabad": "Islamabad"}

# census district (without " DISTRICT") -> COD adm2_name, where the two do not fold together
ALIAS = {
    "DERA ISMAIL KHAN": "D. I. Khan",
    "LOWER CHITRAL": "Chitral Lower",
    "UPPER CHITRAL": "Chitral Upper",
    "LOWER KOHISTAN": "Kohistan Lower",
    "UPPER KOHISTAN": "Kohistan Upper",
    "MALAKAND PROTECTED AREA": "Malakand",
    "LAYYAH": "Leiah",
    "SURAB": "Shaheed Sikandarabad",
}

# COD (adm2_name, adm3_name) -> census district: tehsils whose district changed after COD's
# 2022 vintage. The whole of COD's Lehri district, split as the census prints it.
TEHSIL_MOVES = {
    ("Lehri", "Lehri"): "SIBI DISTRICT",
    ("Lehri", "Bhag"): "KACHHI DISTRICT",
}
COD_DISSOLVED = {"Lehri"}

# Pairs whose second key finds no tehsil name, with why that is not a wrong pairing. Every one
# must name the reason; an unexplained zero fails the join.
TEHSIL_NAMES_DIFFER = {
    "MALAKAND PROTECTED AREA":
        "COD names the two tehsils for their towns (Bat Khela, Dargai), the census for the "
        "sub-divisions those towns are the seats of (Swat Ranizai, Sam Ranizai); Malakand is "
        "also the only candidate its name has",
}

# Karachi: census district -> OSM relation id (admin_level=6). OSM's English names are the
# pre-2011 ones (West is "Orangi District", South is "Karachi District"); the town check below
# is what confirms each pairing, not these labels.
OSM_KARACHI = {
    "KARACHI WEST DISTRICT": 16347667,      # Orangi District
    "KARACHI CENTRAL DISTRICT": 16349281,   # Nazimabad District
    "KARACHI EAST DISTRICT": 16350242,      # Gulshan District
    "KORANGI DISTRICT": 16350632,
    "KARACHI SOUTH DISTRICT": 16350836,     # Karachi District
    "KEAMARI DISTRICT": 16351022,
    "MALIR DISTRICT": 16351916,
}
COD_KARACHI = {"Central Karachi", "East Karachi", "Korangi Karachi", "Malir Karachi",
               "South Karachi", "West Karachi"}
OVERPASS = ("[out:json][timeout:120];rel(id:" + ",".join(map(str, OSM_KARACHI.values()))
            + ");out geom;")

# Kontur 2023-11 against a 2023 census: the same year, so near 1. Kontur's Pakistan total
# (236.4m, of which ~5m is AJK/GB) runs a little under the census, so the band leans low.
KONTUR_RATIO_MIN = 0.85
KONTUR_RATIO_MAX = 1.10


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if not (os.path.exists(COD_ZIP) and os.path.getsize(COD_ZIP) > 20_000_000):
        r = requests.get(COD_URL, timeout=900, headers={"User-Agent": "religiondots-map-research/1.0"})
        r.raise_for_status()
        if r.content[:4] != b"PK\x03\x04":
            raise SystemExit(f"COD zip starts {r.content[:16]!r}")
        with open(COD_ZIP, "wb") as fh:
            fh.write(r.content)
        print(f"wrote {COD_ZIP} ({len(r.content):,} bytes)")
    if not os.path.exists(os.path.join(COD_DIR, "pak_admin3.shp")):
        with zipfile.ZipFile(COD_ZIP) as z:
            z.extractall(COD_DIR)
    if not (os.path.exists(OSM_JSON) and os.path.getsize(OSM_JSON) > 50_000):
        # [[reference_overpass_user_agent]]: a browser-like UA gets a bare 406 here
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": OVERPASS},
                          timeout=300, headers={"User-Agent": "religiondots-map-research/1.0"})
        r.raise_for_status()
        if not r.content.lstrip().startswith(b"{"):
            raise SystemExit(f"Overpass returned {r.content[:120]!r}")
        with open(OSM_JSON, "wb") as fh:
            fh.write(r.content)
        print(f"wrote {OSM_JSON} ({len(r.content):,} bytes)")
    towns = os.path.join(RAW, "osm_karachi_admin.json")
    if not os.path.exists(towns):
        # names of every admin relation over Karachi, so the town check can read the
        # subarea members of each district relation by name
        q = ('[out:json][timeout:60];rel[boundary=administrative][admin_level~"^(5|6|7)$"]'
             '(24.75,66.65,25.65,67.65);out tags;')
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q},
                          timeout=300, headers={"User-Agent": "religiondots-map-research/1.0"})
        r.raise_for_status()
        with open(towns, "wb") as fh:
            fh.write(r.content)
    gz = os.path.join(KONTUR, GZ_NAME)
    if not os.path.exists(gz):
        os.makedirs(KONTUR, exist_ok=True)
        r = requests.get(GZ_URL, timeout=1800, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(gz, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


def kontur_gpkg():
    """The unpacked gpkg is scratch ([[reference_religiondots_disk]]); rebuild it from the gz."""
    gpkg = os.path.join(KONTUR, GPKG_NAME)
    if os.path.exists(gpkg) and os.path.getsize(gpkg) > 50_000_000:
        return gpkg, False
    gz = os.path.join(KONTUR, GZ_NAME)
    if not os.path.exists(gz):
        raise SystemExit(f"missing {gz} -- run: python sources/pk_2023_geo.py --fetch")
    with gzip.open(gz, "rb") as src, open(gpkg, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with open(gpkg, "rb") as fh:
        if fh.read(4) != b"SQLi":
            raise SystemExit(f"{gpkg} is not a GeoPackage")
    return gpkg, True


def fold(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


_TEHSIL_WORDS = re.compile(r"\b(SUB-DIVISION|SUB DIVISION|SUB-TEHSIL|TEHSIL|TALUKA|TOWN)\b", re.I)


def fold_tehsil(s):
    return fold(_TEHSIL_WORDS.sub(" ", str(s)))


def similar(a, b):
    if not a or not b:
        return False
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def census():
    import pandas as pd

    df = pd.read_csv(NORM, keep_default_na=False, na_values=[""])
    df["province"] = df["note"].str.extract(r"province=([^;]+)")[0]
    df["district"] = df["note"].str.extract(r"district=([^;]+)")[0]
    d = df[df["geo_level"] == "district"]
    wide = d.pivot_table(index=["geo_id", "geo_name", "province"], columns="source_category",
                         values="count", aggfunc="sum").reset_index()
    cats = [c for c in wide.columns if c not in ("geo_id", "geo_name", "province")]
    wide["census_pop"] = wide[cats].sum(axis=1)
    t = df[df["geo_level"] == "tehsil"][["geo_name", "district", "province"]].drop_duplicates()
    tehsils = t.groupby(["province", "district"])["geo_name"].apply(list).to_dict()
    if len(wide) != DISTRICTS:
        raise SystemExit(f"{len(wide)} census districts in {NORM}, expected {DISTRICTS}")
    return wide, tehsils


def osm_polygons():
    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    j = json.load(open(OSM_JSON, encoding="utf-8"))
    by_id = {e["id"]: e for e in j["elements"]}
    rows = []
    for name, rid in OSM_KARACHI.items():
        e = by_id.get(rid)
        if e is None:
            raise SystemExit(f"OSM relation {rid} ({name}) missing from {OSM_JSON}")
        outer, inner = [], []
        for m in e.get("members", []):
            if m["type"] != "way" or "geometry" not in m:
                continue
            ls = LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
            (inner if m.get("role") == "inner" else outer).append(ls)
        polys = list(polygonize(unary_union(outer)))
        if not polys:
            raise SystemExit(f"OSM relation {rid} ({name}): outer ways do not close")
        geom = unary_union(polys)
        if inner:
            holes = list(polygonize(unary_union(inner)))
            if holes:
                geom = geom.difference(unary_union(holes))
        rows.append({"geo_name": name, "osm_id": rid,
                     "osm_name": e["tags"].get("name:en") or e["tags"].get("name"),
                     "subareas": [m["ref"] for m in e.get("members", []) if m["type"] == "relation"],
                     "geometry": geom})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def main():
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    if "--fetch" in sys.argv:
        fetch()
    for p in (NORM, os.path.join(COD_DIR, "pak_admin3.shp"), OSM_JSON):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run sources/pk_2023.py, then this with --fetch")
    os.makedirs(GEO, exist_ok=True)
    ok = True

    wide, tehsils = census()
    a2 = gpd.read_file(os.path.join(COD_DIR, "pak_admin2.shp"))
    a3 = gpd.read_file(os.path.join(COD_DIR, "pak_admin3.shp"))
    if len(a2) != COD_ADM2:
        raise SystemExit(f"COD ADM2 has {len(a2)} features, expected {COD_ADM2}")
    print(f"census districts {len(wide)}; COD-AB ADM2 {len(a2)} (valid_on "
          f"{sorted(set(a2['valid_on'].astype(str)))}), ADM3 {len(a3)}")

    # ---- 1. census name -> COD ADM2, within province, both directions
    karachi = set(OSM_KARACHI)
    cod_key = {(r.adm1_name, fold(r.adm2_name)): r.adm2_name for r in a2.itertuples()}
    match = {}
    for r in wide.itertuples():
        if r.geo_name in karachi:
            continue
        base = r.geo_name[:-len(" DISTRICT")] if r.geo_name.endswith(" DISTRICT") else r.geo_name
        want = ALIAS.get(base, base)
        hit = cod_key.get((PROVINCE[r.province], fold(want)))
        match[r.geo_name] = hit
    miss_c = sorted(k for k, v in match.items() if v is None)
    used = [v for v in match.values() if v]
    dup = sorted({v for v in used if used.count(v) > 1})
    expect_cod = set(a2.loc[~a2["adm1_name"].isin(COD_OUTSIDE_CENSUS), "adm2_name"]) \
        - COD_KARACHI - COD_DISSOLVED
    miss_p = sorted(expect_cod - set(used))
    good = not miss_c and not dup and not miss_p
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(used)} census districts -> COD ADM2 by name within "
          f"province, 1:1 ({len(miss_c)} census unmatched {miss_c}, {len(miss_p)} COD unmatched "
          f"{miss_p}, {len(dup)} COD used twice {dup}); {len(ALIAS)} via ALIAS")
    print(f"      not joined by name, on purpose: 7 Karachi districts (OSM), COD's Lehri "
          f"(split by tehsil), and COD's {int(a2['adm1_name'].isin(COD_OUTSIDE_CENSUS).sum())} "
          f"AJK/GB districts (outside the census)")

    # ---- 2. the second key: COD tehsil names against the census tehsils of the same district
    worst = []
    tot_cod = tot_hit = 0
    for cname, cod in sorted(match.items()):
        if cod is None:
            continue
        prov = wide.loc[wide["geo_name"] == cname, "province"].iloc[0]
        ct = [fold_tehsil(x) for x in tehsils.get((prov, cname), [])]
        dt = [fold_tehsil(x) for x in a3.loc[a3["adm2_name"] == cod, "adm3_name"]]
        hits = sum(any(similar(x, y) for y in ct) for x in dt)
        tot_cod += len(dt)
        tot_hit += hits
        worst.append((hits / max(len(dt), 1), hits, len(dt), cname, cod))
    worst.sort()
    zero = [w for w in worst if w[1] == 0 and w[2] >= 2 and w[3] not in TEHSIL_NAMES_DIFFER]
    good = not zero
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} second key: {tot_hit} of {tot_cod} COD tehsil names "
          f"({100.0 * tot_hit / tot_cod:.0f}%) are found among the census tehsils of the district "
          f"they were paired with; pairs with 2+ COD tehsils and none found, outside "
          f"TEHSIL_NAMES_DIFFER: {len(zero)} {[w[3] for w in zero]}")
    for k, why in TEHSIL_NAMES_DIFFER.items():
        print(f"      accepted: {k}: {why}")
    for w in worst[:8]:
        print(f"      {w[1]}/{w[2]}  {w[3]} -> {w[4]}")

    # ---- 3. Lehri, tehsil by tehsil
    a3["unit_name"] = a3["adm2_name"].map({v: k for k, v in match.items() if v})
    moved = 0
    for (d2, d3), target in TEHSIL_MOVES.items():
        sel = (a3["adm2_name"] == d2) & (a3["adm3_name"] == d3)
        if sel.sum() != 1:
            raise SystemExit(f"TEHSIL_MOVES: COD has {int(sel.sum())} tehsils {d3} under {d2}")
        a3.loc[sel, "unit_name"] = target
        moved += 1
    left = a3[a3["adm2_name"].isin(COD_DISSOLVED) & a3["unit_name"].isna()]
    good = left.empty
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} COD's Lehri split by tehsil: {moved} tehsils moved "
          f"(Lehri -> Sibi, Bhag -> Kachhi, as the census prints them), {len(left)} left over")

    # ---- 4. Karachi from OSM, inside COD's footprint
    kfoot = a2[a2["adm2_name"].isin(COD_KARACHI)]
    if len(kfoot) != len(COD_KARACHI):
        raise SystemExit(f"COD Karachi districts: found {len(kfoot)} of {len(COD_KARACHI)}")
    osm = osm_polygons()
    utm = 32642
    foot = kfoot.to_crs(utm).union_all()
    osm_u = osm.to_crs(utm).union_all()
    # COVERAGE, not intersection-over-union. The first version asserted IoU > 0.85 and got
    # 0.588: OSM's coastal districts (Keamari, Malir) carry their boundary out over the sea,
    # so OSM is 5,841 km2 against COD's 3,849 and IoU punishes water nobody lives on. What
    # matters is that OSM's districts cover the land COD calls Karachi; the excess is clipped.
    # AND NOT AREA AT ALL, in the end. Area coverage came out 93.2%, and a 0.95 bar on it was a
    # number picked before looking. The uncovered land is only a problem if people live on it,
    # so the assertion is on PEOPLE, in section 6, once the hexes are read: the Kontur population
    # inside COD's Karachi but outside every OSM district, which goes to the nearest district.
    cover = foot.intersection(osm_u).area / foot.area
    print(f"  --  OSM's 7 Karachi districts cover {100 * cover:.1f}% of COD's 6-district Karachi "
          f"footprint by area ({foot.area / 1e6:,.0f} km2); OSM's "
          f"{osm_u.difference(foot).area / 1e6:,.0f} km2 outside it (sea, mostly) is clipped off. "
          f"Asserted on population below.")
    towns = {}
    try:
        admin = json.load(open(os.path.join(RAW, "osm_karachi_admin.json"), encoding="utf-8"))
        towns = {e["id"]: (e["tags"].get("name:en") or e["tags"].get("name", ""))
                 for e in admin["elements"]}
    except (OSError, ValueError, KeyError):
        pass
    bad_town = []
    for r in osm.itertuples():
        names = [towns[s] for s in r.subareas if s in towns]
        ct = [fold_tehsil(x) for x in tehsils.get(("Sindh", r.geo_name), [])]
        found = [n for n in names if any(similar(fold_tehsil(n), y) for y in ct)]
        print(f"      {r.geo_name:26s} <- OSM {r.osm_id} {r.osm_name!r}: towns {names}, "
              f"{len(found)} of {len(names)} among the census sub-divisions")
        if names and not found:
            bad_town.append(r.geo_name)
    if towns:
        good = not bad_town
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} every Karachi pairing is confirmed by at least one "
              f"OSM town name among the census's own sub-divisions ({bad_town})")
    else:
        print("  --  osm_karachi_admin.json not on disk, Karachi town check skipped")
    kosm = gpd.overlay(osm.to_crs(utm)[["geo_name", "geometry"]],
                       gpd.GeoDataFrame(geometry=[foot], crs=utm), how="intersection")
    kosm = kosm.dissolve(by="geo_name").reset_index().to_crs(4326)
    kosm["built_from"] = "OSM admin_level=6, clipped to COD's Karachi footprint"

    # ---- 5. the unit polygons
    rest = a3[a3["unit_name"].notna()].copy()
    units = rest.dissolve(by="unit_name").reset_index()[["unit_name", "geometry"]]
    units = units.rename(columns={"unit_name": "geo_name"})
    units["built_from"] = "COD-AB v01 ADM3 tehsils"
    units = pd.concat([units, kosm[["geo_name", "geometry", "built_from"]]], ignore_index=True)
    units = gpd.GeoDataFrame(units, geometry="geometry", crs=4326)
    units = units.merge(wide, on="geo_name", how="left")
    good = len(units) == DISTRICTS and units["geo_id"].notna().all() and \
        units["geo_name"].is_unique and not units.geometry.is_empty.any()
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(units)} unit polygons, every one a census district "
          f"with a non-empty geometry (spec §8.1's three-way check)")
    if not ok:
        raise SystemExit("boundary join FAILED -- nothing written")

    units["unit"] = units["geo_id"]
    units[["unit", "geo_name", "province", "built_from", "census_pop", "geometry"]] \
        .to_file(OUT_UNITS, layer="districts", driver="GPKG")
    print(f"\nwrote {OUT_UNITS}")

    # ---- 6. Kontur, on hex centroids
    gpkg, unpacked = kontur_gpkg()
    hexes = gpd.read_file(gpkg)
    if len(hexes) == 0:
        raise SystemExit("Kontur gpkg read returned ZERO features")
    popcol = next(c for c in hexes.columns if c.lower() == "population")
    print(f"\nKontur hexes {len(hexes):,}, population {hexes[popcol].sum():,.0f}")
    pts = gpd.GeoDataFrame({popcol: hexes[popcol].to_numpy()},
                           geometry=hexes.geometry.centroid, crs=hexes.crs).to_crs(4326)
    j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].reindex(pts.index)
    # Karachi's footprint gaps between clipped OSM polygons: nearest Karachi district
    kf = gpd.GeoDataFrame(geometry=[kfoot.unary_union], crs=4326)
    inside_k = gpd.sjoin(pts[j["unit"].isna()], kf, how="inner", predicate="within").index
    if len(inside_k):
        kunits = units[units["geo_name"].isin(karachi)][["unit", "geometry"]].to_crs(utm)
        near = gpd.sjoin_nearest(pts.loc[inside_k].to_crs(utm), kunits, how="left")
        near = near[~near.index.duplicated(keep="first")]
        j.loc[near.index, "unit"] = near["unit"]
    k_pop = float(units.loc[units["geo_name"].isin(karachi), "census_pop"].sum())
    k_near = float(pts.loc[inside_k, popcol].sum())
    good = k_near < 0.02 * k_pop
    print(f"  {'OK ' if good else 'BAD'} Karachi hexes inside COD's footprint but outside every "
          f"OSM district, given to the nearest: {len(inside_k):,} hexes, {k_near:,.0f} people, "
          f"{100 * k_near / k_pop:.2f}% of Karachi's census population (bar: under 2%)")
    if not good:
        raise SystemExit("OSM's Karachi districts leave too many people unplaced -- nothing written")
    outside = j["unit"].isna()
    lost = float(pts.loc[outside, popcol].sum())
    print(f"  hexes in no drawn district: {int(outside.sum()):,} ({lost:,.0f} people, "
          f"{100.0 * lost / pts[popcol].sum():.2f}%) -- Azad Kashmir and Gilgit-Baltistan, which "
          f"the census does not cover, plus border overrun")
    keep = ~outside
    out = gpd.GeoDataFrame({"unit": j.loc[keep, "unit"].to_numpy(),
                            "pop": pts.loc[keep, popcol].to_numpy(dtype=float)},
                           geometry=hexes.geometry[keep.to_numpy()].to_crs(4326).to_numpy(),
                           crs=4326)
    per = out.groupby("unit")["pop"].agg(["size", "sum"])
    missing = sorted(set(units["unit"]) - set(per.index))
    zero = sorted(per.index[per["sum"] <= 0])
    if missing or zero:
        raise SystemExit(f"districts with no populated hex: {missing}; zero population: {zero}")
    print(f"  every one of the {len(units)} districts has hexes: {per['size'].min():,} to "
          f"{per['size'].max():,}")

    tot = float(out["pop"].sum())
    cen = float(units["census_pop"].sum())
    ratio = tot / cen
    print(f"\n  Kontur 2023-11 {tot:,.0f} vs census 2023 religion universe {cen:,.0f}: ratio "
          f"{ratio:.3f}, band [{KONTUR_RATIO_MIN}, {KONTUR_RATIO_MAX}]")
    if not KONTUR_RATIO_MIN <= ratio <= KONTUR_RATIO_MAX:
        raise SystemExit("national Kontur/census ratio out of band -- check the download or the join")

    lut = units[["unit", "geo_name", "province", "built_from", "census_pop"]].merge(
        per.rename(columns={"size": "hexes", "sum": "kontur_pop"}), left_on="unit",
        right_index=True)
    lut["kontur_over_census"] = lut["kontur_pop"] / lut["census_pop"]
    q = lut["kontur_over_census"].quantile([.01, .25, .5, .75, .99])
    print(f"  per district: median {q[.5]:.2f}, quartiles {q[.25]:.2f}-{q[.75]:.2f}, "
          f"1-99% {q[.01]:.2f}-{q[.99]:.2f}")
    srt = lut.sort_values("kontur_over_census")
    for label, sub in (("lowest", srt.head(6)), ("highest", srt.tail(6))):
        print(f"  {label}:")
        for r in sub.itertuples():
            print(f"     {r.kontur_over_census:5.2f}x  {r.geo_name[:30]:30s} {r.province[:12]:12s} "
                  f"census {int(r.census_pop):>10,}  ({r.built_from[:4]})")
    kk = lut[lut["geo_name"].isin(karachi)]
    print(f"  Karachi's seven: " + ", ".join(f"{r.geo_name.replace(' DISTRICT', '').title()} "
                                          f"{r.kontur_over_census:.2f}" for r in kk.itertuples()))

    # ---- 7. §12's witness: shares are spatially smooth; a scrambled join is not
    rng = np.random.default_rng(20260914)
    cen_pts = units.to_crs(utm).geometry.centroid
    xy = np.c_[cen_pts.x, cen_pts.y]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :5]
    for label, cols in (("Hindu (Hindu Jati + Scheduled Castes)", ["Hindu Jati", "Scheduled Castes"]),
                        ("Christian", ["Christian"])):
        share = (units[cols].sum(axis=1) / units["census_pop"]).to_numpy()
        r0 = np.corrcoef(share, share[nn].mean(axis=1))[0, 1]
        best = max(np.corrcoef(s, s[nn].mean(axis=1))[0, 1]
                   for s in (rng.permutation(share) for _ in range(200)))
        good = r0 > best
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} share vs its 5 nearest districts: r={r0:.2f}, "
              f"best of 200 shuffles {best:.2f}")
    if not ok:
        raise SystemExit("placement witness FAILED -- hexes not written")

    out.to_file(OUT_HEXES, layer="hexes", driver="GPKG")
    print(f"\nwrote {OUT_HEXES} ({len(out):,} hexes)")
    with open(OUT_LOOKUP, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "district", "province", "built_from", "census_pop_2023",
                    "kontur_pop_2023", "hexes", "kontur_over_census"])
        for r in lut.sort_values("unit").itertuples():
            w.writerow([r.unit, r.geo_name, r.province, r.built_from, int(r.census_pop),
                        round(r.kontur_pop, 1), int(r.hexes), round(r.kontur_over_census, 3)])
    print(f"wrote {OUT_LOOKUP}")
    if unpacked:
        os.remove(gpkg)
        print(f"removed the unpacked {GPKG_NAME} again (the .gz stays)")


if __name__ == "__main__":
    main()
