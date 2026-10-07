"""Kazakhstan boundaries on the 1 Jan 2026 KATO geography.

Rayons come from OpenStreetMap (admin_level 6, fetched by fetch_osm.py),
because it is the only open source with the 2022-24 units: Zhanasemey,
Makanshy, Markakol, Ulken Naryn, Kosshy, Sauran, Alatau city, Astana's Nura.
OCHA's COD-AB (2023) predates most of them. OSM lacks two city districts:
Astana's Saraishyk is drawn as Astana minus its other five districts
(common.CITY_GAPS); Shymkent's four OSM districts predate Turan, so Shymkent
ships as one unit (common.MERGES).

Each OSM relation is placed in its region by where its interior point falls in
COD-AB's 20 regions (2023, already the post-2022 regions), then matched to the
KATO unit of the same Russian name in that region. Regions are dissolved from
the rayons so the two levels nest exactly.

English names: COD-AB's adm2 name of the unit that covers most of the OSM
polygon, when the two are the same unit both ways (each holds >= 70% of the
other's area); otherwise ENGLISH below.

Writes helper1m/data/kazakhstan/boundaries/adm1.gpkg and adm2.gpkg.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge, polygonize, unary_union

from common import (CITY_GAPS, DATA, MERGED_NAMES, RAW, REGIONS, REPO, COD_ADM1,
                    load_kato, norm, shipped_code)

GEOM_DIR = RAW / "osm_adm6"
COD = REPO / "data" / "asia1m" / "kazakhstan"
OUT = DATA / "boundaries"

# OSM relations whose Russian name does not match KATO 2026 by name or stem.
# Every one checked by hand against the KATO name; the position check in
# check_join (summed area per region) runs on all of them anyway.
OSM_IDS = {
    214837: "234800000",    # Кызылкогинский район = KATO Кзылкогинский район
    3338024: "391000000",   # Костанайская Г.А. = Костанай Г.А. (not Костанайский район)
    3397457: "155200000",   # Ойылский район = Уилский район
    8768984: "613800000",   # Жетисайский район = Жетысайский район
    17351846: "636300000",  # Улькен Нарынский район = район Үлкен Нарын
}

# English names for units COD-AB does not have as such.
ENGLISH = {
    "104100000": "Zhanasemey District",
    "104500000": "Makanshy District",
    "635500000": "Markakol District",
    "636300000": "Ulken Naryn District",
    "111600000": "Kosshy",
    "615500000": "Sauran District",
    "191800000": "Alatau",
    "191000000": "Konaev",
    "711510000": "Nura District",
    "711610000": "Saraishyk District",
    "711110000": "Almaty District",
    # COD-AB still carries the pre-rename names of these.
    "396400000": "Beimbet Mailin District",   # COD "Taran District"
    "274400000": "Baiterek District",         # COD "Zelenov District"
    "555200000": "Akkuly District",           # COD "Lebyazhye District"
    "635600000": "Samar District",            # COD "Samara District"
    "101000000": "Semey",
    "611000000": "Turkestan",
    "196800000": "Ile District",
    # COD-AB has these units but its outline differs from OSM's by more than
    # the 70% rule allows (a 2022-25 boundary change, or one file's error).
    "311000000": "Taraz",
    "635200000": "Kurchum District",
    "104600000": "Urzhar District",
    "111800000": "Stepnogorsk",
    "711210000": "Esil District",
    "271000000": "Oral (Uralsk)",
    "621800000": "Karazhal",
    "612000000": "Kentau",
}
# Cleanups of COD-AB spellings.
COD_FIXES = {
    "Termitau": "Temirtau",
    "Saran, Kazakhstan": "Saran",
    "Petropavlovsk Public Administration": "Petropavl",
    "Enbekshinsky District": "Enbekshi District",
    "Abaysky District": "Abay District",
    "Kurmangazy district": "Kurmangazy District",
}


def relation_polygon(rel):
    rings = {"outer": [], "inner": []}
    for m in rel.get("members", []):
        if m.get("type") != "way" or "geometry" not in m:
            continue
        role = "inner" if m.get("role") == "inner" else "outer"
        coords = [(p["lon"], p["lat"]) for p in m["geometry"] if p]
        if len(coords) >= 2:
            rings[role].append(LineString(coords))
    outer = unary_union(list(polygonize(linemerge(rings["outer"])))) if rings["outer"] else None
    if outer is None or outer.is_empty:
        return None
    if rings["inner"]:
        inner = unary_union(list(polygonize(linemerge(rings["inner"]))))
        outer = outer.difference(inner)
    return outer.buffer(0)


def main():
    kato = load_kato()
    l2 = kato[kato.level == 2]
    tags = json.loads((RAW / "osm_admin_tags.json").read_text(encoding="utf-8"))
    ids = [e["id"] for e in tags["elements"] if e["tags"].get("admin_level") == "6"]
    recs = []
    for i in ids:
        path = GEOM_DIR / f"{i}.json"
        if not path.exists():
            sys.exit(f"missing {path}; run fetch_osm.py")
        rel = json.loads(path.read_text(encoding="utf-8"))
        geom = relation_polygon(rel)
        if geom is None:
            sys.exit(f"relation {i} has no closed outer ring")
        t = rel["tags"]
        recs.append({"osm_id": i, "osm_name": t.get("name:ru") or t.get("name"),
                     "osm_en": t.get("name:en"), "geometry": geom})
    osm = gpd.GeoDataFrame(recs, crs="EPSG:4326")

    cod1 = gpd.read_file(COD / "kaz_admbnda_adm1_unhcr_2023.shp").to_crs("EPSG:4326")
    cod1["ab"] = cod1.ADM1_EN.map(COD_ADM1)
    pts = osm.copy()
    pts["geometry"] = osm.geometry.representative_point()
    j = gpd.sjoin(pts, cod1[["ab", "geometry"]], how="left", predicate="within")
    osm["ab"] = j.groupby(level=0).ab.first()
    if osm.ab.isna().any():
        sys.exit(f"relations outside every region: {osm[osm.ab.isna()].osm_name.tolist()}")

    codes, fails = [], []
    for _, r in osm.iterrows():
        if r.osm_id in OSM_IDS:
            codes.append(OSM_IDS[r.osm_id])
            continue
        key = norm(r.osm_name)
        cand = l2[(l2.ab == r.ab) & (l2.rus_name.map(norm) == key)]
        if len(cand) != 1:
            # Fall back to a prefix match on the stem ("Аккольский" ~ "Акколь").
            stem = key[:5]
            cand = l2[(l2.ab == r.ab) & l2.rus_name.map(norm).str.startswith(stem)]
            if len(cand) == 1:
                print(f"  stem match: {r.osm_name} -> {cand.rus_name.iloc[0]}")
        if len(cand) != 1:
            fails.append(f"OSM {r.osm_id} {r.osm_name!r} in {r.ab}: {cand.rus_name.tolist()}")
            codes.append(None)
        else:
            codes.append(cand.te.iloc[0])
    if fails:
        sys.exit("unmatched:\n" + "\n".join(fails))
    osm["kato"] = codes
    dup = osm[osm.kato.duplicated(keep=False)]
    if len(dup):
        sys.exit(f"KATO units claimed twice:\n{dup[['osm_id', 'osm_name', 'kato']]}")
    missing = sorted(set(l2.te) - set(osm.kato))
    print(f"{len(osm)} OSM rayons matched; KATO units without an OSM polygon: "
          f"{[(c, l2.set_index('te').rus_name[c]) for c in missing]}")

    # Units that are a city minus the districts OSM has (common.CITY_GAPS).
    for kcode, city_id in CITY_GAPS.items():
        rel = json.loads((GEOM_DIR / f"{city_id}.json").read_text(encoding="utf-8"))
        city = relation_polygon(rel)
        inside = osm[osm.kato.str[:2] == kcode[:2]]
        gap = gpd.GeoSeries([city.difference(unary_union(inside.geometry.tolist()))],
                            crs="EPSG:4326").explode(index_parts=False)
        km2 = gap.to_crs("ESRI:54009").area / 1e6
        gap = gap[km2.values >= 1.0]   # drop edge slivers
        print(f"  {kcode} {l2.set_index('te').rus_name[kcode]} = city {city_id} minus "
              f"{len(inside)} districts: {len(gap)} pieces, {km2[km2 >= 1.0].sum():.1f} km²")
        osm = pd.concat([osm, gpd.GeoDataFrame(
            [{"osm_id": -city_id, "osm_name": l2.set_index("te").rus_name[kcode],
              "osm_en": None, "ab": kcode[:2], "kato": kcode,
              "geometry": unary_union(gap.tolist())}], crs="EPSG:4326")], ignore_index=True)
        missing.remove(kcode)

    osm["code"] = osm.kato.map(shipped_code)
    covered = {shipped_code(c) for c in osm.kato}
    lost = [c for c in missing if shipped_code(c) not in covered]
    if lost:
        sys.exit(f"KATO units with no polygon even after merging: {lost}")

    # English names from COD-AB where the unit is the same both ways.
    cod2 = gpd.read_file(COD / "kaz_admbnda_adm2_unhcr_2023.shp").to_crs("EPSG:4326")
    eq = "ESRI:54009"
    o_eq, c_eq = osm.to_crs(eq), cod2.to_crs(eq)
    inter = gpd.overlay(o_eq[["osm_id", "geometry"]], c_eq[["ADM2_PCODE", "ADM2_EN", "geometry"]],
                        how="intersection", keep_geom_type=True)
    inter["a"] = inter.area
    inter["share_osm"] = inter.a / inter.osm_id.map(o_eq.set_index("osm_id").area)
    inter["share_cod"] = inter.a / inter.ADM2_PCODE.map(c_eq.set_index("ADM2_PCODE").area)
    best = inter.sort_values("a").groupby("osm_id").last()
    same = best[(best.share_osm >= 0.7) & (best.share_cod >= 0.7)]
    osm["name_en"] = osm.osm_id.map(same.ADM2_EN).map(lambda n: COD_FIXES.get(n, n) if isinstance(n, str) else n)
    for i, r in osm.iterrows():
        if r.kato in ENGLISH:
            osm.at[i, "name_en"] = ENGLISH[r.kato]
    nameless = osm[osm.name_en.isna()]
    if len(nameless):
        print("no English name (OSM name:en used):")
        for _, r in nameless.iterrows():
            print(f"  {r.kato} {r.osm_name} -> {r.osm_en}")
        osm.loc[osm.name_en.isna(), "name_en"] = osm.osm_en

    kname = l2.set_index("te").rus_name
    rows = []
    for code, g in osm.groupby("code"):
        geom = unary_union(g.geometry.tolist())
        if code in MERGED_NAMES:
            name, ru = MERGED_NAMES[code], " + ".join(kname[c] for c in sorted(g.kato))
            if code == "790000000":
                ru = "г.Шымкент (5 районов)"
        else:
            name, ru = g.name_en.iloc[0], kname[code]
        ab = code[:2]
        rows.append({"code": code, "name": name, "name_cn": ru,
                     "parent": f"{ab}0000000", "group": f"{ab}0000000", "geometry": geom})
    adm2 = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # OSM rayons are drawn round the cities they surround, without a hole
    # (Tselinograd district contains all of Astana; Kostanay district contains
    # Kostanay city). Smaller units win: each unit loses whatever any smaller
    # unit covers. Elsewhere this only trims slivers along shared borders.
    a2 = adm2.to_crs(eq)
    order = a2.area.sort_values().index
    placed, cut = None, {}
    for i in order:
        g = a2.geometry[i]
        before = g.area
        if placed is not None and g.intersects(placed):
            g = g.difference(placed).buffer(0)
        cut[i] = 1 - g.area / before
        placed = g if placed is None else unary_union([placed, g])
        a2.at[i, "geometry"] = g
    adm2 = a2.to_crs("EPSG:4326")
    big = sorted(((cut[i], adm2.code[i], adm2.name[i]) for i in cut if cut[i] > 0.001), reverse=True)
    print(f"adm2: {len(adm2)} units; {len(big)} lost more than 0.1% of their area to "
          f"smaller units inside them:")
    for share, code, name in big:
        print(f"    {code} {name}: {share:.1%}")

    adm1 = adm2.dissolve(by="group", as_index=False)[["group", "geometry"]]
    adm1["code"] = adm1.group
    adm1["name"] = adm1.code.str[:2].map(REGIONS)
    ru1 = kato[kato.level == 1].set_index("ab").rus_name
    adm1["name_cn"] = adm1.code.str[:2].map(ru1)
    adm1 = adm1[["code", "name", "name_cn", "group", "geometry"]]
    print(f"adm1: {len(adm1)} regions")

    OUT.mkdir(parents=True, exist_ok=True)
    adm1.to_file(OUT / "adm1.gpkg", driver="GPKG")
    adm2.to_file(OUT / "adm2.gpkg", driver="GPKG")
    osm.drop(columns="geometry").to_csv(OUT / "osm_match.csv", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
