"""Hungary — boundaries for the 3,177 census settlements.

Writes data/geo/hu/hu_settlements.gpkg and data/geo/hu/hu_lookup.csv.

**This is the Zagreb problem, and unlike Zagreb it is solved.** The census publishes
religion for Budapest's 23 kerület; GISCO LAU 2021 carries Budapest as ONE polygon of
1.72 million people. Croatia hit exactly this and had to give up, because nothing carried
Zagreb's districts (`sources/hr_geo.md`). Hungary's are in **geoBoundaries HUN ADM2**,
which is 198 units — the 174 járás plus the 23 kerület plus one — so Budapest, 17.9% of
the country, is drawn at census resolution instead of as one 525 km² blob.

Two sources, joined on nothing, which is the risk:

    3,154 settlements   GISCO LAU 2021, LAU_ID == the five-digit KSH code
       23 kerület       geoBoundaries HUN ADM2, named 'I. kerület' … 'XXIII. kerület'

The GISCO half needs no derivation at all and verifies perfectly: all 3,154 codes match and
all 3,154 NAMES agree with KSH's own codelist, which is the independent check §12 asks for.
The Budapest half is joined by Roman numeral against KSH's `Budapest NN. ker.`, a
deterministic mapping over 1–23, and checked by (a) each number appearing exactly once and
(b) the 23 polygons tiling GISCO's Budapest to within a fraction of a percent of its area.

**The districts are clipped to GISCO's Budapest polygon.** The two files are different
vintages (2017 OSM-derived against 2021 GISCO) and their city outlines differ by tens of
metres, so unclipped the districts spill over neighbouring settlements and dots would land
in Budaörs. Clipping makes the union of the 23 exactly the parent, which is the property
the map needs.

Usage:
    python sources/hu_geo.py --fetch    download geoBoundaries ADM2 (GISCO is already local)
    python sources/hu_geo.py
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "hu")
GEO_LAU = os.path.join(ROOT, "data", "geo", "lau2021")
SHP = os.path.join(GEO_LAU, "shp4326", "LAU_RG_01M_2021_4326.shp")
OUT_DIR = os.path.join(ROOT, "data", "geo", "hu")
OUT = os.path.join(OUT_DIR, "hu_settlements.gpkg")
LOOKUP = os.path.join(OUT_DIR, "hu_lookup.csv")
NORM = os.path.join(ROOT, "data", "normalized", "hu.csv")

ADM2_NAME = "geoBoundaries-HUN-ADM2.geojson"
ADM2_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
            "HUN/ADM2/geoBoundaries-HUN-ADM2.geojson")

STRUCTURE = os.path.join(RAW, "hu_structure_WBS003.json")

EXPECTED_GISCO = 3_155        # HU polygons in GISCO LAU 2021, Budapest as one
EXPECTED_SETTLEMENTS = 3_177  # census settlements, Budapest as 23
BUDAPEST_LAU = "13578"        # the one GISCO polygon the census replaces
N_DISTRICTS = 23

ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXI": 21,
    "XXII": 22, "XXIII": 23,
}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, ADM2_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", dest)
        return
    print("downloading", ADM2_URL)
    r = requests.get(ADM2_URL, timeout=600)
    r.raise_for_status()
    doc = r.json()                     # §5a: assert it is really GeoJSON, not a shell
    feats = doc.get("features", [])
    if len(feats) != 198:
        raise SystemExit(f"geoBoundaries HUN ADM2 has {len(feats)} features, expected 198")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    print(f"  {os.path.getsize(dest):,} bytes, {len(feats)} features")


def _census_settlements():
    """{ksh_code: name} for the 3,177 settlements, from KSH's own geography codelist.

    The codelist holds 3,178 five-digit codes, not 3,177. The extra one is `13578`
    **'Budapest kerületre nem bontható adatai'** — Budapest figures that cannot be broken
    down by district — and it is the residual §12 warns a census usually publishes.

    It carries NO religion rows, and that absence is the proof that drawing Budapest as its
    23 kerület loses nobody: had KSH put even one person there, the districts would not sum
    to the city. So it is asserted empty rather than quietly dropped.

    Its code is also, exactly, GISCO's LAU_ID for Budapest-as-one-polygon, which is what
    `BUDAPEST_LAU` is.
    """
    if not os.path.exists(STRUCTURE):
        raise SystemExit(f"missing {STRUCTURE} -- run sources/hu.py --fetch first")
    with open(STRUCTURE, encoding="utf-8") as fh:
        doc = json.load(fh)
    for cl in doc["data"].get("codelists", []):
        if cl["id"].startswith("CL_TERUL_GEO"):
            codes = {c["id"]: c.get("names", {}).get("hu", "")
                     for c in cl["codes"]
                     if c["id"].isdigit() and len(c["id"]) == 5}
            if BUDAPEST_LAU not in codes:
                raise SystemExit(f"{BUDAPEST_LAU} is no longer in the codelist -- KSH "
                                 "changed the Budapest residual, re-read hu_geo.md")
            print(f"  codelist carries {len(codes):,} five-digit codes; dropping "
                  f"{BUDAPEST_LAU} {codes[BUDAPEST_LAU]!r}")
            del codes[BUDAPEST_LAU]
            return codes
    raise SystemExit("no CL_TERUL_GEO* codelist in the structure message")


def main():
    if "--fetch" in sys.argv:
        fetch()

    import geopandas as gpd
    import pandas as pd
    from shapely.ops import unary_union

    if not os.path.exists(SHP):
        raise SystemExit(f"missing {SHP} -- unpack {GEO_LAU}/LAU_RG_01M_2021_4326.shp.zip")
    adm2_path = os.path.join(RAW, ADM2_NAME)
    if not os.path.exists(adm2_path):
        raise SystemExit(f"missing {adm2_path} -- run with --fetch")

    census = _census_settlements()
    print(f"census settlements: {len(census):,}")
    if len(census) != EXPECTED_SETTLEMENTS:
        raise SystemExit(f"expected {EXPECTED_SETTLEMENTS}")

    # ---------------------------------------------------------------- GISCO half
    gdf = gpd.read_file(SHP)
    hu = gdf[gdf["CNTR_CODE"] == "HU"].copy()
    hu["kod"] = hu["LAU_ID"].astype(str).str.zfill(5)
    print(f"GISCO HU polygons: {len(hu):,}")
    if len(hu) != EXPECTED_GISCO:
        raise SystemExit(f"expected {EXPECTED_GISCO} GISCO polygons")

    geo_keys, cen_keys = set(hu["kod"]), set(census)
    missing, extra = cen_keys - geo_keys, geo_keys - cen_keys
    print(f"\n  the join, both ways (§12 — a count match is not a join):")
    print(f"    matched                     {len(cen_keys & geo_keys):,}")
    print(f"    census units with no polygon  {len(missing):,}")
    print(f"    polygons with no census unit  {len(extra):,}")
    if extra != {BUDAPEST_LAU}:
        raise SystemExit(f"expected only Budapest {BUDAPEST_LAU} spare, got {sorted(extra)}")
    if len(missing) != N_DISTRICTS or not all(
            census[k].startswith("Budapest") for k in missing):
        raise SystemExit("the unmatched census units should be exactly Budapest's 23 "
                         f"kerület, got {sorted((k, census[k]) for k in missing)[:5]}")
    print(f"    OK  the only difference is Budapest: 1 polygon vs {N_DISTRICTS} districts")

    # independent verification of the derived key: the NAMES must agree (§12)
    gname = dict(zip(hu["kod"], hu["LAU_NAME"]))

    def fold(s):
        import unicodedata
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return "".join(ch for ch in s.lower() if ch.isalnum())

    dis = [k for k in cen_keys & geo_keys if fold(census[k]) != fold(gname[k])]
    print(f"    {'OK ' if not dis else 'BAD'} names agree on "
          f"{len(cen_keys & geo_keys) - len(dis):,} of {len(cen_keys & geo_keys):,} "
          "matched units")
    for k in dis[:10]:
        print(f"        {k}: census {census[k]!r} vs GISCO {gname[k]!r}")
    if dis:
        raise SystemExit("name verification FAILED — the key is not what it looks like")

    # ---------------------------------------------------------------- Budapest half
    adm2 = gpd.read_file(adm2_path)
    if adm2.crs is None:
        adm2 = adm2.set_crs(4326)
    adm2 = adm2.to_crs(hu.crs)
    ker = adm2[adm2["shapeName"].astype(str).str.strip().str.endswith("kerület")].copy()
    ker["num"] = (ker["shapeName"].astype(str).str.split(".").str[0].str.strip()
                  .map(ROMAN))
    if ker["num"].isna().any():
        raise SystemExit(f"unparsed district names: "
                         f"{ker[ker['num'].isna()]['shapeName'].tolist()}")
    if sorted(ker["num"]) != list(range(1, N_DISTRICTS + 1)):
        raise SystemExit(f"districts are not 1..{N_DISTRICTS}: {sorted(ker['num'])}")
    print(f"\n  geoBoundaries kerület: {len(ker)} , numbered "
          f"{ker['num'].min()}–{ker['num'].max()}, each once")

    # KSH names them 'Budapest 01. ker.' — map number -> census code
    bp_code = {}
    for k in missing:
        n = int(census[k].split()[1].rstrip("."))
        bp_code[n] = k
    if sorted(bp_code) != list(range(1, N_DISTRICTS + 1)):
        raise SystemExit(f"census Budapest districts are not 1..{N_DISTRICTS}")
    ker["kod"] = ker["num"].map(bp_code)
    ker["name"] = ker["num"].map(lambda n: census[bp_code[n]])

    # clip to GISCO's Budapest so the union of the 23 IS the parent polygon
    parent = hu[hu["kod"] == BUDAPEST_LAU].geometry.iloc[0]
    before = unary_union(ker.geometry.values)
    ker["geometry"] = ker.geometry.intersection(parent)
    ker = ker[~ker.geometry.is_empty]

    eq = gpd.GeoSeries([parent], crs=hu.crs).to_crs(23700)          # EOV, metres
    a_parent = eq.area.iloc[0] / 1e6
    a_before = gpd.GeoSeries([before], crs=hu.crs).to_crs(23700).area.iloc[0] / 1e6
    a_after = ker.set_geometry("geometry").to_crs(23700).area.sum() / 1e6
    print(f"    GISCO Budapest        {a_parent:8.2f} km²")
    print(f"    23 districts, raw     {a_before:8.2f} km²  "
          f"({100 * (a_before - a_parent) / a_parent:+.2f}%)")
    print(f"    23 districts, clipped {a_after:8.2f} km²  "
          f"({100 * (a_after - a_parent) / a_parent:+.2f}% — they tile the parent)")
    if len(ker) != N_DISTRICTS:
        raise SystemExit(f"{len(ker)} districts survived clipping, expected {N_DISTRICTS}")
    if abs(a_after - a_parent) / a_parent > 0.01:
        raise SystemExit("the clipped districts do not tile Budapest — check the vintages")

    # ---------------------------------------------------------------- combine
    keep = hu[hu["kod"] != BUDAPEST_LAU][
        ["kod", "LAU_NAME", "POP_2021", "AREA_KM2", "geometry"]].rename(
        columns={"LAU_NAME": "name", "POP_2021": "pop_2021", "AREA_KM2": "area_km2"})
    keep["level"] = "settlement"

    bp = ker[["kod", "name", "geometry"]].copy()
    # geoBoundaries carries no population; GISCO's is only used for reporting, never for
    # allocation, so leaving it absent for the 23 districts costs nothing.
    bp["pop_2021"] = float("nan")
    bp["area_km2"] = ker.to_crs(23700).area.values / 1e6
    bp["level"] = "city_district"

    out = gpd.GeoDataFrame(
        pd.concat([keep, bp], ignore_index=True), crs=hu.crs)
    if len(out) != EXPECTED_SETTLEMENTS:
        raise SystemExit(f"{len(out)} polygons, expected {EXPECTED_SETTLEMENTS}")
    if out["kod"].nunique() != len(out):
        raise SystemExit("duplicate kod in the combined layer")
    if set(out["kod"]) != cen_keys:
        raise SystemExit("the combined layer does not cover exactly the census settlements")

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_file(OUT, layer="settlements", driver="GPKG")
    print(f"\nwrote {OUT} ({len(out):,} polygons: "
          f"{len(keep):,} settlements + {len(bp)} Budapest districts)")

    # ---------------------------------------------------------------- lookup
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    s = df[df["geo_level"] == "settlement"][["geo_id"]].drop_duplicates().copy()
    s["kod"] = s["geo_id"].str.split("_").str[1]
    bad = s[~s["kod"].isin(set(out["kod"]))]
    if not bad.empty:
        raise SystemExit(f"{len(bad)} normalized settlements have no polygon: "
                         f"{bad['geo_id'].tolist()[:5]}")
    s.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(s):,} rows)")


if __name__ == "__main__":
    main()
