"""Turkmenistan — boundaries and populations for the five velayats and Ashgabat city.

Writes data/geo/tm/tm_regions.gpkg and data/geo/tm/tm_lookup.csv.

## THE BOUNDARIES ARE KONTUR'S OSM EXTRACT OF 7 APRIL 2022

There is no OCHA COD-AB for Turkmenistan (`cod-ab-tkm` is a 404 on HDX). The two open options:

* **geoBoundaries gbOpen `TKM-ADM1`** (Wikimedia, "2007"). Its API says six units and the
  GeoJSON holds **five**: no Ashgabat, and Ahal spelt `Ahai`. Not usable.
* **Kontur Boundaries, `kontur_boundaries_TM_20220407.gpkg`** (HDX `kontur-boundaries-turkmenistan`,
  ODbL), OpenStreetMap's administrative relations. Six features at `admin_level` 4, the five
  velayats and Ashgabat, with Ashgabat at 979 km2, which is the city after its 2013 expansion
  into Ahal. Used.

Kontur's 2023-06-28 release is also on HDX and also has six units at level 4. The 2022 one is
kept because it is the vintage the census was taken on: **Arkadag city** was given velayat
status in 2023, and the census counts it inside Ahal (table 1.5: 567 people in December 2022).

## THE POPULATION IS THE 2022 CENSUS, TABLE 1.3

State Committee of Turkmenistan on Statistics, *Results of the Complete Population and Housing
Census of Turkmenistan 2022*, section 1 (`stat.gov.tm/population-census-pdfs/results/en/1.pdf`),
census day 17 December 2022. Table 1.3 gives each unit's total, urban and rural population.

**Do not use the figures in the press coverage** (Turkmenportal, THE AsiaN: Mary 1,616,246,
Dashoguz 1,552,725 and so on). They also sum to 7,057,841, and each one is the rounded
percentage printed on the report's map (22.9%, 22.0%, ...) multiplied by the national total.
Table 1.3's own figures are the counts. The transcription below is asserted four ways: each
unit's urban plus rural is its total; the six totals, urban and rural columns sum to table 1.1's
national rows; every urban share matches the map on page 1 to two decimals; and every figure
appears in the PDF's text layer.

The barometer's own weights target the State Statistics Committee's 1995 figures (wave 4
methods report, Table 6 and p. 45), which is why they are not the population base.

## THE JOIN HAS THREE WITNESSES, AND ONE OF THEM USES NO NAMES

1. **Names.** The barometer's `Region_M` labels, the census rows and OSM's `name_en` each
   name-match one unit, uniquely, with a margin.
2. **Position.** Ashgabat is the smallest polygon; Balkan's centroid is the westernmost,
   Lebap's the easternmost, Dashoguz's the northernmost and Mary's the southernmost.
3. **The census's urban share against the barometer's own design.** Asserted in
   `sources/tm.py` as the held-out check: each unit's share of the survey against its share
   of the census, over all 720 orderings.

Usage:
    python sources/tm_geo.py --fetch    one ~390 KB gzipped gpkg from Kontur, one ~2.6 MB PDF
    python sources/tm_geo.py            rebuild from data/raw/tm/
"""

import difflib
import gzip
import os
import re
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tm")
OUT_DIR = os.path.join(ROOT, "data", "geo", "tm")
OUT = os.path.join(OUT_DIR, "tm_regions.gpkg")
LOOKUP = os.path.join(OUT_DIR, "tm_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

BOUND_GZ = "kontur_boundaries_TM_20220407.gpkg.gz"
BOUND = "kontur_boundaries_TM_20220407.gpkg"
CENSUS_PDF = "tm_census2022_results_en_1.pdf"
DOWNLOADS = {
    BOUND_GZ: "https://geodata-eu-central-1-kontur-public.s3.eu-central-1.amazonaws.com/"
              "kontur_datasets/kontur_boundaries_TM_20220407.gpkg.gz",
    CENSUS_PDF: "https://stat.gov.tm/population-census-pdfs/results/en/1.pdf",
}

N_UNITS = 6

# Table 1.3, "Distribution of the number of population of Turkmenistan by sex and regions,
# persons": both sexes, then urban settlements and rural areas.
NATIONAL = (7_057_841, 3_321_497, 3_736_344)
CENSUS = {
    "TM-S": ("Ashgabat city", 1_030_063, 1_030_063, 0),
    "TM-A": ("Ahal velayat", 886_845, 313_785, 573_060),
    "TM-B": ("Balkan velayat", 529_895, 435_090, 94_805),
    "TM-D": ("Dashoguz velayat", 1_550_354, 473_861, 1_076_493),
    "TM-L": ("Lebap velayat", 1_447_298, 656_021, 791_277),
    "TM-M": ("Mary velayat", 1_613_386, 412_677, 1_200_709),
}
# The map on page 1 of the same section, "Urban population - N%".
MAP_URBAN_PCT = {"TM-S": 100.00, "TM-A": 35.38, "TM-B": 82.11, "TM-D": 30.56, "TM-L": 45.33,
                 "TM-M": 25.58}

# Section 4 (`results/en/4.pdf`), tables 4.1 and 4.3-4.8, "National composition of the
# population", total population, persons, in the tables' own row order. `sources/tm.py`
# post-stratifies the survey on these (sources/tm.md explains why). The report's tables print
# the sixteen rows as "including the most numerous" and they sum to each unit's total.
NATIONALITIES = ["Turkmens", "Uzbeks", "Russians", "Balochi", "Azerbaijanis", "Armenians",
                 "Kazakhs", "Persians", "Tatars", "Kurds", "Afghans", "Ukrainians",
                 "Karakalpaks", "Lezgins", "Koreans", "other nationalities"]
NATIONALITY = {
    "TM": [6_120_854, 642_476, 114_447, 87_503, 26_576, 14_711, 11_825, 10_997, 8_643, 2_739,
           2_655, 2_566, 2_371, 2_292, 1_015, 6_171],
    "TM-S": [925_656, 5_179, 68_188, 184, 10_376, 9_761, 703, 584, 2_585, 2_159, 101, 1_460,
             31, 510, 164, 2_422],
    "TM-A": [874_431, 1_774, 2_154, 923, 1_135, 143, 148, 5_479, 181, 113, 107, 50, 13, 24, 4,
             166],
    "TM-B": [496_541, 2_895, 14_412, 338, 7_389, 2_150, 2_663, 38, 1_127, 21, 15, 306, 19,
             1_479, 13, 489],
    "TM-D": [1_046_202, 489_453, 2_907, 174, 402, 112, 5_796, 47, 1_456, 57, 8, 98, 2_272, 24,
             646, 700],
    "TM-L": [1_292_180, 136_499, 11_791, 500, 938, 397, 1_793, 131, 1_653, 12, 99, 297, 15, 102,
             168, 723],
    "TM-M": [1_485_844, 6_676, 14_995, 85_384, 6_336, 2_148, 722, 4_718, 1_641, 377, 2_325, 355,
             21, 153, 20, 1_671],
}
NATIONALITY_PDF = "tm_census2022_results_en_4.pdf"
DOWNLOADS[NATIONALITY_PDF] = "https://stat.gov.tm/population-census-pdfs/results/en/4.pdf"
NATIONALITY_CSV = os.path.join(OUT_DIR, "tm_nationality.csv")

# The barometer's Region_M labels, verbatim (waves 4-6 and 14 agree).
CAB_REGION = {
    5001: ("Ashgabat", "TM-S"),
    5002: ("Ahal Region", "TM-A"),
    5003: ("Balkan Region", "TM-B"),
    5004: ("Dasoguz Region", "TM-D"),
    5005: ("Lebap Region", "TM-L"),
    5006: ("Mary Region", "TM-M"),
}

EN_NAME = {"TM-S": "Ashgabat", "TM-A": "Ahal", "TM-B": "Balkan", "TM-D": "Dashoguz",
           "TM-L": "Lebap", "TM-M": "Mary"}


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=300)
        r.raise_for_status()
        with open(dst + ".part", "wb") as f:
            f.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({os.path.getsize(dst):,} bytes)")
    gpkg = os.path.join(RAW, BOUND)
    if not os.path.exists(gpkg):
        with gzip.open(os.path.join(RAW, BOUND_GZ), "rb") as src, open(gpkg + ".part", "wb") as d:
            shutil.copyfileobj(src, d)
        os.replace(gpkg + ".part", gpkg)
        print(f"  unpacked {BOUND}")


def check_census():
    """The four assertions on the table 1.3 transcription (module docstring)."""
    for g, (name, tot, urb, rur) in CENSUS.items():
        if urb + rur != tot:
            raise SystemExit(f"{name}: urban {urb:,} + rural {rur:,} is not {tot:,}")
        pct = round(100.0 * urb / tot, 2)
        if abs(pct - MAP_URBAN_PCT[g]) > 0.005:
            raise SystemExit(f"{name}: urban share {pct}% against the map's {MAP_URBAN_PCT[g]}%")
    sums = tuple(sum(v[k] for v in CENSUS.values()) for k in (1, 2, 3))
    if sums != NATIONAL:
        raise SystemExit(f"table 1.3's units sum to {sums}, not the national {NATIONAL}")
    import fitz
    doc = fitz.open(os.path.join(RAW, CENSUS_PDF))
    text = " ".join(p.get_text() for p in doc)
    flat = re.sub(r"\s+", " ", text)
    missing = []
    for name, *figs in CENSUS.values():
        for f in figs:
            if f and f"{f:,}".replace(",", " ") not in flat:
                missing.append((name, f))
    if missing:
        raise SystemExit(f"figures not found in the PDF's text layer: {missing}")
    print(f"  census 2022 table 1.3: {NATIONAL[0]:,} people in {N_UNITS} units; urban + rural "
          "adds up in every row, the columns sum to table 1.1, the urban shares match the map, "
          "and every figure is in the PDF's text")


def check_nationality():
    """Tables 4.1 and 4.3-4.8: rows sum to each unit's table 1.3 total, columns to table 4.1,
    and every figure is in the PDF's text layer. Returns the long table."""
    for g, row in NATIONALITY.items():
        want = NATIONAL[0] if g == "TM" else CENSUS[g][1]
        if len(row) != len(NATIONALITIES) or sum(row) != want:
            raise SystemExit(f"nationality row {g} sums to {sum(row):,}, not {want:,}")
    for j, nat in enumerate(NATIONALITIES):
        s = sum(NATIONALITY[g][j] for g in CENSUS)
        if s != NATIONALITY["TM"][j]:
            raise SystemExit(f"{nat}: the six units sum to {s:,}, table 4.1 says "
                             f"{NATIONALITY['TM'][j]:,}")
    import fitz
    flat = re.sub(r"\s+", " ", " ".join(p.get_text() for p in
                                        fitz.open(os.path.join(RAW, NATIONALITY_PDF))))
    missing = [(g, f) for g, row in NATIONALITY.items() for f in row
               if f"{f:,}".replace(",", " ") not in flat]
    if missing:
        raise SystemExit(f"nationality figures not in the PDF's text layer: {missing}")
    rows = [(g, nat, NATIONALITY[g][j]) for g in CENSUS for j, nat in enumerate(NATIONALITIES)]
    print(f"  census 2022 tables 4.3-4.8: {len(NATIONALITIES)} nationalities in each unit; rows "
          "sum to table 1.3, columns to table 4.1, and every figure is in the PDF's text")
    return pd.DataFrame(rows, columns=["geo_id", "nationality", "persons"])


def _fold(s):
    s = str(s).lower()
    s = s.replace("ş", "sh").replace("ç", "ch").replace("ý", "y").replace("ä", "a")
    s = s.replace("ü", "u").replace("ö", "o").replace("ň", "n").replace("ž", "zh")
    s = re.sub(r"\b(region|velayat|welayaty|city|sheheri)\b", " ", s)
    s = s.replace("dasoguz", "dashoguz").replace("asgabat", "ashgabat")
    return re.sub(r"[^a-z]", "", s)


def _unique_match(label, candidates, what):
    """The pcode whose names match `label` best, by a margin, or stop."""
    score = {p: max(difflib.SequenceMatcher(None, _fold(label), _fold(n)).ratio() for n in names)
             for p, names in candidates.items()}
    ranked = sorted(score, key=lambda p: -score[p])
    margin = score[ranked[0]] - score[ranked[1]]
    if score[ranked[0]] < 0.8 or margin < 0.3:
        raise SystemExit(f"{what} {label!r} does not match one unit clearly: "
                         f"{[(p, round(score[p], 3)) for p in ranked[:3]]}")
    return ranked[0], margin


def main():
    if "--fetch" in sys.argv:
        fetch()
    path = os.path.join(RAW, BOUND)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing — run with --fetch")
    check_census()
    nationality = check_nationality()

    k = gpd.read_file(path)
    k = k[k["admin_level"].astype(str) == "4"].copy()
    if len(k) != N_UNITS or k.crs is None or k.crs.to_epsg() != 4326:
        raise SystemExit(f"{len(k)} admin_level 4 features in {k.crs}, expected {N_UNITS} in "
                         "EPSG:4326")

    # Witness 1: names, three ways, each unique.
    targets = {g: [EN_NAME[g], CENSUS[g][0]] for g in EN_NAME}
    worst = 1.0
    pcode = []
    for r in k.itertuples():
        p, m = _unique_match(r.name_en or r.name, targets, "OSM unit")
        _, m2 = _unique_match(r.name, targets, "OSM local name")
        pcode.append(p)
        worst = min(worst, m, m2)
    if sorted(pcode) != sorted(EN_NAME):
        raise SystemExit(f"OSM units do not pair one-to-one with the census: {pcode}")
    k["pcode"] = pcode
    for code, (label, p) in CAB_REGION.items():
        got, m = _unique_match(label, targets, "barometer region")
        if got != p:
            raise SystemExit(f"barometer {code} {label!r} is written against {p} but matches {got}")
        worst = min(worst, m)
    print(f"  witness 1: OSM's English and Turkmen names, the census rows and the barometer's "
          f"labels each pair one-to-one, worst margin {worst:.3f}")

    # Witness 2: position, no names.
    eq = k.to_crs(6933)
    k["km2"] = eq.geometry.area / 1e6
    cen = eq.geometry.centroid.to_crs(4326)
    k["lon"], k["lat"] = cen.x, cen.y
    s = k.set_index("pcode")
    checks = {"smallest": (s["km2"].idxmin(), "TM-S"), "westernmost": (s["lon"].idxmin(), "TM-B"),
              "easternmost": (s["lon"].idxmax(), "TM-L"),
              "northernmost": (s["lat"].idxmax(), "TM-D"),
              "southernmost": (s["lat"].idxmin(), "TM-M")}
    bad = {k_: v for k_, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"polygon positions contradict the names: {bad}")
    print("  witness 2: Ashgabat is the smallest polygon; Balkan, Lebap, Dashoguz and Mary are "
          "the westernmost, easternmost, northernmost and southernmost centroids")

    k["pop"] = k["pcode"].map(lambda g: CENSUS[g][1])
    k["urban"] = k["pcode"].map(lambda g: CENSUS[g][2])
    k["density"] = k["pop"] / k["km2"]
    order = k.sort_values("density", ascending=False)["pcode"].map(EN_NAME).tolist()
    print(f"    densest to sparsest: {', '.join(order)}")
    if order[0] != "Ashgabat" or order[-1] != "Balkan":
        raise SystemExit(f"density order {order}: expected Ashgabat first and Balkan last")

    k["unit"] = k["pcode"]
    k["geo_id"] = k["pcode"]
    k["name"] = k["pcode"].map(EN_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)
    k[["unit", "name", "pcode", "geo_id", "pop", "geometry"]].to_file(OUT, layer="regions",
                                                                    driver="GPKG")
    print(f"\nwrote {OUT} ({len(k)} polygons)")

    cab_of = {p: (c, lab) for c, (lab, p) in CAB_REGION.items()}
    lut = k[["geo_id", "unit", "name", "pop", "urban", "name_en", "km2"]].copy()
    lut = lut.rename(columns={"name_en": "osm_name"})
    lut["census_name"] = lut["geo_id"].map(lambda g: CENSUS[g][0])
    lut["cab_code"] = lut["geo_id"].map(lambda g: cab_of[g][0])
    lut["cab_region"] = lut["geo_id"].map(lambda g: cab_of[g][1])
    lut["km2"] = lut["km2"].round(1)
    lut = lut.sort_values("cab_code")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")
    nationality.to_csv(NATIONALITY_CSV, index=False, encoding="utf-8")
    print(f"wrote {NATIONALITY_CSV} ({len(nationality)} rows)")
    print(lut[["geo_id", "name", "cab_code", "cab_region", "pop", "urban", "km2"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
