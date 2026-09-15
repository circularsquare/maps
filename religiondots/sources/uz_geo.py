"""Uzbekistan — boundaries and populations for the twelve regions, Karakalpakstan and Tashkent.

Writes data/geo/uz/uz_regions.gpkg and data/geo/uz/uz_lookup.csv.

OCHA COD-AB Uzbekistan (`cod-ab-uzb`), `uzb_admbnda_adm1_2018b.zip`: fourteen ADM1 features,
pcodes `UZ03`..`UZ35`. The newest COD-AB for Uzbekistan and dated 2018. Tashkent city took in
territory from Tashkent region after that (the city's Yangihayot district, 2020), so the
city's polygon is a little smaller than the population figure it carries. Placement inside
each region is by Kontur hexes (`sources/uz_grid.py`), so what that moves is which side of
the line a few suburban hexes land on, not how many people either unit has.

## THE POPULATION IS THE OFFICE'S OWN, 1 JANUARY 2026

National Statistics Committee, SIAT indicator 2.01.02.0001 *Permanent population, total*,
`api.siat.stat.uz/media/uploads/sdmx/sdmx_data_246.json`, linked from `stat.uz`'s demography
page and last modified 2026-04-22. Thousands of people to one decimal, per SOATO territory
code, years 2000-2026. The fourteen region rows sum exactly to the national row, 38,236.7
thousand. It is a register estimate carried forward from the 1989 census; the 2026 census
has published population by region only in a press release so far (`sources/uz.md` §6).

## THREE WITNESSES, AND THE THIRD IS ON THE SURVEY'S OWN LABELS

1. **The office's SOATO code IS COD's pcode.** `17NN` is `UZNN` for all fourteen. Arithmetic,
   no names involved.
2. **The Russian names agree** between the office table and COD's `ADM1_RU`, once `область`
   and `город`/`г.` are folded.
3. **The Central Asia Barometer's region codes run in the office's own row order.** The survey
   codes its regions 4001-4014 and the office's table lists its fourteen region rows in SOATO
   publication order (Karakalpakstan first, Tashkent city last, Tashkent region between
   Syrdarya and Fergana). `CAB_REGION` below is written by name; this witness reads the order
   off the office file and requires code `4000+k` to be its k-th row. On top of that, each
   CAB label has to name-match its own pcode best among the units of the same kind (city or
   region), by a margin. The kind rule is what separates `Toshkent Oblast` from
   `Toshkent Shari`, which no string similarity can.

Usage:
    python sources/uz_geo.py --fetch    one ~140 KB zip from HDX, one ~80 KB JSON from SIAT
    python sources/uz_geo.py            rebuild from data/raw/uz/
"""

import difflib
import json
import os
import re
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uz")
OUT_DIR = os.path.join(ROOT, "data", "geo", "uz")
OUT = os.path.join(OUT_DIR, "uz_regions.gpkg")
LOOKUP = os.path.join(OUT_DIR, "uz_lookup.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

COD_ZIP = "uzb_admbnda_adm1_2018b.zip"
COD_SHP = "uzb_admbnda_adm1_2018b.shp"
SIAT_JSON = "uz_siat_sdmx_246_population.json"
DOWNLOADS = {
    COD_ZIP: "https://data.humdata.org/dataset/66f56d39-3f19-4798-a0df-a0ed9a4836ea/resource/"
             "6a44247a-f4f5-421d-803a-893ecaf201e2/download/uzb_admbnda_adm1_2018b.zip",
    SIAT_JSON: "https://api.siat.stat.uz/media/uploads/sdmx/sdmx_data_246.json",
}

N_UNITS = 14
POP_YEAR = "2026"                 # "на начало года": 1 January 2026
NATIONAL_TENTHS = 382_367         # 38,236.7 thousand, the file's own national row

# The barometer's Region_M labels, verbatim (including its `Respulbikasi`), against the pcode.
CAB_REGION = {
    4001: ("Qoraqalpogiston Respulbikasi", "UZ35"),
    4002: ("Andijan Oblast", "UZ03"),
    4003: ("Bukhara Oblast", "UZ06"),
    4004: ("Djizak Oblast", "UZ08"),
    4005: ("Qashqadaryo Oblast", "UZ10"),
    4006: ("Navoiy Oblast", "UZ12"),
    4007: ("Namangan Oblast", "UZ14"),
    4008: ("Samarqand Oblast", "UZ18"),
    4009: ("Surxandaryo Oblast", "UZ22"),
    4010: ("Sirdaryo Oblast", "UZ24"),
    4011: ("Toshkent Oblast", "UZ27"),
    4012: ("Farg'ona Oblast", "UZ30"),
    4013: ("Xorazm Oblast", "UZ33"),
    4014: ("Toshkent Shari", "UZ26"),
}

EN_NAME = {
    "UZ03": "Andijan", "UZ06": "Bukhara", "UZ08": "Jizzakh", "UZ10": "Kashkadarya",
    "UZ12": "Navoi", "UZ14": "Namangan", "UZ18": "Samarkand", "UZ22": "Surkhandarya",
    "UZ24": "Syrdarya", "UZ26": "Tashkent city", "UZ27": "Tashkent region", "UZ30": "Fergana",
    "UZ33": "Khorezm", "UZ35": "Karakalpakstan",
}


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


def office_population():
    """The fourteen region rows of SIAT 2.01.02.0001, in the file's own order."""
    d = json.load(open(os.path.join(RAW, SIAT_JSON), encoding="utf-8"))
    meta = {m["name_en"]: m["value_en"] for m in d[0]["metadata"]}
    if meta.get("Indicator identification number (code)") != "2.01.02.0001":
        raise SystemExit(f"{SIAT_JSON} is not indicator 2.01.02.0001: {meta}")
    rows = d[0]["data"]
    nat = [r for r in rows if r["Code"] == "1700"]
    reg = [r for r in rows if re.fullmatch(r"17\d\d", r["Code"]) and r["Code"] != "1700"]
    if len(nat) != 1 or len(reg) != N_UNITS:
        raise SystemExit(f"{len(nat)} national and {len(reg)} region rows, expected 1 and "
                         f"{N_UNITS}")
    if round(nat[0][POP_YEAR] * 10) != NATIONAL_TENTHS:
        raise SystemExit(f"the office's {POP_YEAR} national total is now {nat[0][POP_YEAR]} "
                         "thousand; the file has been revised. Update NATIONAL_TENTHS "
                         "deliberately, and the population quoted in countries.py with it.")
    tenths = sum(round(r[POP_YEAR] * 10) for r in reg)
    if tenths != NATIONAL_TENTHS:
        raise SystemExit(f"the fourteen regions sum to {tenths / 10} thousand against the "
                         f"national {NATIONAL_TENTHS / 10}")
    out = pd.DataFrame({
        "office_code": [r["Code"] for r in reg],
        "office_ru": [r["Klassifikator_ru"] for r in reg],
        "pop": [int(round(r[POP_YEAR] * 1000)) for r in reg],
    })
    print(f"  the office's permanent population, 1 January {POP_YEAR}: "
          f"{NATIONAL_TENTHS * 100:,} across {N_UNITS} regions, which sum to it exactly "
          f"(last modified {meta.get('Last modified date')})")
    return out


def _fold_ru(s):
    s = str(s).lower().replace("город", "г").replace("область", "")
    return re.sub(r"[\s.]+", "", s)


def _fold_lat(s):
    s = str(s).lower()
    s = re.sub(r"['’‘`ʻ]", "", s)
    s = re.sub(r"\b(oblast|viloyati|respulbikasi|respublikasi|republic of|shari|shahri|city)\b",
               " ", s)
    s = s.replace("ш.", " ")
    return re.sub(r"\s+", "", s)


def _is_city(s):
    return bool(re.search(r"(?i)\bshari\b|\bshahri\b|\bcity\b|ш\.", str(s)))


def cab_decode_witness(g, office):
    """WITNESS 3, in two halves: the code order, then the names within a kind."""
    order = office["office_code"].tolist()
    by_code = {f"17{p[2:]}": p for p in EN_NAME}
    bad = []
    for k, code in enumerate(sorted(CAB_REGION)):
        want = by_code[order[k]]
        if CAB_REGION[code][1] != want:
            bad.append((code, CAB_REGION[code], order[k], want))
    if bad:
        raise SystemExit(f"CAB region codes do not follow the office's row order: {bad}")
    print(f"  witness 3a: CAB codes 4001-4014 are the office table's {N_UNITS} region rows in "
          "the office's own order")

    cod = g.set_index("pcode")
    worst_margin = 1.0
    for code, (label, pcode) in CAB_REGION.items():
        city = _is_city(label)
        pool = [p for p in cod.index if (cod.loc[p, "ADM1TYPE_E"] == "independent city") == city]
        score = {p: max(difflib.SequenceMatcher(None, _fold_lat(label),
                                                _fold_lat(cod.loc[p, col])).ratio()
                        for col in ("ADM1_UZ", "ADM1_EN")) for p in pool}
        ranked = sorted(score, key=lambda p: -score[p])
        margin = score[ranked[0]] - (score[ranked[1]] if len(ranked) > 1 else 0.0)
        if ranked[0] != pcode or score[ranked[0]] < 0.6 or margin < 0.1:
            raise SystemExit(f"{label!r} is written against {pcode} but name-matches "
                             f"{[(p, round(score[p], 3)) for p in ranked[:3]]}")
        worst_margin = min(worst_margin, margin)
    print(f"  witness 3b: each CAB label name-matches its own pcode best among units of its "
          f"kind, worst margin {worst_margin:.3f}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    zpath = os.path.join(RAW, COD_ZIP)
    if not os.path.exists(zpath):
        raise SystemExit(f"{zpath} missing — run with --fetch")
    with zipfile.ZipFile(zpath) as z:
        if COD_SHP not in z.namelist():
            raise SystemExit(f"{COD_SHP} not in {COD_ZIP}: {z.namelist()}")
    g = gpd.read_file(f"zip://{zpath}!{COD_SHP}")
    if len(g) != N_UNITS or g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"{len(g)} ADM1 features in {g.crs}, expected {N_UNITS} in EPSG:4326")
    g["pcode"] = g["ADM1_PCODE"].astype(str).str.strip()
    if set(g["pcode"]) != set(EN_NAME):
        raise SystemExit(f"COD's pcodes changed: {sorted(set(g['pcode']) ^ set(EN_NAME))}")
    print(f"read {COD_ZIP}: {len(g)} regions, {g.crs}")

    office = office_population()
    office["pcode"] = "UZ" + office["office_code"].str[2:]
    if set(office["pcode"]) != set(g["pcode"]):
        raise SystemExit(f"SOATO 17NN -> UZNN does not pair: "
                         f"{sorted(set(office['pcode']) ^ set(g['pcode']))}")
    print(f"  witness 1: the office's SOATO `17NN` and COD's `UZNN` pair all {N_UNITS} by "
          "arithmetic")

    cod_ru = dict(zip(g["pcode"], g["ADM1_RU"]))
    bad = [(r.pcode, r.office_ru, cod_ru[r.pcode]) for r in office.itertuples()
           if _fold_ru(r.office_ru) != _fold_ru(cod_ru[r.pcode])]
    if bad:
        raise SystemExit(f"the office's Russian names do not match COD's: {bad}")
    print(f"  witness 2: all {N_UNITS} Russian names agree between the office and COD")

    cab_decode_witness(g, office)

    g = g.merge(office[["pcode", "office_code", "pop"]], on="pcode", how="left")
    area = g.to_crs("ESRI:54009").geometry.area / 1e6
    g["density"] = g["pop"] / area
    order = g.sort_values("density", ascending=False)["pcode"].map(EN_NAME).tolist()
    print(f"    densest to sparsest: {', '.join(order)}")
    if order[:2] != ["Tashkent city", "Andijan"] or order[-1] not in ("Navoi",
                                                                      "Karakalpakstan"):
        raise SystemExit(f"density order {order}: expected Tashkent city then Andijan on top "
                         "and Navoi or Karakalpakstan last; the population join is permuted")

    g["unit"] = g["pcode"]
    g["geo_id"] = g["pcode"]
    g["name"] = g["pcode"].map(EN_NAME)
    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "pcode", "geo_id", "pop", "geometry"]].to_file(OUT, layer="regions",
                                                                    driver="GPKG")
    print(f"\nwrote {OUT} ({len(g)} polygons)")

    cab_of = {p: (c, lab) for c, (lab, p) in CAB_REGION.items()}
    lut = g[["geo_id", "unit", "name", "office_code", "pop"]].copy()
    lut["name_ru"] = lut["geo_id"].map(cod_ru)
    lut["cab_code"] = lut["geo_id"].map(lambda p: cab_of[p][0])
    lut["cab_region"] = lut["geo_id"].map(lambda p: cab_of[p][1])
    lut = lut.sort_values("cab_code")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} ({len(lut)} rows)")
    print(lut[["geo_id", "name", "cab_code", "cab_region", "pop"]].to_string(index=False))


if __name__ == "__main__":
    main()
