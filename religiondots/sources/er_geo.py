"""Eritrea: the 6 zobas, COD-AB polygons, each with a population that is a stated choice, since
Eritrea has never held a census.

Writes data/geo/er/er_zobas.gpkg and data/geo/er/er_lookup.csv. `sources/er.md` §4 is the record.

  * **boundaries**: COD-AB `cod-ab-eri` v01 (OCHA ROSEA, valid from 2020-04-27),
    `eri_admin1.geojson`: ER1-ER6, the six zobas of the 1996 reorganisation, the same six the
    surveys use.
  * **national total**: the UN World Population Prospects 2024 figure for 2020, 3,291,271, as
    carried in Pew Research Center's *Religious Composition 2010-2020* dataset
    (`data/raw/estimates/pew.zip`), which names WPP 2024 as its population source for Eritrea. The
    US Census Bureau's figure (6.4 million for 2025, the CIA Factbook's) is about twice it; the
    NSO has published no national total since the 2010 survey's frame.
  * **zoba shares**: EPHS 2010 final report Table 2-16, the weighted de jure household population
    by zoba (150,297 people), read back from the report's text layer on every run. Its weights
    carry the 2010 listing of villages and towns that the zobas compiled for the survey frame.
  * **witnesses, not used**: the same report's Table A-3, proportional allocation of 36,000
    households over the 2010 frame (households, not people); and COD-PS `cod-ps-eri`'s 2001
    figures (NSO, 2,908,795). Each zoba's survey share must sit within `FRAME_TOL` points of the
    2010 frame's household share (the binding check: same year, an independent column), and
    within `SHARE_TOL` of 2001's (a gross-error bar only: nine years apart, with a war and the
    return from Sudan between them, Gash-Barka +3.3, Maekel +2.6, Semenawi Keih Bahri -5.5).

THE JOIN is on COD-AB's own p-code, with the names asserted beside it.

Usage:
    python sources/er_geo.py --fetch    COD-AB, COD-PS 2001 and the EPHS 2010 report into data/raw/er/
    python sources/er_geo.py            rebuild from data/raw/er/
"""

import io
import os
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "er")
GEO = os.path.join(ROOT, "data", "geo", "er")
OUT = os.path.join(GEO, "er_zobas.gpkg")
LOOKUP = os.path.join(GEO, "er_lookup.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
COD_AB_URL = ("https://data.humdata.org/dataset/2025e742-8b5a-4f72-aac5-a28e96ce3cb8/resource/"
              "1e31c86b-8903-4720-a306-f9f60322a7ca/download/eri_admin_boundaries.geojson.zip")
COD_AB = os.path.join(RAW, "eri_admin_boundaries.geojson.zip")
COD_PS_URL = ("https://data.humdata.org/dataset/d50f2b35-5819-414a-93a5-7e30d036f958/resource/"
              "654cc7be-ce89-4ace-8567-47d075b1030a/download/eri_adm1_pop_2001_v2.csv")
COD_PS = os.path.join(RAW, "eri_adm1_pop_2001_v2.csv")
EPHS_URL = "https://www.afro.who.int/sites/default/files/2017-05/ephs2010_final_report_v4.pdf"
EPHS = os.path.join(RAW, "ephs2010_final_report_v4.pdf")
TABLE_2_16_PAGE = 67            # 0-based; printed p.40

# EPHS 2010 Table 2-16, `Population` column (weighted de jure), by COD-AB p-code.
EPHS_2010 = {
    "ER1": ("Debubawi Keih Bahri", 2_282), "ER2": ("Maekel", 32_527),
    "ER3": ("Semenawi Keih Bahri", 16_583), "ER4": ("Anseba", 22_307),
    "ER5": ("Gash-Barka", 34_953), "ER6": ("Debub", 41_644),
}
EPHS_TOTAL = 150_297
# Table A-3, column 2: 36,000 households allocated proportionally over the 2010 frame.
FRAME_2010_HH = {"ER4": 5_004, "ER6": 10_152, "ER1": 720, "ER5": 8_352, "ER2": 7_560, "ER3": 4_212}
COD_NAMES = {"ER1": "Debubawi Keih Bahri", "ER2": "Maekel", "ER3": "Semienawi Keih Bahri",
             "ER4": "Anseba", "ER5": "Gash Barka", "ER6": "Debub"}
NATIONAL_2020 = 3_291_271       # UN WPP 2024 for 2020, via Pew's dataset
FRAME_TOL = 1.0                 # points, survey 2010 against the 2010 frame's households
SHARE_TOL = 6.0                 # points, survey 2010 against COD-PS 2001: a gross-error bar


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for url, path in ((COD_AB_URL, COD_AB), (COD_PS_URL, COD_PS), (EPHS_URL, EPHS)):
        if os.path.exists(path) and os.path.getsize(path) > 100:
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
        print(f"  fetched {url} ({len(data):,} bytes)")


def check_table_2_16():
    import fitz

    page = fitz.open(EPHS)[TABLE_2_16_PAGE].get_text()
    if "Table 2-16" not in page or "de- jure population" not in page.replace("\n", " "):
        raise SystemExit("EPHS 2010 Table 2-16 is not on the pinned page")
    flat = " ".join(page.split())
    for code, (name, n) in EPHS_2010.items():
        if f"{n:,}" not in flat:
            raise SystemExit(f"Table 2-16: {name} {n:,} not found on the page")
    if f"{EPHS_TOTAL:,}" not in flat:
        raise SystemExit("Table 2-16's total is not on the page")
    s = sum(n for _, n in EPHS_2010.values())
    if abs(s - EPHS_TOTAL) > 2:
        raise SystemExit(f"Table 2-16's zobas sum to {s:,}, its total {EPHS_TOTAL:,}")
    print(f"  EPHS 2010 Table 2-16 read back: 6 zobas, {s:,} (total row {EPHS_TOTAL:,}, rounding)")


def pew_population():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    r = t[(t["Country"] == "Eritrea") & (t["Year"] == 2020)].iloc[0]
    got = int(r["Population"])
    if got != NATIONAL_2020:
        raise SystemExit(f"Pew's 2020 population for Eritrea is {got:,}, pinned {NATIONAL_2020:,}")
    return got


def main():
    import geopandas as gpd

    if "--fetch" in sys.argv or not all(os.path.exists(p) for p in (COD_AB, COD_PS, EPHS)):
        fetch()
    check_table_2_16()
    national = pew_population()

    with zipfile.ZipFile(COD_AB) as z:
        g = gpd.read_file(io.BytesIO(z.read("eri_admin1.geojson")))
    if len(g) != 6 or set(g["adm1_pcode"]) != set(EPHS_2010):
        raise SystemExit(f"COD-AB admin1: {len(g)} units, codes {sorted(g['adm1_pcode'])}")
    for _i, r in g.iterrows():
        if r["adm1_name"] != COD_NAMES[r["adm1_pcode"]]:
            raise SystemExit(f"{r['adm1_pcode']} is {r['adm1_name']!r}, expected {COD_NAMES[r['adm1_pcode']]!r}")

    tot = sum(n for _, n in EPHS_2010.values())
    share = {c: n / tot for c, (_, n) in EPHS_2010.items()}
    pop = {c: round(national * s) for c, s in share.items()}
    # largest remainder so the zobas sum exactly to the national figure
    diff = national - sum(pop.values())
    order = sorted(share, key=lambda c: (national * share[c]) % 1, reverse=(diff > 0))
    for c in order[:abs(diff)]:
        pop[c] += 1 if diff > 0 else -1
    assert sum(pop.values()) == national

    ps = pd.read_csv(COD_PS)
    ps_share = dict(zip(ps["ADM1_PCODE"], ps["T_TL"] / ps["T_TL"].sum()))
    hh = sum(FRAME_2010_HH.values())
    area = dict(zip(g["adm1_pcode"], g.to_crs(6933).area / 1e6))
    print(f"\n  zoba shares: EPHS 2010 (drawn) | 2010 frame households | COD-PS 2001 ({int(ps['T_TL'].sum()):,})")
    bad = []
    for c in sorted(EPHS_2010):
        d = 100 * (share[c] - ps_share[c])
        print(f"      {EPHS_2010[c][0]:<20} {100 * share[c]:6.2f}%  {100 * FRAME_2010_HH[c] / hh:6.2f}%  "
              f"{100 * ps_share[c]:6.2f}%  ({d:+.2f})   drawn {pop[c]:>9,}  {area[c]:>8,.0f} km2  "
              f"{pop[c] / area[c]:7.1f}/km2")
        if abs(100 * share[c] - 100 * FRAME_2010_HH[c] / hh) > FRAME_TOL:
            bad.append(f"{c} against the 2010 frame")
        if abs(d) > SHARE_TOL:
            bad.append(f"{c} against COD-PS 2001")
    if bad:
        raise SystemExit(f"zoba share witnesses fail: {bad}")

    os.makedirs(GEO, exist_ok=True)
    out = gpd.GeoDataFrame({"unit": g["adm1_pcode"], "name": [EPHS_2010[c][0] for c in g["adm1_pcode"]],
                            "pop": [pop[c] for c in g["adm1_pcode"]]}, geometry=g.geometry, crs=g.crs)
    out.to_file(OUT, layer="zobas", driver="GPKG")
    pd.DataFrame({"geo_id": out["unit"], "unit": out["unit"], "name": out["name"],
                  "pop": out["pop"]}).to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT} and {LOOKUP} (6 zobas, {national:,} people)")


if __name__ == "__main__":
    main()
