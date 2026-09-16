"""Libya: boundaries and Libyan populations for the 22 districts (muhafazat, shabiyat).

Writes data/geo/ly/ly_districts.gpkg and data/geo/ly/ly_lookup.csv.

Three sources:

  * **boundaries**: COD-AB `cod-ab-lby` v01 ADM2 (OCHA ROMENA, valid from 2018-05-07, reviewed
    2024-12-19), 22 features. ADM1 in that file is three regions (East, West, South).
  * **populations**: the Bureau of Statistics and Census (BSC), *تقدير السكان الليبيين حسب المناطق
    لسنة 2020* ("estimate of the Libyan population by region for 2020"), a one-page PDF linked from
    bsc.ly's front page: 22 regions and a total, 6,872,674. **Libyans only**, by its own title.
    No census has been held since 2006, and nothing newer by region was found (sources/ly.md §6).
  * **a witness and the older counts**: the U.S. Census Bureau's Libya workbook on HDX
    (`Libya_uscb_202304.xlsx`). Its `Population` sheet transcribes the same 2020 estimate as
    `POP_TTL_20`; its `Households` sheet sums the 2006 census (Table 1, Libyans and non-Libyans by
    mahalla) to the 22 districts; its `Nationality` sheet has the 2012 National Population Survey's
    non-Libyans by district.

## THE KEY IS BSC'S DISTRICT CODE

COD's `adm2_pcode` (LY0101 Derna ... LY0322 Murzuq) is the `NSO_CODE` the USCB workbook carries for
each district, so the workbook joins on code. The PDF has no codes: its rows are read in printed
order against `PDF_ORDER`, with the Arabic name on each extracted line as the witness. Checked:

  1. the PDF's 22 rows sum to its printed total, and every line carries the name `PDF_ORDER` says;
  2. **USCB's transcription agrees in 21 of 22 districts.** Al Marj is 227,658 in the PDF (and in
     the map on bsc.ly's front page, read 2026-09-15) and 286,045 in USCB's sheet, which is USCB's
     national sum's whole excess over the printed total (58,387). The PDF is drawn and the
     difference pinned (`USCB_DIFF`). Kontur holds 223,403 people in COD's Al Marj;
  3. COD's name under each p-code is the one `DISTRICTS` expects, and so is USCB's;
  4. COD's area against the 2006 census's land area per district is printed, not asserted. The
     eight north-western districts and Sabha and Ubari agree within 3%; the desert and eastern
     lines do not (Derna 0.67, Ghat 0.74, Murzuq 0.79, Benghazi 1.19, Ajdabiya 1.82), where the
     land between towns was drawn differently and holds few people. The rank witness that pins
     the join, Kontur people per district against the 2020 estimate, is in `sources/ly_grid.py`.

Usage:
    python sources/ly_geo.py --fetch    COD-AB geojson zip (0.5 MB), the BSC PDF, the USCB workbook
    python sources/ly_geo.py            rebuild from data/raw/ly/
"""

import io
import os
import re
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ly")
OUT_DIR = os.path.join(ROOT, "data", "geo", "ly")
OUT = os.path.join(OUT_DIR, "ly_districts.gpkg")
LOOKUP = os.path.join(OUT_DIR, "ly_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

COD_URL = ("https://data.humdata.org/dataset/fd9fb749-5b68-4942-b5f9-115f20b5c00e/resource/"
           "6933bac2-4f6b-4095-a629-21193981b80c/download/lby_admin_boundaries.geojson.zip")
COD_ZIP = os.path.join(RAW, "lby_admin_boundaries.geojson.zip")
BSC_URL = ("https://www.bsc.ly/wp-content/uploads/2024/02/"
           "%D8%AA%D9%82%D8%AF%D9%8A%D8%B1-%D8%A7%D9%84%D8%B3%D9%83%D8%A7%D9%86-"
           "%D8%A7%D9%84%D9%84%D9%8A%D8%A8%D9%8A%D9%8A%D9%86-%D8%AD%D8%B3%D8%A8-"
           "%D8%A7%D9%84%D9%85%D9%86%D8%A7%D8%B7%D9%82-%D9%84%D8%B3%D9%86%D8%A9-2020.pdf")
BSC_PDF = os.path.join(RAW, "bsc_libyans_by_region_2020.pdf")
USCB_URL = ("https://data.humdata.org/dataset/bb33e978-7b11-4d0d-a3f0-5bf50360985a/resource/"
            "2f7ad947-0f64-4930-aecd-ac0b0692e89f/download/libya_uscb_202304.xlsx")
USCB_XLSX = os.path.join(RAW, "libya_uscb_202304.xlsx")

TOTAL_2020 = 6_872_674          # the PDF's المجموع row
# bsc.ly's front page also prints "تقديرات السكان, 2020: 6875635", 2,961 above the PDF's total.
# Not used; recorded in sources/ly.md §6.
USCB_DIFF = {"LY0102": 58_387}  # USCB POP_TTL_20 minus the PDF, per district; asserted exactly
CENSUS_2006 = 5_657_692         # bsc.ly's census list, and the Households sheet's national row
NONLIBYAN_2006 = 359_540
NONLIBYAN_2012 = 187_372

# p-code -> (the name this map prints, COD adm2_name, USCB AREA_NAME)
DISTRICTS = {
    "LY0101": ("Derna", "Derna", "DARNAH"),
    "LY0102": ("Al Marj", "Almarj", "AL MARJ"),
    "LY0103": ("Benghazi", "Benghazi", "BANGHĀZĪ"),
    "LY0104": ("Tobruk", "Tobruk", "AL BUŢNĀN"),
    "LY0105": ("Ajdabiya", "Ejdabia", "AL WĀḨĀT"),
    "LY0106": ("Al Jabal al Akhdar", "Al Jabal Al Akhdar", "AL JABAL AL AKHḐAR"),
    "LY0107": ("Kufra", "Alkufra", "AL KUFRAH"),
    "LY0208": ("Sirte", "Sirt", "SURT"),
    "LY0209": ("Nalut", "Nalut", "NĀLŪT"),
    "LY0210": ("Murqub", "Almargeb", "AL MARQAB"),
    "LY0211": ("Tripoli", "Tripoli", "ŢARĀBULUS"),
    "LY0212": ("Jafara", "Aljfara", "AL JAFĀRAH"),
    "LY0213": ("Zawiya", "Azzawya", "AZ ZĀWIYAH"),
    "LY0214": ("Misrata", "Misrata", "MIŞRĀTAH"),
    "LY0215": ("Zuwara", "Zwara", "AN NUQĀŢ AL KHAMS"),
    "LY0216": ("Al Jabal al Gharbi", "Al Jabal Al Gharbi", "AL JABAL AL GHARBĪ"),
    "LY0317": ("Jufra", "Aljufra", "AL JUFRAH"),
    "LY0318": ("Wadi al Shati", "Wadi Ashshati", "WĀDĪ ASH SHĀŢI’"),
    "LY0319": ("Sabha", "Sebha", "SABHĀ"),
    "LY0320": ("Ubari", "Ubari", "WĀDĪ AL ḨAYĀT"),
    "LY0321": ("Ghat", "Ghat", "GHĀT"),
    "LY0322": ("Murzuq", "Murzuq", "MURZUQ"),
}

# The PDF's rows in printed order, each with the fragments its extracted line must contain.
# PyMuPDF's text layer garbles some names (سرت comes out رست, مصراته as مضاته); the fragments are
# as extracted. "المنطقة الغربية" is BSC's name for Zuwara's district, which its footnote defines
# as Zuwara, Sabratha, Al Ajaylat, Riqdalin and Al Jumayl; "اجدابيا والواحات" is Ajdabiya and the
# Oases (USCB's Al Wahat); "وادي الحياة" is Ubari's.
PDF_ORDER = [
    ("LY0104", ("طبر",)), ("LY0101", ("درنة",)), ("LY0106", ("الجبل", "خضر")),
    ("LY0102", ("المرج",)), ("LY0103", ("بنغازي",)), ("LY0105", ("اجدابيا", "الواحات")),
    ("LY0107", ("الكفرة",)), ("LY0208", ("رست",)), ("LY0317", ("الجفرة",)),
    ("LY0214", ("مضاته",)), ("LY0210", ("المرقب",)), ("LY0211", ("طرابلس",)),
    ("LY0212", ("الجفارة",)), ("LY0213", ("الزاوية",)), ("LY0215", ("المنطقة", "الغربية")),
    ("LY0216", ("الجبل", "الغرب")), ("LY0209", ("نالوت",)), ("LY0319", ("سبها",)),
    ("LY0318", ("الشاطئ",)), ("LY0320", ("الحياة",)), ("LY0322", ("مرزق",)),
    ("LY0321", ("غات",)),
]

EQUAL_AREA = "EPSG:6933"


def _get(url, dst, magic, minsize):
    if os.path.exists(dst) and os.path.getsize(dst) > minsize:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("  GET", url)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        data = r.read()
    if not data.startswith(magic):
        raise SystemExit(f"{url} did not return the expected file; starts {data[:24]!r}")
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(COD_URL, COD_ZIP, b"PK", 100_000)
    _get(BSC_URL, BSC_PDF, b"%PDF", 10_000)
    _get(USCB_URL, USCB_XLSX, b"PK", 500_000)


def bsc_table():
    """The PDF's 22 rows as {pcode: people}, witnessed by name and closed on the printed total."""
    import fitz

    with open(BSC_PDF, "rb") as fh:
        if b"%%EOF" not in fh.read()[-2048:]:
            raise SystemExit(f"{BSC_PDF} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(BSC_PDF)
    if len(doc) != 1:
        raise SystemExit(f"the BSC PDF has {len(doc)} pages, expected 1")
    lines = [ln.strip() for ln in doc[0].get_text().splitlines() if ln.strip()]
    if not any("الليبي" in ln and "2020" in ln for ln in lines):
        raise SystemExit("the PDF's title no longer says it estimates Libyans for 2020")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("المناطق")) + 1
    rows = []
    for ln in lines[start:start + len(PDF_ORDER)]:
        m = re.fullmatch(r"(.*?)(\d{4,})", ln)
        if not m:
            raise SystemExit(f"PDF line {ln!r} is not a name and a count")
        rows.append((m.group(1), int(m.group(2))))
    out = {}
    for (pc, frags), (name, n) in zip(PDF_ORDER, rows):
        if not all(f in name for f in frags):
            raise SystemExit(f"PDF row {name!r} should be {pc} (fragments {frags})")
        out[pc] = n
    k = next(i for i, ln in enumerate(lines) if "المجموع" in ln)
    total = int(lines[k + 1])
    if total != TOTAL_2020 or sum(out.values()) != total:
        raise SystemExit(f"the PDF's rows sum to {sum(out.values()):,} and its total is "
                         f"{total:,}; pinned {TOTAL_2020:,}")
    return out


def uscb(sheet):
    t = pd.read_excel(USCB_XLSX, sheet_name=sheet, header=0).iloc[1:]   # row 1 is the alias row
    return t


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (COD_ZIP, BSC_PDF, USCB_XLSX):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run with --fetch")

    with zipfile.ZipFile(COD_ZIP) as zf:
        g = gpd.read_file(io.BytesIO(zf.read("lby_admin2.geojson")))
    if len(g) != 22 or g.crs is None or g.crs.to_epsg() != 4326:
        raise SystemExit(f"COD-AB ADM2: {len(g)} features, crs {g.crs}; expected 22 in EPSG:4326")
    print(f"read COD-AB ADM2: {len(g)} districts")

    # ---- witness 1: the PDF closes on itself, row by row named ----
    bsc = bsc_table()
    print(f"  BSC 2020: 22 rows, each line naming the district expected, summing to the printed "
          f"total {TOTAL_2020:,} (Libyans)")

    # ---- witness 3: names under each code, COD and USCB ----
    if sorted(g["adm2_pcode"]) != sorted(DISTRICTS):
        raise SystemExit("COD's p-codes are not DISTRICTS'")
    bad = [(pc, n) for pc, n in zip(g["adm2_pcode"], g["adm2_name"]) if DISTRICTS[pc][1] != n]
    pop = uscb("Population")
    pop1 = pop[pop["ADM_LEVEL"] == 1].set_index("NSO_CODE")
    bad += [(pc, pop1.loc[pc, "AREA_NAME"]) for pc in DISTRICTS
            if pop1.loc[pc, "AREA_NAME"] != DISTRICTS[pc][2]]
    if bad or sorted(pop1.index) != sorted(DISTRICTS):
        raise SystemExit(f"a name under a p-code is not the expected one: {bad}")
    print("  witness 3: COD's and USCB's names under each of the 22 codes are the expected ones")

    # ---- witness 2: USCB's transcription of the same estimate ----
    diff = {pc: int(pop1.loc[pc, "POP_TTL_20"]) - bsc[pc] for pc in DISTRICTS}
    diff = {pc: d for pc, d in diff.items() if d}
    if diff != USCB_DIFF:
        raise SystemExit(f"USCB's POP_TTL_20 minus the PDF: {diff}, pinned {USCB_DIFF}")
    print(f"  witness 2: USCB's 2020 column equals the PDF in 21 districts; Al Marj is "
          f"{USCB_DIFF['LY0102']:,} higher in USCB (pinned), its whole national excess")

    hh = uscb("Households")
    hh0 = hh[hh["ADM_LEVEL"] == 0].iloc[0]
    hh1 = hh[hh["ADM_LEVEL"] == 1].set_index("NSO_CODE")
    if int(hh0["HH_PBTN"]) != CENSUS_2006 or int(hh0["HH_PNLN"]) != NONLIBYAN_2006 \
            or int(pd.to_numeric(hh1["HH_PBTN"]).sum()) != CENSUS_2006:
        raise SystemExit("the 2006 census totals in the Households sheet are not the pinned ones")
    nat = uscb("Nationality")
    nat1 = nat[nat["ADM_LEVEL"] == 1].set_index("NSO_CODE")
    if int(pd.to_numeric(nat1["NAT_TOT_NL"]).sum()) != NONLIBYAN_2012:
        raise SystemExit("the 2012 survey's non-Libyans do not sum to the pinned total")
    print(f"  2006 census: {CENSUS_2006:,} people, {NONLIBYAN_2006:,} non-Libyans "
          f"({NONLIBYAN_2006 / CENSUS_2006:.2%}); 2012 survey: {NONLIBYAN_2012:,} non-Libyans")

    lut = pd.DataFrame([dict(geo_id=pc, unit=pc, name=m, cod_name=c, uscb_name=u, pop=bsc[pc],
                             pop_2006=int(hh1.loc[pc, "HH_PBTN"]),
                             nonlibyan_2006=int(hh1.loc[pc, "HH_PNLN"]),
                             nonlibyan_2012=int(nat1.loc[pc, "NAT_TOT_NL"]),
                             area_2006=float(hh1.loc[pc, "HH_LAREA"]))
                        for pc, (m, c, u) in DISTRICTS.items()])
    g = g.merge(lut, left_on="adm2_pcode", right_on="geo_id", how="inner", validate="1:1")
    g["cod_area"] = g.to_crs(EQUAL_AREA).geometry.area / 1e6

    # ---- 4: COD's area against the 2006 census's land area, printed ----
    g["area_ratio"] = g["cod_area"] / g["area_2006"]
    print("  COD area over the 2006 census's land area:")
    for _i, r in g.sort_values("area_ratio").iterrows():
        print(f"      {r['name']:<20} {r['cod_area']:>9,.0f} km2 / {r['area_2006']:>9,.0f}  "
              f"{r['area_ratio']:.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    g[["unit", "name", "geo_id", "pop", "geometry"]].to_file(OUT, layer="districts", driver="GPKG")
    print(f"\nwrote {OUT} (22 polygons)")
    lk = g[["geo_id", "unit", "name", "pop", "pop_2006", "nonlibyan_2006", "nonlibyan_2012",
            "cod_area", "area_2006"]].sort_values("geo_id")
    lk.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"wrote {LOOKUP} (22 rows, {int(lk['pop'].sum()):,} Libyans)")


if __name__ == "__main__":
    main()
