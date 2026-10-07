"""Bhutan: boundaries and the 2017 census count of Bhutanese and non-Bhutanese in each dzongkhag.

Writes data/geo/bt/bt_dzongkhags.gpkg and data/geo/bt/bt_lookup.csv.

Three sources:

  * **boundaries**: COD-AB `cod-ab-btn` version 01 (OCHA ROAP; HDX, reviewed 2025-10-30, boundaries
    created 2020-01-01), `btn_admin1.geojson` (20 dzongkhags) and `btn_admin2.geojson` (205 gewogs).
  * **populations**: National Statistics Bureau, *2017 Population and Housing Census of Bhutan,
    National Report* (NSB download 5006, read from the Wayback copy of 2022-07-07; 286 pages with a
    text layer):
      - Table 2.6 (pdf p.37), Bhutanese population by dzongkhag, 681,720;
      - Table 2.8 (pdf p.39), non-Bhutanese population by dzongkhag, 45,425, which leaves out the
        8,408 tourists and others found in hotels and the 16,057 day workers who sleep in India;
      - Table A2.8 (pdf p.120), each dzongkhag's area and its 2017 population, 727,145, which is
        the two tables above added together.
  * Nothing else. Nationality is not tabulated by dzongkhag; `sources/bt.py` takes it from UN DESA.

## THE KEY IS THE CENSUS'S DZONGKHAG NAME, PINNED TO COD'S PCODE

The census prints no codes. `DZONGKHAGS` pins each pcode to COD's `adm1_name`, the census's spelling
and the spelling of the Centre for Bhutan & GNH Studies' 2015 report, which `sources/bt.py` reads.

## TABLE A2.8 SWAPS LHUENTSE'S AND MONGGAR'S AREAS

A2.8 prints Lhuentse at 1,944 km2 and Monggar at 2,859; COD-AB draws them the other way round (2,858
and 1,944), with Lhuentse the northern one on the Tibetan border, as every map has it. The swap is the
census table's, so the area witness compares the two the other way round and says so
(`AREA_SWAPPED`). Read that way, 17 of the 20 agree within 1%. The other three are high in the
table, not low in COD: Trashigang 3,066 km2 against COD's 2,202, Sarpang 1,946 against 1,655, Thimphu
2,067 against 1,796, and the table's 20 areas add to 40,454 km2 where its own national row (and the
report's text, "38,394 km2 (Atlas of Bhutan)") says 38,394; COD's add to 38,763. Which people sit in
which polygon is tested in `sources/bt_grid.py` against Kontur, the check that matters for dots; this
one only prints.

## CHECKS

  1. every dzongkhag's Bhutanese plus non-Bhutanese equals its 2017 total in A2.8, and the three
     tables sum to their printed national rows;
  2. COD-AB has 20 and 205 features, the name under every pcode is the pinned one, every gewog's
     parent is one of the 20, and the gewogs dissolved match the dzongkhags in area;
  3. Lhuentse's polygon lies north of Monggar's (the swap above is the census table's, not COD's);
  4. the area witness is printed per dzongkhag.

Usage:
    python sources/bt_geo.py --fetch    COD-AB geojson zip (3.2 MB), the census report (23.8 MB)
    python sources/bt_geo.py            rebuild from data/raw/bt/
"""

import io
import os
import re
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "6")

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bt")
OUT_DIR = os.path.join(ROOT, "data", "geo", "bt")
OUT_UNITS = os.path.join(OUT_DIR, "bt_dzongkhags.gpkg")
LOOKUP = os.path.join(OUT_DIR, "bt_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

COD_URL = ("https://data.humdata.org/dataset/cff13277-eac9-49b0-8436-e80f42eb4684/resource/"
           "c82bb508-86e6-4b61-bbe4-93bdf3d24faa/download/btn_admin_boundaries.geojson.zip")
COD_ZIP = os.path.join(RAW, "btn_admin_boundaries.geojson.zip")
# nsb.gov.bt/download/5006/ answers 404 with an HTML page since 2024; the Wayback copy is the file.
PHCB_URL = "http://web.archive.org/web/20220707154821id_/https://www.nsb.gov.bt/download/5006/"
PHCB = os.path.join(RAW, "phcb_2017_national_report.pdf")
PHCB_PAGES = 286
T26_PAGE, T28_PAGE, A28_PAGE = 37, 39, 120        # 1-based

BHUTANESE = 681_720
NON_BHUTANESE = 45_425
TOTAL_2017 = 727_145
EXPECTED_ADM1, EXPECTED_ADM2 = 20, 205
EQUAL_AREA = "EPSG:6933"
AREA_TOL = 0.10
AREA_SWAPPED = {"BT006": "BT007"}       # Lhuentse <-> Monggar in Table A2.8

# pcode -> (the name this map prints, COD adm1_name, the census's spelling, GNH 2015's spelling)
DZONGKHAGS = {
    "BT001": ("Bumthang", "Bumthang", "Bumthang", "Bumthang"),
    "BT002": ("Chhukha", "Chhukha", "Chhukha", "Chukha"),
    "BT003": ("Dagana", "Dagana", "Dagana", "Dagana"),
    "BT004": ("Gasa", "Gasa", "Gasa", "Gasa"),
    "BT005": ("Haa", "Haa", "Haa", "Haa"),
    "BT006": ("Lhuentse", "Lhuentse", "Lhuentse", "Lhuntse"),
    "BT007": ("Monggar", "Monggar", "Monggar", "Mongar"),
    "BT008": ("Paro", "Paro", "Paro", "Paro"),
    "BT009": ("Pema Gatshel", "Pemagatshel", "Pema Gatshel", "Pema Gatshel"),
    "BT010": ("Punakha", "Punakha", "Punakha", "Punakha"),
    "BT011": ("Samdrup Jongkhar", "Samdrupjongkhar", "Samdrup Jongkhar", "Samdrup Jongkhar"),
    "BT012": ("Samtse", "Samtse", "Samtse", "Samtse"),
    "BT013": ("Sarpang", "Sarpang", "Sarpang", "Sarpang"),
    "BT014": ("Thimphu", "Thimphu", "Thimphu", "Thimphu"),
    "BT015": ("Trashigang", "Trashigang", "Trashigang", "Tashigang"),
    "BT016": ("Trashi Yangtse", "Yangtse", "Trashi Yangtse", "Tashi Yangtse"),
    "BT017": ("Trongsa", "Trongsa", "Trongsa", "Trongsa"),
    "BT018": ("Tsirang", "Tsirang", "Tsirang", "Tsirang"),
    "BT019": ("Wangdue Phodrang", "Wangduephodrang", "Wangdue Phodrang", "Wangdue Phodrang"),
    "BT020": ("Zhemgang", "Zhemgang", "Zhemgang", "Zhemgang"),
}

LEADING_NUMBER = re.compile(r"^\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+(?:\.[0-9]+)?)(?![0-9,])")


def _get(url, dst, magic, minsize):
    if os.path.exists(dst) and os.path.getsize(dst) > minsize:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("  GET", url)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r:
        data = r.read()
    if not data.startswith(magic):
        raise SystemExit(f"{url} did not return the expected file; starts {data[:24]!r}")
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(COD_URL, COD_ZIP, b"PK", 2_000_000)
    _get(PHCB_URL, PHCB, b"%PDF", 20_000_000)


def page_lines(doc, page):
    return [ln.strip() for ln in doc[page - 1].get_text().splitlines() if ln.strip()]


def rows_by_name(lines, width, label):
    """{pcode: [width numbers]} read after each dzongkhag's census name, which must occur once."""
    out = {}
    for pc, (_n, _c, census, _g) in DZONGKHAGS.items():
        hits = [i for i, ln in enumerate(lines) if ln == census]
        if len(hits) != 1:
            raise SystemExit(f"{label}: census name {census!r} occurs {len(hits)} times")
        vals, j = [], hits[0] + 1
        while len(vals) < width:
            if j >= len(lines):
                raise SystemExit(f"{label}: {census} ran off the table with {vals}")
            m = LEADING_NUMBER.match(lines[j])
            if m:
                vals.append(float(m.group(1).replace(",", "")))
            elif re.search(r"[A-Za-z]", lines[j]):
                raise SystemExit(f"{label}: {census} reached {lines[j]!r} after {len(vals)} numbers")
            j += 1
        out[pc] = vals
    return out


def national_row(lines, width, label):
    hits = [i for i, ln in enumerate(lines) if ln == "Bhutan"]
    if not hits:
        raise SystemExit(f"{label}: no national row")
    vals, j = [], hits[-1] + 1
    while len(vals) < width:
        m = LEADING_NUMBER.match(lines[j])
        if m:
            vals.append(float(m.group(1).replace(",", "")))
        j += 1
    return vals


def read_census():
    import fitz

    with open(PHCB, "rb") as fh:
        if b"%%EOF" not in fh.read()[-4096:]:
            raise SystemExit(f"{PHCB} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(PHCB)
    if len(doc) != PHCB_PAGES:
        raise SystemExit(f"the census report has {len(doc)} pages, expected {PHCB_PAGES}")
    l26, l28, la28 = (page_lines(doc, p) for p in (T26_PAGE, T28_PAGE, A28_PAGE))
    for lines, title in ((l26, "Table 2.6 Distribution of Bhutanese Populations"),
                         (l28, "Table 2.8 Distribution of Non Bhutanese Population"),
                         (la28, "Table A2.8 Population Density by Dzongkhag")):
        if not any(" ".join(ln.split()).startswith(title) for ln in lines):
            raise SystemExit(f"no {title!r} on its page")
    t26 = rows_by_name(l26, 3, "Table 2.6")
    t28 = rows_by_name(l28, 3, "Table 2.8")
    a28 = rows_by_name(la28, 3, "Table A2.8")
    bad = [pc for pc, v in list(t26.items()) + list(t28.items()) if v[0] + v[1] != v[2]]
    if bad:
        raise SystemExit(f"rows where male + female is not both: {bad}")
    n26, n28, na28 = (national_row(x, 3, lab) for x, lab in
                      ((l26, "2.6"), (l28, "2.8"), (la28, "A2.8")))
    s26 = sum(v[2] for v in t26.values())
    s28 = sum(v[2] for v in t28.values())
    sa = sum(v[2] for v in a28.values())
    if (s26, n26[2]) != (BHUTANESE, BHUTANESE) or (s28, n28[2]) != (NON_BHUTANESE, NON_BHUTANESE) \
            or (sa, na28[2]) != (TOTAL_2017, TOTAL_2017):
        raise SystemExit(f"census sums: Bhutanese {s26:,.0f}/{n26[2]:,.0f}, non-Bhutanese "
                         f"{s28:,.0f}/{n28[2]:,.0f}, all {sa:,.0f}/{na28[2]:,.0f}")
    bad = {pc: (t26[pc][2], t28[pc][2], a28[pc][2]) for pc in DZONGKHAGS
           if t26[pc][2] + t28[pc][2] != a28[pc][2]}
    if bad:
        raise SystemExit(f"Bhutanese + non-Bhutanese is not A2.8's 2017 total: {bad}")
    print(f"PHCB 2017: 20 dzongkhags, {BHUTANESE:,} Bhutanese + {NON_BHUTANESE:,} non-Bhutanese = "
          f"{TOTAL_2017:,}; every dzongkhag's two counts add to its A2.8 total, and every row adds by sex")
    return ({pc: int(v[2]) for pc, v in t26.items()}, {pc: int(v[2]) for pc, v in t28.items()},
            {pc: v[0] for pc, v in a28.items()})


def main():
    if "--fetch" in sys.argv:
        fetch()
    for p in (COD_ZIP, PHCB):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run with --fetch")
    bhut, nonb, census_area = read_census()

    with zipfile.ZipFile(COD_ZIP) as zf:
        a1 = gpd.read_file(io.BytesIO(zf.read("btn_admin1.geojson")))
        a2 = gpd.read_file(io.BytesIO(zf.read("btn_admin2.geojson")))
    if (len(a1), len(a2)) != (EXPECTED_ADM1, EXPECTED_ADM2) or a1.crs.to_epsg() != 4326:
        raise SystemExit(f"COD-AB: {len(a1)} ADM1 and {len(a2)} ADM2 features, expected "
                         f"{EXPECTED_ADM1} and {EXPECTED_ADM2}")
    got = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    bad = [pc for pc, (_n, cod, _c, _g) in DZONGKHAGS.items() if got.get(pc) != cod]
    if bad or len(got) != len(DZONGKHAGS):
        raise SystemExit(f"COD names under these pcodes are not the pinned ones: {bad}")
    orphans = sorted(set(a2["adm1_pcode"]) - set(DZONGKHAGS))
    if orphans:
        raise SystemExit(f"COD gewogs with a parent that is not a dzongkhag: {orphans}")
    a1 = a1.set_index("adm1_pcode")
    area = a1.to_crs(EQUAL_AREA).geometry.area / 1e6
    a2u = a2.dissolve("adm1_pcode").to_crs(EQUAL_AREA).geometry.area / 1e6
    worst = max(abs(a2u[pc] / area[pc] - 1) for pc in DZONGKHAGS)
    print(f"  COD-AB: 20 dzongkhags and 205 gewogs, every name under its pcode as pinned; gewogs "
          f"dissolved against dzongkhags, largest area difference {worst:.2%}")
    if worst > 0.01:
        raise SystemExit("the gewog layer does not tile the dzongkhags")

    lat = a1.to_crs(EQUAL_AREA).geometry.centroid.to_crs(4326).y
    if not lat["BT006"] > lat["BT007"]:
        raise SystemExit("COD's Lhuentse is not north of Monggar; the area swap may be COD's")
    print(f"  Lhuentse's polygon is north of Monggar's ({lat['BT006']:.2f} against "
          f"{lat['BT007']:.2f} N), so Table A2.8's swapped areas are the table's")
    swapped = dict(AREA_SWAPPED)
    swapped.update({v: k for k, v in AREA_SWAPPED.items()})
    print("  area witness, COD-AB km2 against Table A2.8 (Lhuentse and Monggar read crosswise):")
    for pc in DZONGKHAGS:
        ref = census_area[swapped.get(pc, pc)]
        r = area[pc] / ref
        flag = "" if abs(r - 1) <= AREA_TOL else "   <- outside the band"
        print(f"      {pc} {DZONGKHAGS[pc][0]:<17} COD {area[pc]:>7,.0f}  census {ref:>7,.0f}  "
              f"{r:.2f}{flag}")
    print(f"  COD {area.sum():,.0f} km2 in all, census {sum(census_area.values()):,.0f}")

    u = sorted(DZONGKHAGS)
    lut = pd.DataFrame([dict(geo_id=pc, unit=pc, name=DZONGKHAGS[pc][0],
                             census_name=DZONGKHAGS[pc][2], gnh_name=DZONGKHAGS[pc][3],
                             bhutanese=bhut[pc], non_bhutanese=nonb[pc],
                             pop=bhut[pc] + nonb[pc], cod_area=round(float(area[pc]), 1),
                             census_area=census_area[swapped.get(pc, pc)]) for pc in u])
    os.makedirs(OUT_DIR, exist_ok=True)
    g1 = a1.reset_index()[["adm1_pcode", "geometry"]].rename(columns={"adm1_pcode": "unit"})
    g1 = g1.merge(lut[["unit", "name", "pop"]], on="unit", how="inner", validate="1:1")
    g1[["unit", "name", "pop", "geometry"]].to_file(OUT_UNITS, layer="dzongkhags", driver="GPKG")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_UNITS} (20) and {LOOKUP} ({int(lut['pop'].sum()):,} people, "
          f"{int(lut['bhutanese'].sum()):,} of them Bhutanese)")


if __name__ == "__main__":
    main()
