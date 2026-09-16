"""Afghanistan: boundaries and NSIA's 2025-26 population estimates for the 34 provinces.

Writes data/geo/af/af_provinces.gpkg and data/geo/af/af_lookup.csv.

Nobody in Afghanistan is asked their religion (`sources/af.md` §1), so this file is the whole of the
country's data: where people are. `sources/af.py` puts every one of them on Islam.

Three sources:

  * **boundaries**: COD-AB `cod-ab-afg` version 03 (source AGCHO and NSIA; HDX, last edited
    2026-03-25, boundaries created 2019-05-01), `afg_admin1.geojson` (34 provinces) and
    `afg_admin2.geojson` (401 districts). The de facto authorities have designated 457 district-level
    units; COD keeps AGCHO's 401 for lack of boundaries (the dataset's own caveat).
  * **populations**: NSIA, *Estimated Population of Afghanistan 2025-26* (1404, September 2025), 168
    pages with a text layer. NSIA's introduction (pp. III-IV) says the figures are projected with
    Pt = P0 e^rt from the 2002-2005 (1381-1384) household listing, base year 1383, because no census
    has been taken since 1979 (1358); the nomadic Kuchi population is held at a fixed 1.5 million
    for the whole country. Read here:
      - Table 3, *Population of Afghanistan by Zone, Sex and Residence* (pdf pp. 29-30): each
        province's rural, urban and total, each by female, male and both;
      - the table after it (pdf pp. 31-32): each province's total for 1402, 1403 and 1404.
  * **witness**: COD-PS `cod-ps-afg`, `afg_admpop_adm1_2026.csv` (OCHA; "2021 estimates based on 2017
    study conducted by Flowminder/UNFPA"), a modelled population of a different lineage, on COD's
    pcodes. It is compared, never drawn.

## THE KEY IS NSIA'S ENGLISH PROVINCE NAME, PINNED TO COD'S PCODE

NSIA prints no codes. `PROVINCES` pins each pcode to COD's `adm1_name` and to NSIA's spelling
(`Maydanwardag`, `Urozgan`, `Helmand`, `Herat`), and every name must occur exactly once in each of the
two NSIA tables. The join's witness is COD-PS's ranking of the provinces, which neither name decides.

## CHECKS

  1. Table 3: female + male = both, and rural + urban = total, in every province; the 34 rows sum to
     the settled national row, `SETTLED_1404`;
  2. the three-year table's 1404 column equals Table 3's total in every province, and its 1402 and
     1403 columns sum to their printed national rows;
  3. COD-AB has 34 and 401 features, the name under every pcode is the pinned one, and every
     district's parent is one of the 34; the districts dissolved match the provinces in area;
  4. **rank witness**: Spearman of NSIA against COD-PS over the 34 provinces, against `N_PERM`
     shuffles; the per-province ratio is printed.

Usage:
    python sources/af_geo.py --fetch    COD-AB geojson zip (14 MB), NSIA 1404 (18.5 MB), COD-PS (11 KB)
    python sources/af_geo.py            rebuild from data/raw/af/
"""

import io
import os
import re
import ssl
import sys
import urllib.request
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("OMP_NUM_THREADS", "6")

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "af")
OUT_DIR = os.path.join(ROOT, "data", "geo", "af")
OUT_PROVINCES = os.path.join(OUT_DIR, "af_provinces.gpkg")
LOOKUP = os.path.join(OUT_DIR, "af_lookup.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}

COD_URL = ("https://data.humdata.org/dataset/4c303d7b-8eae-4a5a-a3aa-b2331fa39d74/resource/"
           "330aad34-2254-4622-afac-e98ace1524ae/download/afg_admin_boundaries.geojson.zip")
COD_ZIP = os.path.join(RAW, "afg_admin_boundaries.geojson.zip")
CODPS_URL = ("https://data.humdata.org/dataset/79484094-6e4a-4fbd-9152-66b42a9b32e0/resource/"
             "90ba48e4-648f-4efc-97df-54fe05132e5e/download/afg_admpop_adm1_2026.csv")
CODPS = os.path.join(RAW, "afg_admpop_adm1_2026.csv")
# nsia.gov.af:8443 serves an incomplete certificate chain, so the fetch does not verify it; the file
# is checked by its %PDF magic, its %%EOF trailer, its page count and every total below.
NSIA_URL = ("https://nsia.gov.af:8443/wp-content/uploads/2025/09/"
            "%D8%A8%D8%B1%D8%A7%D9%88%D8%B1%D8%AF-%D9%86%D9%81%D9%88%D8%B3-"
            "%DA%A9%D8%B4%D9%88%D8%B1-%D8%B3%D8%A7%D9%84-1404.pdf")
NSIA = os.path.join(RAW, "nsia_estimated_population_1404.pdf")
NSIA_PAGES = 168
TABLE3_PAGES = (29, 30)          # 1-based
YEARS_PAGES = (31, 32)

SETTLED_1404 = 34_935_197
KUCHI = 1_500_000
TOTAL_1404 = 36_435_197
SETTLED_1402 = 33_471_517
SETTLED_1403 = 34_195_527
EXPECTED_ADM1, EXPECTED_ADM2 = 34, 401
N_PERM = 20_000
EQUAL_AREA = "EPSG:6933"

# pcode -> (the name this map prints, COD adm1_name, NSIA's English spelling)
PROVINCES = {
    "AF01": ("Kabul", "Kabul", "Kabul"),
    "AF02": ("Kapisa", "Kapisa", "Kapisa"),
    "AF03": ("Parwan", "Parwan", "Parwan"),
    "AF04": ("Maidan Wardak", "Maidan Wardak", "Maydanwardag"),
    "AF05": ("Logar", "Logar", "Logar"),
    "AF06": ("Nangarhar", "Nangarhar", "Nangarhar"),
    "AF07": ("Laghman", "Laghman", "Laghman"),
    "AF08": ("Panjshir", "Panjsher", "Panjsher"),
    "AF09": ("Baghlan", "Baghlan", "Baghlan"),
    "AF10": ("Bamyan", "Bamyan", "Bamyan"),
    "AF11": ("Ghazni", "Ghazni", "Ghazni"),
    "AF12": ("Paktika", "Paktika", "Paktika"),
    "AF13": ("Paktia", "Paktya", "Paktya"),
    "AF14": ("Khost", "Khost", "Khost"),
    "AF15": ("Kunar", "Kunar", "Kunar"),
    "AF16": ("Nuristan", "Nuristan", "Nuristan"),
    "AF17": ("Badakhshan", "Badakhshan", "Badakhshan"),
    "AF18": ("Takhar", "Takhar", "Takhar"),
    "AF19": ("Kunduz", "Kunduz", "Kunduz"),
    "AF20": ("Samangan", "Samangan", "Samangan"),
    "AF21": ("Balkh", "Balkh", "Balkh"),
    "AF22": ("Sar-e Pol", "Sar-e-Pul", "Sar-e-Pul"),
    "AF23": ("Ghor", "Ghor", "Ghor"),
    "AF24": ("Daykundi", "Daykundi", "Daykundi"),
    "AF25": ("Uruzgan", "Uruzgan", "Urozgan"),
    "AF26": ("Zabul", "Zabul", "Zabul"),
    "AF27": ("Kandahar", "Kandahar", "Kandahar"),
    "AF28": ("Jowzjan", "Jawzjan", "Jawzjan"),
    "AF29": ("Faryab", "Faryab", "Faryab"),
    "AF30": ("Helmand", "Hilmand", "Helmand"),
    "AF31": ("Badghis", "Badghis", "Badghis"),
    "AF32": ("Herat", "Hirat", "Herat"),
    "AF33": ("Farah", "Farah", "Farah"),
    "AF34": ("Nimroz", "Nimroz", "Nimroz"),
}

LEADING_NUMBER = re.compile(r"^\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?![0-9,])")


def _get(url, dst, magic, minsize, verify=True):
    if os.path.exists(dst) and os.path.getsize(dst) > minsize:
        print(f"  have {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        return
    print("  GET", url)
    ctx = None if verify else ssl._create_unverified_context()
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900,
                                context=ctx) as r:
        data = r.read()
    if not data.startswith(magic):
        raise SystemExit(f"{url} did not return the expected file; starts {data[:24]!r}")
    with open(dst + ".part", "wb") as f:
        f.write(data)
    os.replace(dst + ".part", dst)
    print(f"  got  {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _get(COD_URL, COD_ZIP, b"PK", 10_000_000)
    _get(CODPS_URL, CODPS, b"admin0_name", 5_000)
    _get(NSIA_URL, NSIA, b"%PDF", 10_000_000, verify=False)


def page_lines(doc, pages):
    out = []
    for p in pages:
        out += [ln.strip() for ln in doc[p - 1].get_text().splitlines() if ln.strip()]
    return out


def rows_by_name(lines, width, label):
    """{pcode: [width numbers]} read after each province's NSIA name, which must occur once."""
    out = {}
    for pc, (_n, _c, nsia) in PROVINCES.items():
        hits = [i for i, ln in enumerate(lines) if ln == nsia]
        if len(hits) != 1:
            raise SystemExit(f"{label}: NSIA name {nsia!r} occurs {len(hits)} times")
        vals, j = [], hits[0] + 1
        while len(vals) < width:
            if j >= len(lines):
                raise SystemExit(f"{label}: {nsia} ran off the table with {vals}")
            m = LEADING_NUMBER.match(lines[j])
            if m:
                vals.append(int(m.group(1).replace(",", "")))
            elif re.search(r"[A-Za-z]", lines[j]):
                raise SystemExit(f"{label}: {nsia} reached {lines[j]!r} after {len(vals)} numbers")
            j += 1
        out[pc] = vals
    return out


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    for p in (COD_ZIP, CODPS, NSIA):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; run with --fetch")
    with open(NSIA, "rb") as fh:
        if b"%%EOF" not in fh.read()[-2048:]:
            raise SystemExit(f"{NSIA} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(NSIA)
    if len(doc) != NSIA_PAGES:
        raise SystemExit(f"NSIA 1404 has {len(doc)} pages, expected {NSIA_PAGES}")

    # ---- 1: Table 3 ----
    t3_lines = page_lines(doc, TABLE3_PAGES)
    if not any("Table 3: Population of Afghanistan by Zone" in ln for ln in t3_lines):
        raise SystemExit(f"pdf pages {TABLE3_PAGES} do not hold Table 3")
    t3 = rows_by_name(t3_lines, 9, "Table 3")
    bad = []
    for pc, v in t3.items():
        rf, rm, rb, uf, um, ub, tf, tm, tb = v
        if rf + rm != rb or uf + um != ub or tf + tm != tb or rb + ub != tb:
            bad.append((pc, v))
    if bad:
        raise SystemExit(f"Table 3 rows that do not add up: {bad}")
    settled = sum(v[8] for v in t3.values())
    if settled != SETTLED_1404:
        raise SystemExit(f"Table 3's provinces sum to {settled:,}, printed {SETTLED_1404:,}")
    print(f"NSIA 1404 Table 3: 34 provinces, {settled:,} settled people "
          f"({sum(v[2] for v in t3.values()):,} rural, {sum(v[5] for v in t3.values()):,} urban); "
          f"every row adds up by sex and by residence")

    # ---- 2: the three-year table ----
    yr_lines = page_lines(doc, YEARS_PAGES)
    if not any("1402/ 2023-24" in ln for ln in yr_lines):
        raise SystemExit(f"pdf pages {YEARS_PAGES} do not hold the 1402-1404 table")
    yrs = rows_by_name(yr_lines, 9, "1402-1404 table")
    bad = {pc: (yrs[pc][8], t3[pc][8]) for pc in PROVINCES if yrs[pc][8] != t3[pc][8]}
    if bad:
        raise SystemExit(f"the 1402-1404 table's 1404 totals differ from Table 3's: {bad}")
    s1402 = sum(v[2] for v in yrs.values())
    s1403 = sum(v[5] for v in yrs.values())
    if (s1402, s1403) != (SETTLED_1402, SETTLED_1403):
        raise SystemExit(f"1402 and 1403 provinces sum to {s1402:,} and {s1403:,}")
    print(f"  the 1402-1404 table agrees with Table 3 in every province; 1402 {s1402:,}, "
          f"1403 {s1403:,}; plus {KUCHI:,} Kuchi each year, national only")

    # ---- 3: COD-AB ----
    with zipfile.ZipFile(COD_ZIP) as zf:
        a1 = gpd.read_file(io.BytesIO(zf.read("afg_admin1.geojson")))
        a2 = gpd.read_file(io.BytesIO(zf.read("afg_admin2.geojson")))
    if (len(a1), len(a2)) != (EXPECTED_ADM1, EXPECTED_ADM2) or a1.crs.to_epsg() != 4326:
        raise SystemExit(f"COD-AB: {len(a1)} ADM1 and {len(a2)} ADM2 features, expected "
                         f"{EXPECTED_ADM1} and {EXPECTED_ADM2}")
    got = dict(zip(a1["adm1_pcode"], a1["adm1_name"]))
    bad = [pc for pc, (_n, cod, _s) in PROVINCES.items() if got.get(pc) != cod]
    if bad or len(got) != len(PROVINCES):
        raise SystemExit(f"COD names under these pcodes are not the pinned ones: {bad}")
    orphans = sorted(set(a2["adm1_pcode"]) - set(PROVINCES))
    if orphans:
        raise SystemExit(f"COD districts with a parent that is not a province: {orphans}")
    a1 = a1.set_index("adm1_pcode")
    area = a1.to_crs(EQUAL_AREA).geometry.area / 1e6
    a2u = a2.dissolve("adm1_pcode").to_crs(EQUAL_AREA).geometry.area / 1e6
    worst = max(abs(a2u[pc] / area[pc] - 1) for pc in PROVINCES)
    print(f"  COD-AB: 34 provinces and 401 districts, every name under its pcode as pinned; districts "
          f"dissolved against provinces, largest area difference {worst:.2%}")
    if worst > 0.01:
        raise SystemExit("the district layer does not tile the provinces")

    # ---- 4: the rank witness, COD-PS ----
    ps = pd.read_csv(CODPS, skiprows=[1])
    ps = ps.set_index("province_code")["population_total"].astype(float)
    if sorted(ps.index) != sorted(PROVINCES):
        raise SystemExit(f"COD-PS pcodes differ: {sorted(set(ps.index) ^ set(PROVINCES))}")
    u = sorted(PROVINCES)
    a = np.array([t3[pc][8] for pc in u], dtype=float)
    b = np.array([ps[pc] for pc in u])
    rho = stats.spearmanr(a, b).statistic
    rng = np.random.default_rng(0)
    perm = np.array([stats.spearmanr(a, rng.permutation(b)).statistic for _ in range(N_PERM)])
    beaten = int((perm >= rho).sum())
    nat = b.sum() / a.sum()
    print(f"  COD-PS 2026 {b.sum():,.0f} against NSIA's settled {a.sum():,.0f} (ratio {nat:.3f}); "
          f"Spearman over 34 provinces {rho:+.3f}, {beaten} of {N_PERM:,} shuffles reach it "
          f"(best {perm.max():+.3f})")
    if beaten:
        raise SystemExit("COD-PS does not rank the provinces like NSIA; the name join may be permuted")
    rel = sorted(((ps[pc] / t3[pc][8]) / nat, pc) for pc in u)
    print("  COD-PS / NSIA per province, over the national ratio: "
          + ", ".join(f"{PROVINCES[pc][0]} {r:.2f}" for r, pc in rel))

    lut = pd.DataFrame([dict(geo_id=pc, unit=pc, name=PROVINCES[pc][0], nsia_name=PROVINCES[pc][2],
                             pop=t3[pc][8], rural=t3[pc][2], urban=t3[pc][5],
                             pop_1402=yrs[pc][2], pop_1403=yrs[pc][5],
                             codps_2026=round(float(ps[pc])),
                             cod_area=round(float(area[pc]), 1)) for pc in u])
    os.makedirs(OUT_DIR, exist_ok=True)
    g1 = a1.reset_index()[["adm1_pcode", "geometry"]].rename(columns={"adm1_pcode": "unit"})
    g1 = g1.merge(lut[["unit", "name", "pop"]], on="unit", how="inner", validate="1:1")
    g1[["unit", "name", "pop", "geometry"]].to_file(OUT_PROVINCES, layer="provinces", driver="GPKG")
    lut.to_csv(LOOKUP, index=False, encoding="utf-8")
    print(f"\nwrote {OUT_PROVINCES} (34) and {LOOKUP} ({int(lut['pop'].sum()):,} settled people; "
          f"{KUCHI:,} Kuchi not in any province, {TOTAL_1404:,} in all)")


if __name__ == "__main__":
    main()
