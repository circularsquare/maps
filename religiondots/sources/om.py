"""Oman: nobody is asked their religion. Omanis are drawn on Islam, and expatriates by nationality
and sex in each governorate, on NCSI's register at the end of December 2024.

Reads, all fetched into data/raw/om/ by --fetch:

  * NCSI, *Statistical Year Book 2025* (Issue 53, 306 pages), now only whole on the Wayback
    Machine: Table 7-2 (Omanis and expatriates by governorate and wilaya, 2022-2024), Table 8-2
    (the register by nationality and sex), Table 8-4 (workers by nationality, sex and governorate,
    2024), Tables 17-4 and 18-4 (workers in the government and in the private, family, communal and
    other sectors, by nationality and sex, 2024);
  * GLMM's copies (`gulfmigration.grc.net`) of NCSI's *Population Statistics* mid-2018 table of the
    population by country of citizenship and sex (eleven nationalities), NCSI's Monthly Statistical
    Bulletin series of employed foreign workers by nationality (2013-2022), and NCSI's table of
    Omanis and non-Omanis by governorate and wilaya (2021-2023), a witness for Table 7-2;
  * Pew Research Center, *Religious Composition 2010-2020* (data/raw/estimates/pew.zip).

Writes data/normalized/om.csv (Omanis) and data/normalized/om_foreign.csv (expatriates), one row
per register wilaya. `sources/om.md` is the record in prose.

## NOBODY IS ASKED

The 2003 census form has no religion item and the 2003, 2010 and 2020 results print no religion
(sources.md §11ao); the 2020 census and everything since are read off registers that hold none.
Oman is not in the Arab Barometer. Built on Anita's Maghreb and Mauritania rulings (ask/RULINGS.md
2026-09-15 and 2026-09-16) and her priority line of 2026-09-15, as Saudi Arabia (sources/sa.md).

## OMANIS ARE ALL DRAWN ON ISLAM

Pew's Oman estimate is for everyone living there, so drawing its shares beside an expatriate layer
counts foreign non-Muslims twice. Every Omani is drawn on `islam`: no source counts an Omani who is
not Muslim. Ibadi, Sunni and Shia are not split: the only figures are national (Peterson 2004: about
45% Ibadi, 50% Sunni, under 5% Shia among Omanis; other estimates run from 21% to 75% Ibadi), and
nothing below the nation has a number. Filed as an ask (sources/om.md §6).

## EXPATRIATES: GOVERNORATE, SEX, NATIONALITY

The register publishes expatriates per wilaya, and expatriate WORKERS per governorate by sex
(Table 8-4, 1,808,926 of the 2,283,279). It publishes workers' nationality only for the whole
country, by sex (Tables 17-4 and 18-4), and nothing on the nationality of the 474,353 expatriates
who are not workers (dependants, from here on). So each governorate's expatriates are three parts:

  * **male workers** at the national mix of male workers;
  * **female workers** at the national mix of female workers (the sexes come from different
    places: 32,696 women and 363 men from Myanmar, 25,791 women and 609,996 men from Bangladesh);
  * **dependants** (the governorate's expatriates less its workers) at the mix of mid-2018's
    expatriates less 2018's workers, by nationality (`dependants_2018`), the only year both are
    published for the same nationalities.

Each wilaya's expatriates take their governorate's composition.

The tables' residual rows:
  * **Other nationalities, women** (59,371): the four nationalities in NCSI's mid-2018 table that
    the 2024 tables no longer name (Uganda, Indonesia, Ethiopia, Nepal), at their 2018 female
    counts, which come to 55,832, 94% of the cell.
  * **Other nationalities, men** (94,516): the mix of the named male workers. The 2018 table's four
    hold 12,290 men, and the 2022 worker series names Nepal and Yemen (32,040 of both sexes), so
    nothing named reaches half the cell.
  * **Other Arabs** (government sector, 3,008): Pew's All Middle East-North Africa row.
  * **Dependants of other nationalities**: the mix of the named dependants.

Religion by nationality: Pew 2020 through `taxonomy/origin_religion.py`, Muslim branches folded to
`islam`, Pew's unplaced `Other religions` on `other.om`. **India's Hindu share comes from Pew's own
Oman estimate**, as for Saudi Arabia: Pew's *Faith on the Move* (2012, pp.21-22) says migrants from
India to Muslim-majority Middle Eastern countries are mostly Muslim, which Pew's India row does not
know; so India's Hindu share is set so the layer's Hindus equal Pew 2020's Oman Hindu share of the
register count, every other nationality kept at its own row, and the Hindus removed are drawn on
Islam. Myanmar's 33,059 workers (98.9% women) stay at Pew's Myanmar row: nothing found says who they
are (origin_religion rule 2).

`christian_witness`: the layer's Christians against Pew 2020's Oman share inside `CHRISTIAN_BAND`.
The band was written after a rough sum over the largest nationalities while scouting gave about
0.3, and the State Department's 2023 report puts Hindus, Buddhists and Christians together at 5% of
everyone, below Pew's Christians alone (8.1%); so the band only catches an error of several times.
sources/om.md §4 says so.

Usage:
    python sources/om.py --fetch    the yearbook (16.7 MB, Wayback) and three GLMM pages
    python sources/om.py            rebuild data/normalized/om.csv and om_foreign.csv
"""

import io
import os
import re
import sys
import urllib.request
import zipfile
from io import StringIO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import pandas as pd

from afrobarometer import round_within_rows
from om_geo import GOVERNORATE_2024, WILAYAT_2024, OMANIS, EXPATRIATES
import origin_religion as origin

RAW = os.path.join(ROOT, "data", "raw", "om")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "om.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "om_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
YEARBOOK_URL = ("http://web.archive.org/web/20250731134218id_/https://www.ncsi.gov.om/Elibrary/"
                "LibraryContentDoc/bar_Statistical_Year_Book_2025_Issue53_2_"
                "dc2e45d2-169c-43bb-831f-995eddafb309.pdf")
YEARBOOK = os.path.join(RAW, "ncsi_statistical_yearbook_2025.pdf")
YEARBOOK_SIZE = 16_718_500
YEARBOOK_PAGES = 306
GLMM = "https://gulfmigration.grc.net/"
GLMM_PAGES = {
    "nationality_sex_2018": "oman-population-by-country-of-citizenship-selected-nationalities-and-"
                            "sex-2018/",
    "workers_2013_2022": "oman-employed-foreign-workers-by-country-of-citizenship-all-sectors-"
                         "selected-nationalities-2013-2022/",
    "governorate_wilaya_2021_2023": "oman-population-by-nationality-omani-non-omani-governorate-"
                                    "and-wilaya-administrative-region-of-residence-2021-2023/",
}

OTHER_NODE = "other.om"
POPULATION = OMANIS + EXPATRIATES                      # 5,268,072, Table 8-2
EXPAT_M, EXPAT_F = 1_723_012, 560_267                  # Table 8-2
OMANI_M, OMANI_F = 1_501_736, 1_483_057

# Table 8-4: governorate -> (the label as printed, expatriate workers total, women, men)
WORKERS_2024 = {
    "Muscat": ("Muscat", 670_482, 109_394, 561_088),
    "Dhofar": ("Dhofar", 223_362, 21_544, 201_818),
    "Musandam": ("Musandam", 14_655, 1_515, 13_140),
    "Al Buraymi": ("Al Buraymi", 43_524, 5_148, 38_376),
    "Ad Dakhliyah": ("Ad - Dakhliyah", 131_546, 22_767, 108_779),
    "Al Batinah North": ("Al-Batinah North", 275_518, 38_177, 237_341),
    "Al Batinah South": ("Al - Batinah South", 157_728, 22_288, 135_440),
    "Ash Sharqiyah South": ("Ash - Sharqiyah South", 106_851, 15_999, 90_852),
    "Ash Sharqiyah North": ("Ash-Sharqiyah North", 96_589, 10_876, 85_713),
    "Adh Dhahirah": ("Adh - Dhahirah", 58_050, 9_445, 48_605),
    "Al Wusta": ("Al - Wusta", 30_621, 1_419, 29_202),
}
WORKERS_M, WORKERS_F = 1_550_354, 258_572

# Tables 17-4 (government) and 18-4 (private, family, communal, other), 2024:
# printed label -> (key, total, women, men). Keys are ISO codes, or OTHER / ARAB.
GOV_SECTOR = [("Egyptians", "EG", 9_333, 2_703, 6_630), ("Sudanis", "SD", 1_708, 823, 885),
              ("Jordanians", "JO", 446, 152, 294), ("Other Arabs", "ARAB", 3_008, 599, 2_409),
              ("Indians", "IN", 19_876, 7_711, 12_165), ("Pakistanis", "PK", 1_952, 190, 1_762),
              ("Other Nationalities", "OTHER", 6_445, 2_605, 3_840)]
GOV_OMANI = (392_021, 105_126, 286_895)
GOV_TOTAL = (434_789, 119_909, 314_880)
PRIVATE_SECTOR = [("Bangladeshis", "BD", 635_787, 25_791, 609_996),
                  ("Indians", "IN", 486_751, 42_397, 444_354),
                  ("Pakistanis", "PK", 315_345, 2_424, 312_921),
                  ("Philipinos", "PH", 42_475, 35_432, 7_043),
                  ("Egyptians", "EG", 36_587, 5_624, 30_963),
                  ("Myanmar", "MM", 33_059, 32_696, 363),
                  ("Sri Lankans", "LK", 23_447, 16_351, 7_096),
                  ("Tanzanian", "TZ", 23_435, 21_213, 2_222),
                  ("Sudanese", "SD", 21_830, 5_095, 16_735),
                  ("Other Nationalities", "OTHER", 147_442, 56_766, 90_676)]
PRIVATE_OMANI = (466_582, 144_096, 322_486)
PRIVATE_TOTAL = (2_232_740, 387_885, 1_844_855)

# GLMM's mid-2018 table: label -> ISO (None: a total row)
POP_2018 = {"India": "IN", "Bangladesh": "BD", "Pakistan": "PK", "Egypt": "EG",
            "Philippines": "PH", "Uganda": "UG", "Sri Lanka": "LK", "Nepal": "NP",
            "Tanzania": "TZ", "Indonesia": "ID", "Ethiopia": "ET"}
NON_OMANIS_2018 = (1_690_362, 332_108, 2_022_470)
# the 2018 table's nationalities that the 2024 tables fold into "Other nationalities"
OTHER_WOMEN_FROM_2018 = ["UG", "ID", "ET", "NP"]
WORKERS_2018 = {"Egypt": "EG", "Bangladesh": "BD", "India": "IN", "Pakistan": "PK",
                "Philippines": "PH", "Sri Lanka": "LK", "Nepal": "NP", "Tanzania": "TZ",
                "Uganda": "UG"}
ARAB_ROW = ["All Middle East-North Africa"]
INDIA = "IN"

CHRISTIAN_BAND = (0.2, 2.0)
# note_public's figures, measured 2026-09-15 and asserted
NOTE = dict(omanis=2984793, expatriates=2283279, foreign_muslim=1562977, non_muslim=720302,
            christians=133118, hindus=502123, buddhists=60559, unaffiliated=1406)


def fetch():
    os.makedirs(RAW, exist_ok=True)

    def get(url, dst, ok, minsize):
        if os.path.exists(dst) and os.path.getsize(dst) > minsize:
            return
        print("  GET", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r:
            data = r.read()
        if not ok(data):
            raise SystemExit(f"{url} did not return the expected file ({len(data):,} bytes)")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)

    get(YEARBOOK_URL, YEARBOOK,
        lambda d: d.startswith(b"%PDF") and len(d) == YEARBOOK_SIZE and b"%%EOF" in d[-2048:],
        10_000_000)
    for key, path in GLMM_PAGES.items():
        get(GLMM + path, os.path.join(RAW, f"glmm_{key}.html"), lambda d: b"<table" in d, 10_000)


# ---------------------------------------------------------------------------------------------
# the yearbook
# ---------------------------------------------------------------------------------------------
NUM = re.compile(r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?")


def load_yearbook():
    import fitz

    with open(YEARBOOK, "rb") as fh:
        data = fh.read()
    if len(data) != YEARBOOK_SIZE or b"%%EOF" not in data[-2048:]:
        raise SystemExit(f"{YEARBOOK} is {len(data):,} bytes, pinned {YEARBOOK_SIZE:,}, or has no %%EOF")
    doc = fitz.open(YEARBOOK)
    if len(doc) != YEARBOOK_PAGES:
        raise SystemExit(f"the yearbook has {len(doc)} pages, expected {YEARBOOK_PAGES}")
    return lambda p: re.sub(r"\s+", " ", doc[p - 1].get_text())


def as_num(tok):
    return float(tok) if "." in tok and "," not in tok else int(tok.replace(",", ""))


class Reader:
    """Finds printed row labels in page order and the numbers after each."""

    def __init__(self, text, where):
        self.text, self.pos, self.where = text, 0, where

    def row(self, label, n):
        pat = r"\s*".join(re.escape(w) for w in label.replace("-", " - ").split())
        m = re.compile(pat, re.I).search(self.text, self.pos)
        if not m:
            raise SystemExit(f"{self.where}: label {label!r} not found after position {self.pos}")
        nums, p = [], m.end()
        for nm in NUM.finditer(self.text, p):
            nums.append(as_num(nm.group()))
            p = nm.end()
            if len(nums) == n:
                break
        if len(nums) < n:
            raise SystemExit(f"{self.where}: fewer than {n} numbers after {label!r}")
        self.pos = p
        return nums


def yearbook_checks(page):
    # Table 7-2: every governorate and wilaya, 2024 pair first, then 2023, then 2022
    r = Reader(page(32) + " " + page(33), "Table 7-2")
    y2023 = {}
    for gov, pair in GOVERNORATE_2024.items():
        got = r.row(gov, 6)
        if (got[0], got[1]) != pair:
            raise SystemExit(f"Table 7-2 {gov}: printed {got[:2]}, om_geo has {pair}")
        y2023[gov] = (got[2], got[3])
        for (g, w), wpair in WILAYAT_2024.items():
            if g != gov:
                continue
            got = r.row(w, 6)
            if (got[0], got[1]) != wpair:
                raise SystemExit(f"Table 7-2 {gov} / {w}: printed {got[:2]}, om_geo has {wpair}")
    got = r.row("Sultanate Total", 2)
    if tuple(got) != (EXPATRIATES, OMANIS):
        raise SystemExit(f"Table 7-2 national row {got}")
    print("  yearbook Table 7-2: all 11 governorates and 63 wilayat as om_geo pins them (end-2024)")

    # Table 8-2's total row
    t = page(34)
    seq = [5_268_072, 2_043_324, 3_224_748, 100.0, EXPATRIATES, EXPAT_F, EXPAT_M, 100.0, OMANIS,
           OMANI_F, OMANI_M]
    pat = r"\D+".join(f"{v:,}" if isinstance(v, int) else r"100\.0" for v in seq)
    if not re.search(pat, t) or EXPAT_M + EXPAT_F != EXPATRIATES or OMANI_M + OMANI_F != OMANIS:
        raise SystemExit("Table 8-2's total row is not the pinned register totals by sex")
    print(f"  yearbook Table 8-2: {EXPATRIATES:,} expatriates ({EXPAT_M:,} men, {EXPAT_F:,} women)")

    # Table 8-4: grand total, expatriate, Omani, each as (total, women, men)
    r = Reader(page(58), "Table 8-4")
    for gov, (label, t_, f_, m_) in WORKERS_2024.items():
        got = r.row(label, 9)
        if tuple(got[3:6]) != (t_, f_, m_) or t_ != f_ + m_ or got[0] != got[3] + got[6]:
            raise SystemExit(f"Table 8-4 {gov}: printed {got}, pinned {(t_, f_, m_)}")
    got = r.row("Total", 9)
    if tuple(got[3:6]) != (WORKERS_M + WORKERS_F, WORKERS_F, WORKERS_M):
        raise SystemExit(f"Table 8-4 total {got}")
    if sum(v[1] for v in WORKERS_2024.values()) != WORKERS_M + WORKERS_F:
        raise SystemExit("Table 8-4's governorates do not sum to its total")
    print(f"  yearbook Table 8-4: expatriate workers by governorate and sex, {WORKERS_M + WORKERS_F:,}")

    # Tables 17-4 then 18-4 on page 67: female %, total, women, men
    r = Reader(page(67), "Tables 17-4 and 18-4")
    for rows, omani, total in ((GOV_SECTOR, GOV_OMANI, GOV_TOTAL),
                               (PRIVATE_SECTOR, PRIVATE_OMANI, PRIVATE_TOTAL)):
        got = r.row("Omani", 4)
        if tuple(got[1:]) != omani:
            raise SystemExit(f"Omani row {got}")
        for label, _k, t_, f_, m_ in rows:
            got = r.row(label, 4)
            if tuple(got[1:]) != (t_, f_, m_) or t_ != f_ + m_:
                raise SystemExit(f"{label}: printed {got}, pinned {(t_, f_, m_)}")
        got = r.row("Total", 4)
        if tuple(got[1:]) != total:
            raise SystemExit(f"total row {got}")
        if tuple(sum(x) for x in zip(omani, *[(t_, f_, m_) for _l, _k, t_, f_, m_ in rows])) != total:
            raise SystemExit("a sector's nationality rows do not sum to its total")
    exp_w = [sum(x[i] for x in GOV_SECTOR + PRIVATE_SECTOR) for i in (3, 4)]
    if tuple(exp_w) != (WORKERS_F, WORKERS_M):
        raise SystemExit(f"Tables 17-4 and 18-4 give expatriate workers {exp_w}, Table 8-4 "
                         f"{(WORKERS_F, WORKERS_M)}")
    print("  yearbook Tables 17-4 and 18-4: expatriate workers by nationality and sex, equal to "
          "Table 8-4's total by sex")
    return y2023


def glmm_table(key):
    with open(os.path.join(RAW, f"glmm_{key}.html"), encoding="utf-8") as fh:
        return pd.read_html(StringIO(fh.read()), thousands=None)[0]


def glmm_int(v):
    s = str(v).strip()
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):          # the 2018 table writes 1.299.741
        return int(s.replace(".", ""))
    if re.fullmatch(r"\d{1,3}(,\d{3})+", s):           # the worker series writes 31,725
        return int(s.replace(",", ""))
    if re.fullmatch(r"\d+(\.0)?", s):
        return int(float(s))
    raise SystemExit(f"GLMM cell {v!r} is not a count")


def glmm_checks(y2023):
    t = glmm_table("governorate_wilaya_2021_2023")
    rows = [set(str(x).replace(",", "").replace(".0", "").strip() for x in t.iloc[i])
            for i in range(len(t))]
    miss = [gov for gov, (e, o) in y2023.items()
            if not any({str(e), str(o)} <= row for row in rows)]
    if miss:
        raise SystemExit(f"GLMM's 2021-2023 table has no row with the yearbook's 2023 pair for {miss}")
    print("  GLMM 2021-2023: every governorate's 2023 Omani and expatriate counts equal the yearbook's")

    t = glmm_table("nationality_sex_2018")
    lab = t.iloc[:, 0].astype(str).str.strip()
    pop = {}
    for i in range(len(t)):
        if lab[i] in POP_2018:
            pop[POP_2018[lab[i]]] = tuple(glmm_int(t.iat[i, j]) for j in (1, 2, 3))
        elif lab[i].lower() == "total non-omanis":
            if tuple(glmm_int(t.iat[i, j]) for j in (1, 2, 3)) != NON_OMANIS_2018:
                raise SystemExit("GLMM 2018: the non-Omani total is not the pinned one")
    if set(pop) != set(POP_2018.values()) or any(m + f != tt for m, f, tt in pop.values()):
        raise SystemExit(f"GLMM 2018 table: rows {sorted(pop)}")

    t = glmm_table("workers_2013_2022")
    years = [str(x).replace(".0", "") for x in t.iloc[0]]
    c18 = years.index("2018")
    w18 = {}
    for i in range(1, len(t)):
        name = str(t.iat[i, 0]).strip()
        if name in WORKERS_2018:
            w18[WORKERS_2018[name]] = glmm_int(t.iat[i, c18])
    if set(w18) != set(WORKERS_2018.values()):
        raise SystemExit(f"GLMM worker series 2018: rows {sorted(w18)}")
    total_row = [i for i in range(1, len(t)) if str(t.iat[i, 0]).strip() == "Total"]
    w18_total = glmm_int(t.iat[total_row[0], c18])
    parts = sum(glmm_int(t.iat[i, c18]) for i in range(1, total_row[0])
                if str(t.iat[i, c18]) not in ("nan", ""))
    if len(total_row) != 1 or parts != w18_total:
        raise SystemExit(f"GLMM worker series 2018: rows sum to {parts:,}, total {w18_total:,}")
    print(f"  GLMM mid-2018 population by nationality and sex (11 nationalities, "
          f"{sum(v[2] for v in pop.values()):,} of {NON_OMANIS_2018[2]:,} non-Omanis); 2018 workers for "
          f"{len(w18)} of them")
    return pop, w18, w18_total


# ---------------------------------------------------------------------------------------------
# religion
# ---------------------------------------------------------------------------------------------
def pew_table():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    return t[t["Year"] == 2020].set_index("Country")


def fold(comp):
    out = {}
    for node, s in comp.items():
        node = "islam" if node.startswith("islam") else node
        out[node] = out.get(node, 0.0) + s
    return out


def composition(pew, iso=None, regional=None):
    if regional:
        row = {f: float(pew.loc[regional, f].sum()) for f in origin.FAMILIES}
        return fold(origin.composition("XX", row, OTHER_NODE))
    pn = origin.PEW_BY_ISO[iso]
    if pn is None or pn not in pew.index:
        raise SystemExit(f"Pew has no row for {iso}")
    return fold(origin.composition(iso, {f: float(pew.loc[pn, f]) for f in origin.FAMILIES},
                                   OTHER_NODE))


def pew_share(pew, family):
    return float(pew.loc["Oman", family]) / float(pew.loc["Oman", "Population"])


def mix(weights, comps):
    """{node: share} for {key: weight} over {key: composition}."""
    tot = sum(weights.values())
    out = {}
    for k, w in weights.items():
        for node, s in comps[k].items():
            out[node] = out.get(node, 0.0) + w / tot * s
    return out


def main():
    if "--fetch" in sys.argv or not os.path.exists(YEARBOOK) or not all(
            os.path.exists(os.path.join(RAW, f"glmm_{k}.html")) for k in GLMM_PAGES):
        fetch()
    page = load_yearbook()
    y2023 = yearbook_checks(page)
    pop18, w18, w18_total = glmm_checks(y2023)
    pew = pew_table()

    # ---- workers by nationality and sex, nationally ----
    men, women = {}, {}
    for _label, key, _t, f_, m_ in GOV_SECTOR + PRIVATE_SECTOR:
        men[key] = men.get(key, 0) + m_
        women[key] = women.get(key, 0) + f_
    other_m, other_f = men.pop("OTHER"), women.pop("OTHER")
    women_2018 = {iso: pop18[iso][1] for iso in OTHER_WOMEN_FROM_2018}
    print(f"\n  other nationalities: {other_m:,} men on the named men's mix; {other_f:,} women on "
          f"2018's {', '.join(OTHER_WOMEN_FROM_2018)} women ({sum(women_2018.values()):,}, "
          f"{sum(women_2018.values()) / other_f:.0%} of the cell)")
    for iso, n in women_2018.items():
        women[iso] = women.get(iso, 0) + other_f * n / sum(women_2018.values())
    named_m = {k: v for k, v in men.items()}
    for k, v in named_m.items():
        men[k] = v + other_m * v / sum(named_m.values())

    # ---- dependants: mid-2018 population less 2018 workers, by nationality ----
    dep18 = {iso: pop18[iso][2] - w18[iso] for iso in w18}
    print("  dependants in 2018 (population less workers): "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(dep18.items(), key=lambda kv: -kv[1])))
    if min(dep18.values()) < 0:
        print("    negative rows set to zero: "
              + ", ".join(k for k, v in dep18.items() if v < 0))
        dep18 = {k: max(0, v) for k, v in dep18.items()}
    dep_other = NON_OMANIS_2018[2] - w18_total - sum(dep18.values())
    print(f"  2018's other nationalities' dependants, drawn at the named mix: {dep_other:,} "
          f"({NON_OMANIS_2018[2]:,} non-Omanis less {w18_total:,} workers less the named dependants)")

    dependants = {g: GOVERNORATE_2024[g][0] - WORKERS_2024[g][1] for g in GOVERNORATE_2024}
    if min(dependants.values()) < 0:
        raise SystemExit(f"a governorate has more expatriate workers than expatriates: {dependants}")
    dep_tot = sum(dependants.values())
    print(f"  dependants in 2024: {dep_tot:,} ({EXPAT_M - WORKERS_M:,} men, {EXPAT_F - WORKERS_F:,} women)")

    # ---- compositions ----
    keys = set(men) | set(women) | set(dep18)
    comps = {k: composition(pew, regional=ARAB_ROW) if k == "ARAB" else composition(pew, iso=k)
             for k in keys}

    # India's Hindu share from Pew's Oman estimate
    share_m = {k: v / sum(men.values()) for k, v in men.items()}
    share_f = {k: v / sum(women.values()) for k, v in women.items()}
    share_d = {k: v / sum(dep18.values()) for k, v in dep18.items()}
    people = {k: WORKERS_M * share_m.get(k, 0) + WORKERS_F * share_f.get(k, 0)
              + dep_tot * share_d.get(k, 0) for k in keys}
    target = pew_share(pew, "Hindus") * POPULATION
    others = sum(people[k] * comps[k].get("hinduism", 0.0) for k in keys if k != INDIA)
    h_row = comps[INDIA].get("hinduism", 0.0)
    h = (target - others) / people[INDIA]
    print(f"\n  India ({people[INDIA]:,.0f} in the layer): Pew 2020's Oman Hindus are "
          f"{pew_share(pew, 'Hindus'):.3%}, {target:,.0f} of the register; other nationalities give "
          f"{others:,.0f}, so Indians are {h:.2%} Hindu (India's own row {h_row:.2%}, which would give "
          f"{others + people[INDIA] * h_row:,.0f})")
    if not 0.0 < h < h_row:
        raise SystemExit("India's Hindu share from Pew's Oman estimate is not between 0 and India's row")
    comps[INDIA]["islam"] = comps[INDIA].get("islam", 0.0) + (h_row - h)
    comps[INDIA]["hinduism"] = h
    for k, c in comps.items():
        if abs(sum(c.values()) - 1) > 1e-9:
            raise SystemExit(f"composition {k} sums to {sum(c.values())}")
    mix_m, mix_f, mix_d = mix(men, comps), mix(women, comps), mix(dep18, comps)
    nodes = sorted(set(mix_m) | set(mix_f) | set(mix_d))
    chr_nodes = [n for n in nodes if n.startswith("christianity")]
    print("  Christian share: male workers {:.2%}, female workers {:.2%}, dependants {:.2%}".format(
        *(sum(x.get(n, 0) for n in chr_nodes) for x in (mix_m, mix_f, mix_d))))

    # ---- per governorate, then per wilaya ----
    gov_counts = pd.DataFrame({g: {n: WORKERS_2024[g][3] * mix_m.get(n, 0)
                                   + WORKERS_2024[g][2] * mix_f.get(n, 0)
                                   + dependants[g] * mix_d.get(n, 0) for n in nodes}
                               for g in GOVERNORATE_2024}).T[nodes]
    gov_share = gov_counts.div(gov_counts.sum(axis=1), axis=0)
    wil = list(WILAYAT_2024)
    wm = pd.DataFrame([gov_share.loc[g].to_numpy() * WILAYAT_2024[(g, w)][0] for g, w in wil],
                      index=[f"{g}|{w}" for g, w in wil], columns=nodes)
    fcounts = round_within_rows(wm)
    for g, w in wil:
        if int(fcounts.loc[f"{g}|{w}"].sum()) != WILAYAT_2024[(g, w)][0]:
            raise SystemExit(f"{g} / {w}: rounded expatriates do not sum to the register")

    print("\n  per governorate: Christians and Hindus as drawn, and on one national mix for everyone")
    flat = {n: gov_counts[n].sum() / EXPATRIATES for n in nodes}
    for g in sorted(GOVERNORATE_2024, key=lambda x: -GOVERNORATE_2024[x][0]):
        exp, om = GOVERNORATE_2024[g]
        c = sum(gov_counts.loc[g, n] for n in chr_nodes)
        c_flat = exp * sum(flat[n] for n in chr_nodes)
        print(f"      {g:<20} Christians {c:>8,.0f} ({c_flat:>8,.0f} flat)  Hindus "
              f"{gov_counts.loc[g, 'hinduism']:>8,.0f}  ({100 * c / (exp + om):.2f}% Christian)")

    chr_ = int(fcounts[chr_nodes].sum().sum())
    ratio = (chr_ / POPULATION) / pew_share(pew, "Christians")
    print(f"\n  witness: {chr_:,} Christians, {chr_ / POPULATION:.3%} of the register; Pew 2020 has "
          f"{pew_share(pew, 'Christians'):.3%} for everyone living in Oman; ratio {ratio:.2f}, band "
          f"{CHRISTIAN_BAND}")
    for fam, node in (("Buddhists", "buddhism"), ("Religiously_unaffiliated", "unaffiliated"),
                      ("Jews", "judaism"), ("Hindus", "hinduism")):
        got = int(fcounts[node].sum()) if node in fcounts else 0
        print(f"      {fam}: {got:,} drawn against Pew's {pew_share(pew, fam) * POPULATION:,.0f} (not asserted)")
    if not CHRISTIAN_BAND[0] <= ratio <= CHRISTIAN_BAND[1]:
        raise SystemExit("the layer's Christians are outside the band; read sources/om.md §4")
    bud = {k: (WORKERS_M * share_m.get(k, 0), WORKERS_F * share_f.get(k, 0),
               dep_tot * share_d.get(k, 0)) for k in keys}
    bud = {k: tuple(x * comps[k].get("buddhism", 0.0) for x in v) for k, v in bud.items()}
    print("      Buddhists by nationality (male workers, female workers, dependants): "
          + ", ".join(f"{k} {m:,.0f}/{f:,.0f}/{d:,.0f}" for k, (m, f, d)
                      in sorted(bud.items(), key=lambda kv: -sum(kv[1])) if sum((m, f, d)) >= 500))

    # ---- write ----
    out = pd.DataFrame({"geo_id": [f"{g}|{w}" for g, w in wil], "geo_level": "wilaya",
                        "geo_name": [w for _g, w in wil], "source_category": "Omani citizens",
                        "count": [WILAYAT_2024[k][1] for k in wil], "basis": "estimate",
                        "year": 2024, "source_id": "om_register2024_omanis",
                        "note": "no source asks religion; every Omani is drawn on Islam (sources/om.py), "
                                "on NCSI's end-2024 register count of Omanis per wilaya"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    ext = fcounts.stack().rename("count").reset_index()
    ext.columns = ["geo_id", "node", "count"]
    ext = ext[ext["count"] > 0].copy()
    ext["geo_level"] = "wilaya"
    ext["geo_name"] = ext["geo_id"].str.split("|").str[1]
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_sex_derived"
    ext["year"] = 2024
    ext["source_id"] = "register2024_expatriates_x_pew2020"
    ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
         "source_id"]].to_csv(OUT_FOREIGN, index=False, encoding="utf-8")

    fx = ext.groupby("node")["count"].sum().sort_values(ascending=False)
    muslim_f = int(fx.get("islam", 0))
    non_muslim = int(ext["count"].sum()) - muslim_f
    print(f"\nwrote {OUT} ({OMANIS:,} Omanis) and {OUT_FOREIGN} ({int(ext['count'].sum()):,} "
          f"expatriates, {ext['node'].nunique()} nodes)")
    print(f"    drawn Muslim {(OMANIS + muslim_f) / POPULATION:.3%}; non-Muslim {non_muslim:,}; Pew 2020 "
          f"for everyone: {pew_share(pew, 'Muslims'):.3%} Muslim")
    print("    expatriates: " + ", ".join(f"{n} {int(v):,}" for n, v in fx.items()))
    got = dict(omanis=OMANIS, expatriates=int(ext["count"].sum()), foreign_muslim=muslim_f,
               non_muslim=non_muslim, christians=chr_, hindus=int(fx.get("hinduism", 0)),
               buddhists=int(fx.get("buddhism", 0)), unaffiliated=int(fx.get("unaffiliated", 0)))
    print(f"\n  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
