"""Saudi Arabia: nobody is asked their religion. Citizens are drawn on Islam, and non-Saudis by
nationality and sex in each region, all on the 2022 census.

Reads, all fetched into data/raw/sa/ by --fetch:

  * GASTAT, *Saudi Census 2022: Population Summary Report* (17 pages, June 2023), GASTAT's own PDF,
    now only on the Wayback Machine because `portal.saudicensus.sa` no longer resolves;
  * the census portal's tables as mirrored, with citations, by the Gulf Labour Markets, Migration and
    Population programme (GLMM, `gulfmigration.grc.net`): Saudis and non-Saudis by region; non-Saudis
    by country of citizenship and sex in four tables (Arab; non-Arab Asian; European; Sub-Saharan
    African); and the 34 largest nationalities;
  * the Royal Commission for Riyadh City's open data portal (KSA Open Data License): the census by
    citizenship and sex for Ar Riyadh region, a witness for the report's sex ratios;
  * Pew Research Center, *Religious Composition 2010-2020* (data/raw/estimates/pew.zip).

Writes data/normalized/sa.csv (citizens) and data/normalized/sa_foreign.csv (non-Saudis).
`sources/sa.md` is the record in prose.

## NOBODY IS ASKED

The 2022 census asks no religion (sources.md §11r). The Arab Barometer's 1,404 wave II Saudi rows
leave `Q1012` empty and wave V names the country against zero rows (§11af). The Ministry of Islamic
Affairs is reported to publish counts of mosques, not people (§11ao). Built on Anita's Maghreb and
Mauritania rulings (ask/RULINGS.md 2026-09-15 and 2026-09-16): a near-uniformly Muslim country is
drawn on a compiler's figure, with foreigners placed by region.

## CITIZENS ARE ALL DRAWN ON ISLAM

Pew's Saudi estimate is for everyone living in the country, so drawing its shares beside a foreigner
layer counts foreign non-Muslims twice (as for Mauritania, sources/mr.md §4). Every citizen is drawn on
`islam`: citizens are legally Muslim, and no source counts a citizen who is not. The US State
Department's 2023 report puts citizens at 85-90% Sunni and 10-12% Shia; no Sunni/Shia split is drawn,
because Saudi Shia have been attacked and discriminated against, which is spec §14's case and an ask.

## NON-SAUDIS: REGION, SEX, NATIONALITY

The census publishes non-Saudis per region, and their nationality only for the whole country. It
publishes both by sex: nationality by sex in the four GLMM tables, and each region's non-Saudi sex
ratio in the report's Figure 12. The sexes carry different nationalities (1,181 Bangladeshi men per
100 women, 61 Filipino men per 100 women), so each region's non-Saudi men take the national mix of
non-Saudi men and its women the national mix of non-Saudi women. That assumes the mix is the same
everywhere within a sex, which is weaker than the same mix everywhere; `main` prints both.

  * **`Other` rows** of a table take Pew's regional row (All Asia-Pacific, All Europe, All
    Sub-Saharan Africa). Non-Saudis in none of the four tables (national totals less the four
    tables; the report's prose puts America at 0.3%) take Pew's All North America and All Latin
    America-Caribbean summed.
  * **Burma is the Rohingya.** GAStat's 163,717 `Burma` nationals are drawn on Islam, not Pew's
    Myanmar row (88% Buddhist). Refugee Law Initiative, Lysa 2023: the Rohingya are "often referred
    to as 'the Burmese' in Saudi Arabia", many arrived on Bangladeshi, Pakistani or Indian documents,
    and the authorities issued 250,000 special residency permits in 2017; the MHRSD's 2022 private
    sector table lists `Myanmar/ holder of a Pakistani passport`, `...Bangladeshi passport` and
    `Myanmar/ resident`. Pew's Myanmar row would draw about 144,000 Buddhists; Pew's whole Saudi
    estimate has 22,774.
  * **India's Hindu share comes from Pew's own Saudi estimate.** Pew's *Faith on the Move* (2012,
    pp.21-22) says migrants from India to Muslim-majority Middle Eastern countries are mostly Muslim,
    and estimates them from Egypt's census of its Indian migrants; Pew's India row (79% Hindu) does
    not know that. So India's Hindu share is set so the layer's Hindus equal Pew 2020's Saudi Hindu
    share of the census count, every other nationality kept at its own row, and the Hindus removed
    are drawn on Islam. Indian Christians stay at India's row. Printed; see sources/sa.md §4.
  * **everyone else**: Pew 2020 per country through `taxonomy/origin_religion.py`, Muslim branches
    folded to `islam` (no sect is drawn anywhere in Saudi Arabia), Pew's unplaced `Other religions`
    on `other.sa`.

`christian_witness` holds the layer's Christians against Pew 2020's Saudi share inside
`CHRISTIAN_BAND`. The band was written after a rough sum over the largest nationalities while
scouting, so it is not blind; sources/sa.md §4 says so.

Usage:
    python sources/sa.py --fetch    six GLMM pages, the census report (0.4 MB), one RCRC API call
    python sources/sa.py            rebuild data/normalized/sa.csv and sa_foreign.csv
"""

import io
import json
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
from sa_geo import REGIONS, REGION_2022
import origin_religion as origin

RAW = os.path.join(ROOT, "data", "raw", "sa")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "sa.csv")
OUT_FOREIGN = os.path.join(ROOT, "data", "normalized", "sa_foreign.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
GLMM = "https://gulfmigration.grc.net/"
PAGES = {
    "region": "saudi-arabia-population-by-nationality-saudi-non-saudi-and-administrative-region-"
              "governorate-of-residence-2022/",
    "arab": "saudi-arabia-arab-population-by-country-of-citizenship-and-sex-census-2022/",
    "asian": "saudi-arabia-non-arab-asian-population-by-region-country-of-citizenship-and-sex-"
             "census-2022/",
    "europe": "saudi-arabia-european-population-by-region-country-of-citizenship-and-sex-"
              "census-2022/",
    "ssa": "saudi-arabia-sub-saharan-african-population-by-country-of-citizenship-and-sex-"
           "census-2022/",
    "selected": "saudi-arabia-non-saudi-population-by-country-of-citizenship-and-sex-selected-"
                "countries-2022/",
}
REPORT_URL = ("http://web.archive.org/web/20230531131948id_/https://portal.saudicensus.sa/"
              "static-assets/media/content/20230531_GASTAT_Population_Report.pdf"
              "?crafterSite=gastat-portal")
REPORT = os.path.join(RAW, "20230531_GASTAT_Population_Report.pdf")
REPORT_SIZE = 393_645
RCRC_URL = ("https://opendata.rcrc.gov.sa/api/explore/v2.1/catalog/datasets/"
            "population-by-age-citizenship-gender-and-governorate-2022/records?"
            "select=admregcen,gov,ctz,gender,sum(obsvalue)%20as%20n"
            "&group_by=admregcen,gov,ctz,gender&limit=100")
RCRC = os.path.join(RAW, "rcrc_population_citizenship_gender_2022.json")

OTHER_NODE = "other.sa"
CITIZENS, NON_SAUDIS, TOTAL = 18_792_262, 13_382_962, 32_175_224
NON_SAUDI_M, NON_SAUDI_F = 10_244_464, 3_138_498

# map name (sa_geo.REGIONS) -> (GLMM label, the report's Figure 11 label, Figure 12's three bars:
# males per 100 females overall, Saudi, non-Saudi). Figure 12's bars were read off the rendered
# chart (page 15), where every bar carries its value; the text layer holds the values but not in the
# chart's order, so `report_checks` asserts the values are there and that each region's overall bar
# is what its Saudi and non-Saudi bars and the census counts give.
CENSUS = {
    "Riyadh": ("Riyadh", "Ar Riyadh", (168, 102, 310)),
    "Makkah": ("Makkah Al Mukarramah", "Makkah Al Mukarramah", (155, 100, 264)),
    "Eastern Province": ("Eastern Region", "Eastern Region", (176, 104, 430)),
    "Madinah": ("Al Madinah Al Munawwarah", "Al Madinah Al Munawwarah", (149, 99, 335)),
    "Asir": ("Aseer", "Aseer", (146, 98, 510)),
    "Jazan": ("Jazan", "Jazan", (132, 99, 300)),
    "Qassim": ("Al Qaseem", "Al Qaseem", (146, 99, 422)),
    "Tabuk": ("Tabuk", "Tabuk", (147, 102, 485)),
    "Hail": ("Hail", "Hail", (146, 101, 443)),
    "Al Jawf": ("Al Jawf", "Al Jawf", (138, 100, 415)),
    "Najran": ("Najran", "Najran", (144, 100, 338)),
    "Northern Borders": ("Northern Borders", "Nothern Borders", (135, 101, 327)),
    "Al Bahah": ("Al Bahah", "Al Bahah", (134, 96, 432)),
}
OVERALL_TOL = 1.5          # a region's overall bar against its Saudi and non-Saudi bars

# ---- the four nationality tables: (subtotal row label, {row label: ISO, or None for a regional row})
ARAB = {"Yemen": "YE", "Egypt": "EG", "Sudan": "SD", "Syria": "SY", "Jordan": "JO",
        "Palestine": "PS", "Lebanon": "LB", "Kuwait": "KW", "Morocco": "MA", "Tunisia": "TN",
        "Mauritania": "MR", "Bahrain": "BH", "Algeria": "DZ", "UAE": "AE", "Qatar": "QA",
        "Iraq": "IQ", "Oman": "OM", "Libya": "LY"}
ARAB_IN_ASIA = {"Yemen", "Syria", "Jordan", "Palestine", "Lebanon", "Kuwait", "Bahrain", "UAE",
                "Qatar", "Iraq", "Oman"}
ASIAN = [
    ("West Asia (non-Arab states)", {"Turkey": "TR", "Azerbaijan": "AZ", "Cyprus": "CY",
                                     "Armenia": "AM", "Georgia": "GE", "Other": None}),
    ("Central Asia", {"Uzbekistan": "UZ", "Kyrgyzstan": "KG", "Kazakhstan": "KZ",
                      "Tajikistan": "TJ", "Turkmenistan": "TM"}),
    ("South Asia", {"Bangladesh": "BD", "India": "IN", "Pakistan": "PK", "Nepal": "NP",
                    "Afghanistan": "AF", "Sri Lanka": "LK", "British Indian Ocean Territory": None,
                    "Iran": "IR", "Maldives": "MV", "Other": None}),
    ("South-East Asia", {"Philippines": "PH", "Indonesia": "ID", "Myanmar": "MM",
                         "Malaysia": "MY", "Thailand": "TH", "VietNam": "VN", "Singapore": "SG",
                         "Cambodia": "KH", "Brunei Darussalam": "BN",
                         "Lao Peoples Democratic Republic": "LA", "Other": None}),
    ("East Asia", {"China": "CN", "Korea (the Republic of)": "KR",
                   "Taiwan (Province of China)": "TW", "Japan": "JP", "Mongolia": "MN",
                   "Others": None}),
]
EUROPE = [
    ("Total", {"France": "FR", "Germany": "DE", "Netherlands": "NL", "Belgium": "BE",
               "Austria": "AT", "Switzerland": "CH", "Luxembourg": "LU"}),
    ("Total", {"United Kingdom": "GB", "Ireland": "IE", "Sweden": "SE", "Denmark": "DK",
               "Norway": "NO", "Finland": "FI", "Lithuania": "LT", "Latvia": "LV", "Estonia": "EE",
               "Iceland": "IS", "Other": None}),
    ("Total", {"Spain": "ES", "Italy": "IT", "Greece": "GR", "Portugal": "PT",
               "Bosnia and Herzegovina": "BA", "Albania": "AL", "Serbia": "RS", "Croatia": "HR",
               "Macedonia": "MK", "Montenegro": "ME", "Andorra": "AD", "Malta": "MT",
               "Slovenia": "SI", "Other": None}),
    ("Total", {"Russian Federation": "RU", "Ukraine": "UA", "Romania": "RO", "Poland": "PL",
               "Czechia": "CZ", "Hungary": "HU", "Slovakia": "SK", "Belarus": "BY",
               "Bulgaria": "BG", "Moldova": "MD"}),
]
SSA = {"Ethiopia": "ET", "Uganda": "UG", "Kenya": "KE", "Nigeria": "NG", "Eritrea": "ER",
       "Chad": "TD", "Somalia": "SO", "Mali": "ML", "Niger": "NE", "Ghana": "GH", "Burundi": "BI",
       "Burkina Faso": "BF", "South Africa": "ZA", "Senegal": "SN", "Djibouti": "DJ",
       "Guinea": "GN", "Cameroon": "CM", "Madagascar": "MG", "Tanzania": "TZ",
       "Ivory Coast": "CI", "Gambia": "GM", "Benin": "BJ", "Sierra Leone": "SL", "Zimbabwe": "ZW",
       "Others": None}
TABLES = {"arab": ([("Total", ARAB)], None), "asian": (ASIAN, "Total"),
          "europe": (EUROPE, "Total Europe"), "ssa": ([("Total", SSA)], None)}
# how far a printed subtotal may be from its rows: the Arab and Sub-Saharan tables round some rows
# (Kuwait 50,000 against the selected table's 50,282; Ethiopia 159,300 against 159,221)
# The Asian table's 1: South Asia's `Other` row is one man with blank women and total, and the
# subtotal counts him among the men but not in the total.
SUBTOTAL_TOL = {"arab": 1_000, "asian": 1, "europe": 0, "ssa": 300}
SELECTED_TOL = {"arab": 300, "asian": 0, "europe": 0, "ssa": 100}
# cells GLMM leaves blank in a row whose subtotal shows what they are
BLANKS = {("asian", "South Asia", "Other"): (1, 0, 1),
          ("asian", "South-East Asia", "Other"): (1, 0, 1)}
REGIONAL_ROW = {"arab": ["All Middle East-North Africa"], "asian": ["All Asia-Pacific"],
                "europe": ["All Europe"], "ssa": ["All Sub-Saharan Africa"]}
REST_ROWS = ["All North America", "All Latin America-Caribbean"]
SELECTED_ALIAS = {"Burma": "Myanmar", "UK": "United Kingdom"}

# the report's continent prose, page 6
CONTINENT_PROSE = ("Asia (76.2%), followed by Africa (23.2%) and America (0.3%)",
                   "67.5% from Asia, 31.4% from Africa, 0.5% from Europe and 0.5% from the Americas")

ROHINGYA = "MM"
INDIA = "IN"
# Written 2026-09-15, after the rough sum described in the docstring: the layer's Christians as a
# share of the census count, over Pew 2020's Christian share for everyone living in Saudi Arabia.
CHRISTIAN_BAND = (0.5, 2.0)

# note_public's figures, measured 2026-09-15 and asserted
NOTE = dict(citizens=18792262, non_saudis=13382962, foreign_muslim=10921014, non_muslim=2461948,
            christians=1337918, hindus=843672, buddhists=120949, unaffiliated=60679)


def fetch():
    os.makedirs(RAW, exist_ok=True)

    def get(url, dst, ok):
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
            return
        print("  GET", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            data = r.read()
        if not ok(data):
            raise SystemExit(f"{url} did not return the expected file ({len(data):,} bytes)")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)

    for key, path in PAGES.items():
        get(GLMM + path, os.path.join(RAW, f"glmm_{key}.html"), lambda d: b"<table" in d)
    get(REPORT_URL, REPORT, lambda d: d.startswith(b"%PDF") and len(d) == REPORT_SIZE)
    get(RCRC_URL, RCRC, lambda d: json.loads(d.decode("utf-8")).get("total_count") == 20)


def cell(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "N.A.":
        return None
    if re.fullmatch(r"\d+(\.0)?", s):
        return int(float(s))
    raise SystemExit(f"cell {v!r} is not a count")


def glmm_rows(key, header):
    with open(os.path.join(RAW, f"glmm_{key}.html"), encoding="utf-8") as fh:
        t = pd.read_html(StringIO(fh.read()))[0]
    got = tuple(str(x).strip() for x in t.iloc[0, 1:4])
    if got != header or t.shape[1] != 4:
        raise SystemExit(f"GLMM {key}: header {got}, expected {header}")
    return [(str(t.iat[i, 0]).strip(), *(cell(t.iat[i, j]) for j in (1, 2, 3)))
            for i in range(1, len(t))]


def group_table(key):
    """{(table, subtotal, row): (males, females)} and the table's printed (males, females, total)."""
    rows = glmm_rows(key, ("Males", "Females", "Total"))
    groups, final = TABLES[key]
    out, i, grand = {}, 0, [0, 0, 0]
    for sub, members in groups:
        acc, seen = [0, 0, 0], set()
        while True:
            lab, m, f, t = rows[i]
            i += 1
            if lab == sub and len(seen) == len(members):
                diff = [p - a for p, a in zip((m, f, t), acc)]
                if any(d != 0 for d in diff):
                    print(f"    {key} {sub}: printed minus rows {diff}")
                if any(abs(d) > SUBTOTAL_TOL[key] for d in diff):
                    raise SystemExit(f"GLMM {key} {sub}: rows sum to {acc}, printed {(m, f, t)}")
                grand = [g + p for g, p in zip(grand, (m, f, t))]
                break
            if lab not in members or lab in seen:
                raise SystemExit(f"GLMM {key}: row {lab!r} is not an expected row of {sub!r}")
            m, f, t = BLANKS.get((key, sub, lab), (m, f, t))
            if None in (m, f, t) or m + f != t:
                raise SystemExit(f"GLMM {key} {sub} {lab}: {(m, f, t)}")
            seen.add(lab)
            out[(key, sub, lab)] = (m, f)
            acc = [a + b for a, b in zip(acc, (m, f, t))]
    if final:
        lab, m, f, t = rows[i]
        i += 1
        if lab != final or [m, f, t] != grand:
            raise SystemExit(f"GLMM {key}: final row {lab!r} {(m, f, t)} against subtotals {grand}")
    if i != len(rows):
        raise SystemExit(f"GLMM {key}: {len(rows) - i} rows after the total")
    return out, tuple(grand)


def region_table():
    rows = glmm_rows("region", ("Saudis", "Non-Saudis", "Total"))
    by_label = {lab: (s, n, t) for lab, s, n, t in rows}
    if by_label.pop("Total") != (CITIZENS, NON_SAUDIS, TOTAL) or len(by_label) != 13:
        raise SystemExit("GLMM region table: national row or row count is not the census's")
    out = {}
    for pc, (name, _cod) in REGIONS.items():
        s, n, t = by_label[CENSUS[name][0]]
        if s + n != t or (s, n) != REGION_2022[pc]:
            raise SystemExit(f"GLMM region {name}: {(s, n, t)} against sa_geo's {REGION_2022[pc]}")
        out[pc] = (s, n)
    if sum(s for s, _n in out.values()) != CITIZENS or sum(n for _s, n in out.values()) != NON_SAUDIS:
        raise SystemExit("GLMM region rows do not sum to the census")
    print(f"GLMM region table: 13 regions, {CITIZENS:,} Saudis and {NON_SAUDIS:,} non-Saudis, equal "
          "to sa_geo.REGION_2022")
    return out


def load_report():
    import fitz

    with open(REPORT, "rb") as fh:
        data = fh.read()
    if len(data) != REPORT_SIZE or b"%%EOF" not in data[-2048:]:
        raise SystemExit(f"{REPORT} is {len(data):,} bytes, pinned {REPORT_SIZE:,}, or has no %%EOF")
    doc = fitz.open(REPORT)
    if len(doc) != 17:
        raise SystemExit(f"the report has {len(doc)} pages, expected 17")
    return [[ln.strip() for ln in pg.get_text().splitlines() if ln.strip()] for pg in doc]


def report_checks(pages, regions, tables, selected):
    flat = lambda p: re.sub(r"\s+", " ", " ".join(pages[p - 1]))
    if ("The Saudi population in Saudi Arabia is 18,792,262 (58.4%), while the Non-Saudi population "
            "is 13,382,962 (41.6%).") not in flat(2):
        raise SystemExit("the report's key facts do not carry the census totals")

    # Figure 11: each region's non-Saudi share, printed to one decimal
    lines = pages[13]
    worst = 0.0
    for pc, (s, n) in regions.items():
        lab = CENSUS[REGIONS[pc][0]][1]
        k = lines.index(lab)
        printed = float(lines[k + 1].rstrip("%"))
        worst = max(worst, abs(100.0 * n / (s + n) - printed))
    if worst > 0.051:
        raise SystemExit(f"Figure 11's non-Saudi shares are up to {worst:.3f} points off GLMM's table")
    print(f"  report Figure 11: all 13 regions' non-Saudi shares equal GLMM's counts to {worst:.3f} "
          "points")

    # Figure 12: the bars' values are in the text layer, and each overall bar follows from the other two
    tokens = set(pages[14])
    missing = [v for (_g, _l, bars) in CENSUS.values() for v in bars[1:] if str(v) not in tokens]
    if missing:
        raise SystemExit(f"Figure 12 values not in page 15's text: {missing}")
    for pc, (s, n) in regions.items():
        name = REGIONS[pc][0]
        ov, rs, rn = CENSUS[name][2]
        males = s * rs / (100 + rs) + n * rn / (100 + rn)
        implied = 100 * males / (s + n - males)
        if abs(implied - ov) > OVERALL_TOL:
            raise SystemExit(f"Figure 12 {name}: overall bar {ov}, the other two give {implied:.1f}")
    print("  report Figure 12: every bar value is on the page, and each region's overall bar is what "
          f"its Saudi and non-Saudi bars give, within {OVERALL_TOL}")

    # the 34 nationalities and the continent prose, page 6
    t6 = flat(6)
    toks6 = set(pages[5])
    absent = [lab for lab, (_m, _f, t) in selected.items() if f"{t:,}" not in toks6]
    if absent:
        raise SystemExit(f"the report's Figure 3 does not print these totals: {absent}")
    for phrase in CONTINENT_PROSE:
        if phrase not in t6:
            raise SystemExit(f"the report's continent prose is not {phrase!r}")
    print("  report Figure 3: the 34 nationalities' totals are GLMM's; continent prose found")


def continent_check(tables, rest):
    rows = {k: v for t in tables.values() for k, v in t[0].items()}
    tot = lambda keys, j: sum(rows[k][j] for k in keys)
    arab_asia = [k for k in rows if k[0] == "arab" and k[2] in ARAB_IN_ASIA]
    arab_africa = [k for k in rows if k[0] == "arab" and k[2] not in ARAB_IN_ASIA]
    asia = [k for k in rows if k[0] == "asian"] + arab_asia
    africa = [k for k in rows if k[0] == "ssa"] + arab_africa
    europe = [k for k in rows if k[0] == "europe"]
    n_all = lambda keys: tot(keys, 0) + tot(keys, 1)
    got = [(100 * n_all(asia) / NON_SAUDIS, 76.2, 0.06), (100 * n_all(africa) / NON_SAUDIS, 23.2, 0.06),
           (100 * sum(rest) / NON_SAUDIS, 0.3, 0.15),
           (100 * tot(asia, 1) / NON_SAUDI_F, 67.5, 0.06), (100 * tot(africa, 1) / NON_SAUDI_F, 31.4, 0.06),
           (100 * tot(europe, 1) / NON_SAUDI_F, 0.5, 0.06), (100 * rest[1] / NON_SAUDI_F, 0.5, 0.15)]
    print("  continents (computed, printed): " + ", ".join(f"{g:.2f} {p}" for g, p, _t in got))
    bad = [(round(g, 2), p) for g, p, t in got if abs(g - p) > t]
    if bad:
        raise SystemExit(f"the four tables do not reproduce the report's continent shares: {bad}")


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
    if pn is None:
        row = origin.REGIONAL[iso]
    else:
        if pn not in pew.index:
            raise SystemExit(f"Pew has no row {pn!r} for {iso}")
        row = {f: float(pew.loc[pn, f]) for f in origin.FAMILIES}
    return fold(origin.composition(iso, row, OTHER_NODE))


def pew_share(pew, family):
    return float(pew.loc["Saudi Arabia", family]) / float(pew.loc["Saudi Arabia", "Population"])


def main():
    if "--fetch" in sys.argv or not os.path.exists(REPORT) or not os.path.exists(RCRC):
        fetch()
    regions = region_table()
    tables = {k: group_table(k) for k in TABLES}
    for k, (_rows, grand) in tables.items():
        print(f"  GLMM {k}: {grand[2]:,} non-Saudis ({grand[0]:,} men, {grand[1]:,} women)")
    # The remainder is taken off the tables' ROWS, not their printed subtotals, so that the rows and
    # the remainder partition the census's non-Saudi men and women exactly; the rounded Arab and
    # Sub-Saharan rows and South Asia's blank cell make the two differ by a person or two.
    rows = {k: v for t in tables.values() for k, v in t[0].items()}
    rest = (NON_SAUDI_M - sum(m for m, _f in rows.values()),
            NON_SAUDI_F - sum(f for _m, f in rows.values()))
    if min(rest) <= 0:
        raise SystemExit(f"the four tables hold more non-Saudis than the census: {rest}")
    print(f"  in none of the four tables: {sum(rest):,} ({rest[0]:,} men, {rest[1]:,} women)")
    continent_check(tables, rest)
    selected = {lab: (m, f, t) for lab, m, f, t in glmm_rows("selected", ("Males", "Females", "Total"))
                if not lab.startswith("Total")}
    if len(selected) != 34:
        raise SystemExit(f"GLMM selected table has {len(selected)} nationalities, expected 34")
    for lab, (m, f, t) in selected.items():
        if lab == "USA":
            if t > sum(rest):
                raise SystemExit("more US nationals than non-Saudis outside the four tables")
            continue
        hit = [k for k in rows if k[2] == SELECTED_ALIAS.get(lab, lab)]
        if len(hit) != 1:
            raise SystemExit(f"selected {lab!r} matches {hit}")
        rm, rf = rows[hit[0]]
        if abs(rm + rf - t) > SELECTED_TOL[hit[0][0]] or (m is not None and (rm, rf) != (m, f)):
            raise SystemExit(f"selected {lab}: {(m, f, t)} against {hit[0]} {(rm, rf)}")
    print("  GLMM's 34 largest nationalities agree with the four tables (sexes exactly for the nine "
          "printed)")
    report_checks(load_report(), regions, tables, selected)

    # ---- the sexes by region: Figure 12's non-Saudi ratio, raked to the national sexes ----
    with open(RCRC, encoding="utf-8") as fh:
        rc = json.load(fh)["results"]
    riy = {g: sum(r["n"] for r in rc if r["admregcen"] == "Ar Riyadh" and r["ctz"] == "Non-Saudi"
                  and r["gender"] == g) for g in ("Male", "Female")}
    nat = {g: sum(r["n"] for r in rc if r["admregcen"] == "Total (KSA Regions)" and r["ctz"] == "Non-Saudi"
                  and r["gender"] == g) for g in ("Male", "Female")}
    if (nat["Male"], nat["Female"]) != (NON_SAUDI_M, NON_SAUDI_F):
        raise SystemExit(f"RCRC's national non-Saudi sexes are {nat}")
    fem = {pc: n * 100.0 / (100 + CENSUS[REGIONS[pc][0]][2][2]) for pc, (_s, n) in regions.items()}
    k = NON_SAUDI_F / sum(fem.values())
    fem = {pc: v * k for pc, v in fem.items()}
    male = {pc: regions[pc][1] - fem[pc] for pc in regions}
    print(f"\n  non-Saudi women by region from Figure 12's ratios sum to {NON_SAUDI_F / k:,.0f}; scaled "
          f"by {k:.5f} to the census's {NON_SAUDI_F:,}")
    worst = max(abs(100 * male[pc] / fem[pc] - CENSUS[REGIONS[pc][0]][2][2]) for pc in regions)
    if worst > 1.0:
        raise SystemExit(f"after scaling a region's ratio moves {worst:.2f} from Figure 12")
    riy_pc = next(pc for pc in regions if REGIONS[pc][0] == "Riyadh")
    off = male[riy_pc] / riy["Male"] - 1
    print(f"  Riyadh region, men: {male[riy_pc]:,.0f} here against RCRC's {riy['Male']:,} ({off:+.3%}); "
          f"largest ratio shift from scaling {worst:.2f}")
    if abs(off) > 0.005:
        raise SystemExit("the sex split disagrees with RCRC's Riyadh count")

    # ---- the composition of each nationality ----
    pew = pew_table()
    iso_of = {table: {lab: iso for _sub, members in groups for lab, iso in members.items()}
              for table, (groups, _final) in TABLES.items()}
    comps = {}
    for key in rows:
        table, _sub, lab = key
        iso = iso_of[table][lab]          # None for `Other` rows and the Chagos row
        comps[key] = (composition(pew, regional=REGIONAL_ROW[table]) if iso is None
                      else composition(pew, iso=iso))
        comps[key]["_iso"] = iso
    rest_key = ("rest", "", "not in the four tables")
    rows[rest_key] = rest
    comps[rest_key] = composition(pew, regional=REST_ROWS)
    comps[rest_key]["_iso"] = None

    for key, c in comps.items():
        if c.get("_iso") == ROHINGYA:
            pew_nm = 1.0 - c.get("islam", 0.0)
            print(f"  Burma ({sum(rows[key]):,}): Pew's Myanmar row would draw {sum(rows[key]) * pew_nm:,.0f} "
                  "non-Muslims; drawn on Islam (the Rohingya)")
            comps[key] = {"islam": 1.0, "_iso": ROHINGYA}

    total_pop = sum(s + n for s, n in regions.values())
    target = pew_share(pew, "Hindus") * total_pop
    india = next(k for k, c in comps.items() if c.get("_iso") == INDIA)
    others = sum(sum(rows[k]) * c.get("hinduism", 0.0) for k, c in comps.items() if k != india)
    n_in = sum(rows[india])
    h_pew = comps[india].get("hinduism", 0.0)
    h = (target - others) / n_in
    print(f"  India ({n_in:,}): Pew 2020's Saudi Hindus are {pew_share(pew, 'Hindus'):.3%}, {target:,.0f} of "
          f"the census count; other nationalities give {others:,.0f}, so Indians are {h:.2%} Hindu "
          f"(India's own row {h_pew:.2%}, which would give {others + n_in * h_pew:,.0f})")
    if not 0.0 < h < h_pew:
        raise SystemExit("India's Hindu share from Pew's Saudi estimate is not between 0 and India's row")
    comps[india]["islam"] = comps[india].get("islam", 0.0) + (h_pew - h)
    comps[india]["hinduism"] = h

    nodes = sorted({n for c in comps.values() for n in c if n != "_iso"})
    for c in comps.values():
        s = sum(v for n, v in c.items() if n != "_iso")
        if abs(s - 1) > 1e-9:
            raise SystemExit(f"a composition sums to {s}")
    men = sum(m for m, _f in rows.values())
    women = sum(f for _m, f in rows.values())
    if (men, women) != (NON_SAUDI_M, NON_SAUDI_F):
        raise SystemExit(f"nationality rows give {men:,} men and {women:,} women")
    mix_m = {n: sum(rows[k][0] * c.get(n, 0.0) for k, c in comps.items()) / men for n in nodes}
    mix_f = {n: sum(rows[k][1] * c.get(n, 0.0) for k, c in comps.items()) / women for n in nodes}
    mix_all = {n: (mix_m[n] * men + mix_f[n] * women) / NON_SAUDIS for n in nodes}
    chr_nodes = [n for n in nodes if n.startswith("christianity")]
    print("  non-Saudi men {:.2%} Christian, women {:.2%}".format(sum(mix_m[n] for n in chr_nodes),
                                                                  sum(mix_f[n] for n in chr_nodes)))

    # ---- by region ----
    fm = pd.DataFrame({pc: {n: male[pc] * mix_m[n] + fem[pc] * mix_f[n] for n in nodes}
                       for pc in regions}).T[nodes]
    fcounts = round_within_rows(fm)
    if not all(int(fcounts.loc[pc].sum()) == regions[pc][1] for pc in regions):
        raise SystemExit("a region's rounded non-Saudi counts do not sum to its non-Saudis")
    print("\n  Christians by region, by sex (drawn) and on one national mix (not drawn):")
    for pc in sorted(regions, key=lambda p: -regions[p][1]):
        by_sex = sum(fm.loc[pc, n] for n in chr_nodes)
        flat_mix = regions[pc][1] * sum(mix_all[n] for n in chr_nodes)
        print(f"      {REGIONS[pc][0]:<18} {by_sex:>9,.0f}  {flat_mix:>9,.0f}  "
              f"({100 * by_sex / (regions[pc][0] + regions[pc][1]):.2f}% of the region)")

    chr_ = int(fcounts[chr_nodes].sum().sum())
    ratio = (chr_ / total_pop) / pew_share(pew, "Christians")
    print(f"\n  witness: {chr_:,} Christians, {chr_ / total_pop:.3%} of the census; Pew 2020 has "
          f"{pew_share(pew, 'Christians'):.3%} for everyone living in Saudi Arabia; ratio {ratio:.2f}, "
          f"band {CHRISTIAN_BAND}")
    for fam, node in (("Buddhists", "buddhism"), ("Religiously_unaffiliated", "unaffiliated"),
                      ("Jews", "judaism"), ("Hindus", "hinduism")):
        got = int(fcounts[node].sum()) if node in fcounts else 0
        print(f"      {fam}: {got:,} drawn against Pew's {pew_share(pew, fam) * total_pop:,.0f} (not asserted)")
    if not CHRISTIAN_BAND[0] <= ratio <= CHRISTIAN_BAND[1]:
        raise SystemExit("the layer's Christians are outside the band; read sources/sa.md §4")

    # ---- write ----
    out = pd.DataFrame({"geo_id": list(regions), "geo_level": "region",
                        "geo_name": [REGIONS[pc][0] for pc in regions],
                        "source_category": "Saudi citizens",
                        "count": [regions[pc][0] for pc in regions],
                        "basis": "estimate", "year": 2022, "source_id": "sa_census2022_citizens",
                        "note": "no source asks religion; every Saudi citizen is drawn on Islam "
                                "(sources/sa.py), on the 2022 census count of Saudis per region"})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    ext = fcounts.stack().rename("count").reset_index()
    ext.columns = ["geo_id", "node", "count"]
    ext = ext[ext["count"] > 0].copy()
    ext["geo_level"] = "region"
    ext["geo_name"] = ext["geo_id"].map(lambda pc: REGIONS[pc][0])
    ext["tier"] = "modelled"
    ext["basis"] = "nationality_sex_derived"
    ext["year"] = 2022
    ext["source_id"] = "census2022_nonsaudis_x_pew2020"
    ext[["geo_id", "geo_level", "geo_name", "node", "count", "tier", "basis", "year",
         "source_id"]].to_csv(OUT_FOREIGN, index=False, encoding="utf-8")

    fx = ext.groupby("node")["count"].sum().sort_values(ascending=False)
    muslim_f = int(fx.get("islam", 0))
    non_muslim = int(ext["count"].sum()) - muslim_f
    print(f"\nwrote {OUT} ({CITIZENS:,} citizens) and {OUT_FOREIGN} ({int(ext['count'].sum()):,} "
          f"non-Saudis, {ext['node'].nunique()} nodes)")
    print(f"    drawn Muslim {(CITIZENS + muslim_f) / total_pop:.3%}; non-Muslim {non_muslim:,}; Pew 2020 "
          f"for everyone: {pew_share(pew, 'Muslims'):.3%} Muslim")
    print("    non-Saudi: " + ", ".join(f"{n} {int(v):,}" for n, v in fx.items()))
    got = dict(citizens=CITIZENS, non_saudis=int(ext["count"].sum()), foreign_muslim=muslim_f,
               non_muslim=non_muslim, christians=chr_, hindus=int(fx.get("hinduism", 0)),
               buddhists=int(fx.get("buddhism", 0)), unaffiliated=int(fx.get("unaffiliated", 0)))
    print(f"\n  note_public's figures: {got}")
    if NOTE and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
