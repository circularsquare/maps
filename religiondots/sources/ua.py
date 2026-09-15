"""Ukraine — ESS rounds 2-6 (2005-2013) for the oblast pattern; round 11 (2023-24) as its witness.

Writes data/normalized/ua.csv (everyone; there is no foreign half, see below).

Usage:
    python sources/ua.py --fetch     # ESS tabulations, Ukrstat population, DESS and Razumkov witnesses
    python sources/ua.py             # rebuild from data/raw/ua/

THE OFFICE WAS CHECKED FIRST (§9cu). Ukraine's censuses (1989, 2001) did not ask religion and there
has been no census since. The State Statistics Service publishes no religion table. What the state
does publish is the State Service for Ethnic Policy and Freedom of Conscience's (DESS) annual Form 1,
registered religious ORGANISATIONS (communities, monasteries, missions) by denomination and oblast,
on data.gov.ua under CC BY. That counts congregations, not people (Latvia's KUR010, sources/lv.md §1);
it is fetched as a witness and not drawn. sources/ua.md §1.

WHICH ROUNDS. Ukraine is in ESS rounds 2-6 and 11, and in none of 7-10 (probed 2026-09-14).

    2   2005     `regionua` (oblast names), harmonised card `rlgdnm` only
    3   2006-07  `regionua`, `rlgdnm` only
    4   2009     `regionua`, `rlgdnm` + the Ukrainian card `rlgdnua` (MP / KP / UAOC / Greek Catholic ...)
    5   2011     `region` (UA11-UA83), `rlgdnm` + `rlgdnua`
    6   2013     `region`, `rlgdnm` + `rlgdnua`
   11   2023-24  `region`, `rlgdnm` + `rlgdnaua` (OCU / MP / "no patriarchate" ...); Crimea, Donetsk and
                 Luhansk not sampled

THE PATTERN IS ROUNDS 2-6 AND ROUND 11 IS NOT POOLED. Three reasons, in order of weight. Round 11
does not sample three of the 26 survey units (and never could: they are occupied), so a pool across
the gap would draw those three from 2005-2013 and everything else from a 2005-2023 average, a vintage
step along the front line. Round 11 places respondents where they live in 2023, after millions were
displaced. And its country card is a different card. It is used instead as an out-of-sample witness
for the oblast ordering and to check the level (§3.4 applies only if a category moved more than 3.5
points on the oblasts both sample).

THE CARD IS THE HARMONISED ONE, `rlgdnm`, because it is the only card in all five pooled rounds. So
the Orthodox answer is ONE answer, drawn at `christianity.orthodox`. The Ukrainian card in rounds 4-6
splits it into the Moscow and Kyiv patriarchates and the Autocephalous Church, and round 11's card
into the OCU, the Moscow Patriarchate and "no patriarchate", and the two cannot be pooled: the
churches merged in 2018 and self-identification moved by tens of points after 2014 and 2022. build()
prints both jurisdiction splits; the map does not draw either. That is a §14 call and is in
ask/ (sources/ua.md §7). `rlgdnm`'s `Roman Catholic` holds the Greek Catholics (round 4: 142 of 146),
so it is split into Eastern and Latin on the country cards of rounds 4, 5, 6 and 11.

EVERYONE, NOT CITIZENS. Ukraine has no census citizenship table since 2001 to build a foreign half on,
and ESS's non-citizens are 0.5% of the sample. So the survey is drawn for all residents.

THE POPULATION IS UKRSTAT'S LAST ESTIMATE BEFORE THE FULL-SCALE INVASION: present population at 1
January 2022 for the 24 oblasts and Kyiv city (Donetsk and Luhansk as whole oblasts), and at 1 January
2014 for Crimea and Sevastopol, which Ukrstat has not estimated since. The boundaries are COD-AB
Ukraine's 27 first-level units (sources/ua_geo.py). Sevastopol is never named by ESS and takes
Crimea's composition.
"""

import argparse
import json
import os
import re
import ssl
import sys
import unicodedata
import urllib.error
import urllib.request

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
sys.path.insert(0, HERE)
RAW = os.path.join(ROOT, "data", "raw", "ua")
OUT = os.path.join(ROOT, "data", "normalized", "ua.csv")

UA_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}
BROWSER = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
}

# THE STATISTIC IS IMPORTED, NOT COPIED: Norway's `_stability` and `sources/stability.py`'s
# halvings (every halving of an odd round count; per-round permutation null; Sweden's chi-square
# gate).
import be  # noqa: E402,F401
import no as _no  # noqa: E402
import stability  # noqa: E402

# --- ESS --------------------------------------------------------------------------------
ESS_API = _no.ESS_API
ESS_FILES = {          # main integrated files, from the ESS series' study list (search.seriesMetadata)
    2: ("edee45f2-976b-4c8b-902d-b65dc003c92e", 59),    # ESS2e03_6
    3: ("89f49986-51f5-47e1-b3e1-e0e45f168415", 65),    # ESS3e03_7
    4: ("99ba8b91-a921-4a2a-9436-52c536d7ec9d", 70),    # ESS4e04_6
    5: ("0189b86b-8aa4-4be3-88ad-39c58b02f19f", 89),    # ESS5e03_6
    6: ("450fa78e-68ab-493f-b169-dbc7ab8ffec2", 85),    # ESS6e02_7
    11: ("242aaa39-3bbb-40f5-98bf-bfb1ce53d8ef", 179),  # ESS11e04_2
}
POOL_ROUNDS = [2, 3, 4, 5, 6]
LATE_ROUND = 11
CARD_ROUNDS = [4, 5, 6, 11]
ESS_YEARS = {2: "2005", 3: "2006-07", 4: "2009", 5: "2011", 6: "2013", 11: "2023-24"}

REGION_VAR = {2: "regionua", 3: "regionua", 4: "regionua", 5: "region", 6: "region", 11: "region"}
CARD_VAR = {4: "rlgdnua", 5: "rlgdnua", 6: "rlgdnua", 11: "rlgdnaua"}
LANG_CANDIDATES = ("lnghoma", "lnghom1")
YEAR_CANDIDATES = ("inwyr", "inwyys", "inwyye")
WEIGHT_CANDIDATES = ("pspwght", "dweight")

# --- the witnesses and the population, fetched so the evidence regenerates -----------------
UKRSTAT_2022 = ("http://db.ukrcensus.gov.ua/PXWEB2007/ukr/publ_new1/2022/"
                "zb_%D0%A1huselnist.pdf")
# ukrstat.gov.ua serves an expired certificate; the Wayback copy of the 2014 release is whole.
UKRSTAT_2014 = ("https://web.archive.org/web/2015id_/http://www.ukrstat.gov.ua/operativ/operativ2014/"
                "ds/kn/kn_u/kn0114_u.html")
DESS_2024 = ("https://data.gov.ua/dataset/5f62ea97-4248-4916-99ae-41aa1df54c44/resource/"
             "40be19d0-9fc3-4891-a73b-84215183d54e/download/forma1_zagalna_na_01-01-2024.xlsx")
RAZUMKOV_2025 = ("https://razumkov.org.ua/en/research-areas/surveys/religiosity-confessional-"
                 "affiliation-and-inter-church-relations-in-ukrainian-society-november-2025")

# --- the geography ------------------------------------------------------------------------
# COD-AB Ukraine ADM1 pcodes (KOATUU order) and names. Latin names as COD-AB spells them.
UNIT_NAMES = {
    "UA01": "Autonomous Republic of Crimea", "UA05": "Vinnytska", "UA07": "Volynska",
    "UA12": "Dnipropetrovska", "UA14": "Donetska", "UA18": "Zhytomyrska", "UA21": "Zakarpatska",
    "UA23": "Zaporizka", "UA26": "Ivano-Frankivska", "UA32": "Kyivska", "UA35": "Kirovohradska",
    "UA44": "Luhanska", "UA46": "Lvivska", "UA48": "Mykolaivska", "UA51": "Odeska",
    "UA53": "Poltavska", "UA56": "Rivnenska", "UA59": "Sumska", "UA61": "Ternopilska",
    "UA63": "Kharkivska", "UA65": "Khersonska", "UA68": "Khmelnytska", "UA71": "Cherkaska",
    "UA73": "Chernivetska", "UA74": "Chernihivska", "UA80": "Kyiv", "UA85": "Sevastopol",
}
UNITS = sorted(UNIT_NAMES)
CRIMEA, SEVASTOPOL, KYIV_CITY = "UA01", "UA85", "UA80"
SURVEY_UNITS = [u for u in UNITS if u != SEVASTOPOL]        # 26: ESS never names Sevastopol

# ESS labels (Latin in rounds 2-4, Ukrainian in 5, 6 and 11) -> pcode, by stem, asserted unique.
UNIT_STEMS = {
    "UA01": ("crimea", "крим"), "UA05": ("vynnyts", "vinnyts", "вінниц"), "UA07": ("volyn", "волин"),
    "UA12": ("dnipro", "дніпро"), "UA14": ("donets", "донец"), "UA18": ("zhytom", "житомир"),
    "UA21": ("zakarpat", "закарпат"), "UA23": ("zaporiz", "запоріз"), "UA26": ("ivano", "івано"),
    "UA32": ("kyivska", "київська"), "UA35": ("kirovo", "кіровогр"),
    "UA44": ("lugan", "luhan", "луган"), "UA46": ("lvivs", "львів"), "UA48": ("mykola", "микола"),
    "UA51": ("odes", "одес"), "UA53": ("poltav", "полтав"), "UA56": ("riven", "рівнен"),
    "UA59": ("sumsk", "сумськ"), "UA61": ("ternop", "терноп"), "UA63": ("kharkiv", "харків"),
    "UA65": ("kherson", "херсон"), "UA68": ("khmel", "хмельн"), "UA71": ("cherkas", "черкас"),
    "UA73": ("chernov", "chernivt", "чернівец"), "UA74": ("chernig", "chernih", "чернігів"),
    "UA80": ("kyiv city", "м. київ", "м.київ"), "UA85": ("sevastop", "севастоп"),
}

# Oblasts each round did not sample (probed 2026-09-14), asserted.
EXPECT_ABSENT = {
    2: {"UA05", "UA61"},
    3: {"UA61", "UA68", "UA71", "UA73"},
    4: {"UA32", "UA53", "UA68"},
    5: {"UA32", "UA53", "UA68"},
    6: set(),
    11: {CRIMEA, "UA14", "UA44"},
}
# The split-half needs every unit in both halves of every halving; these are the oblasts sampled in
# all five pooled rounds. The shares themselves are drawn for all 26.
TEST_UNITS = sorted(set(SURVEY_UNITS) - set().union(*(EXPECT_ABSENT[r] for r in POOL_ROUNDS)))

# THE COARSE LEVEL: the eight groups ESS's own region codes carry in rounds 5, 6 and 11 (the first
# digit of UA11-UA83, Ukraine's draft NUTS 1). All eight are sampled in every pooled round.
MACRO_OF = {
    "UA53": "M1", "UA59": "M1", "UA63": "M1", "UA74": "M1",
    "UA14": "M2", "UA44": "M2",
    "UA12": "M3", "UA23": "M3", "UA35": "M3",
    "UA48": "M4", "UA51": "M4", "UA65": "M4", "UA01": "M4", "UA85": "M4",
    "UA05": "M5", "UA61": "M5", "UA68": "M5",
    "UA32": "M6", "UA71": "M6", "UA80": "M6",
    "UA21": "M7", "UA26": "M7", "UA46": "M7", "UA73": "M7",
    "UA07": "M8", "UA18": "M8", "UA56": "M8",
}
MACRO_NAME = {
    "M1": "Poltava, Sumy, Kharkiv, Chernihiv", "M2": "Donetsk, Luhansk",
    "M3": "Dnipropetrovsk, Zaporizhzhia, Kirovohrad", "M4": "Odesa, Mykolaiv, Kherson, Crimea",
    "M5": "Vinnytsia, Ternopil, Khmelnytskyi", "M6": "Kyiv city, Kyiv oblast, Cherkasy",
    "M7": "Lviv, Ivano-Frankivsk, Zakarpattia, Chernivtsi", "M8": "Volyn, Rivne, Zhytomyr",
}
assert set(MACRO_OF) == set(UNITS)

# Razumkov Centre's four regions (its November 2025 release, footnote), for the witness only.
RAZUMKOV_REGION = {
    "West": ["UA07", "UA21", "UA26", "UA46", "UA56", "UA61", "UA73"],
    "Centre": ["UA32", "UA05", "UA18", "UA35", "UA53", "UA59", "UA68", "UA71", "UA74", "UA80"],
    "South": ["UA48", "UA51", "UA65"],
    "East": ["UA12", "UA14", "UA23", "UA63", "UA44"],
}
# "Please tell me, what religion you belong to?", 22-29 November 2025, n=2,009, government-controlled
# territory free of hostilities, respondents placed by their region before 24 February 2022. Percent.
RAZUMKOV_2025_TABLE = {                     # Ukraine, West, Centre, South, East
    "Orthodoxy": (58.3, 44.9, 67.4, 60.6, 55.8),
    "Greek Catholicism": (11.8, 39.8, 2.5, 0.4, 1.9),
    "Roman Catholicism": (1.2, 2.7, 0.8, 0.0, 0.7),
    "Protestant and Evangelical churches": (2.6, 0.6, 3.2, 2.5, 3.8),
    "Just Christian": (10.1, 5.0, 11.3, 22.8, 6.9),
    "I do not identify myself with any religious denomination": (15.5, 6.9, 14.4, 12.9, 30.0),
}
RAZUMKOV_2025_ORTHODOX = {                  # "Which Orthodox Church do you belong to?", all respondents
    "Orthodox Church of Ukraine": (42.1, 30.4, 52.8, 40.0, 36.6),
    "Ukrainian Orthodox Church (Moscow Patriarchate)": (5.4, 9.6, 4.3, 2.9, 4.0),
    "just Orthodox": (10.2, 4.2, 9.9, 17.5, 13.9),
}

# Present population. 1 January 2022: Ukrstat, "Чисельність наявного населення України на 1 січня 2022
# року", table 1, which excludes Crimea and Sevastopol. 1 January 2014: Ukrstat express release
# kn0114, in thousands.
POP_2022 = {
    "UA05": 1_509_515, "UA07": 1_021_356, "UA12": 3_096_485, "UA14": 4_059_372, "UA18": 1_179_032,
    "UA21": 1_244_476, "UA23": 1_638_462, "UA26": 1_351_822, "UA32": 1_795_079, "UA35": 903_712,
    "UA44": 2_102_921, "UA46": 2_478_133, "UA48": 1_091_821, "UA51": 2_351_392, "UA53": 1_352_283,
    "UA56": 1_141_784, "UA59": 1_035_772, "UA61": 1_021_713, "UA63": 2_598_961, "UA65": 1_001_598,
    "UA68": 1_228_829, "UA71": 1_160_744, "UA73": 890_457, "UA74": 959_315, "UA80": 2_952_301,
}
POP_2022_TOTAL = 41_167_335
POP_2014 = {"UA01": 1_967_200, "UA85": 385_900}
POP_2014_PRINTED = {"UA01": "1967,2", "UA85": "385,9"}
POP = {**POP_2022, **POP_2014}
assert set(POP) == set(UNITS) and sum(POP_2022.values()) == POP_2022_TOTAL

# --- the categories -----------------------------------------------------------------------
NO_RELIGION = "No religion"
REFUSAL = "__refused__"
ORTHODOX = "Eastern Orthodox"
CATHOLIC = "Roman Catholic"                 # rlgdnm's label; it holds the Greek Catholics
GREEK_CATHOLIC = "Greek-Catholic church"
LATIN_CATHOLIC = "(Other) Roman Catholic denominations"

# Every country-card answer and the rlgdnm answers it may sit in. `_check_card` asserts it.
CARD_NEST = {
    "Orthodox church of the Moscow patriarchate": {ORTHODOX},
    "Ukrainian orthodox church (Kyiv patriarchate)": {ORTHODOX},
    "Ukrainian autocephalous church": {ORTHODOX},
    "Other eastern orthodox churches (denominations)": {ORTHODOX},
    "Orthodox Church of Ukraine (Primate, Metropolitan Epifaniy)": {ORTHODOX},
    "Orthodox Church of the Moscow Patriarchate (Primate, Metropolitan Onufriy)": {ORTHODOX},
    "Orthodox (I do not belong to any patriarchate)": {ORTHODOX},
    GREEK_CATHOLIC: {CATHOLIC},
    LATIN_CATHOLIC: {CATHOLIC},
    "Roman Catholic Church": {CATHOLIC},
    "Protestant church": {"Protestant"},
    "Other Christian denominations": {"Other Christian denomination"},
    "Other Christian denominations (Write in)": {"Other Christian denomination"},
    "Muslim religion": {"Islam"},
    "Jewish religion": {"Jewish"},
    "Eastern Religions": {"Eastern religions"},
    "Other non Christian religions": {"Other Non-Christian religions"},
    "Other non-Christian religions (Write in)": {"Other Non-Christian religions"},
    # Round 11's write-in: ESS coded the harmonised answer from what was written, so it lands in
    # several (3 respondents in `Protestant`).
    "Other (Write in)": {"Other Christian denomination", "Other Non-Christian religions",
                         "Eastern religions", "Protestant"},
}
EASTERN_CARD = {GREEK_CATHOLIC}
LATIN_CARD = {LATIN_CATHOLIC, "Roman Catholic Church"}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# Asserted totals; None prints instead.
N_POOL = 9_641                 # unweighted answered respondents with a region, rounds 2-6

# The big category that closes the partition stays the residual (spec §12, nested units).
KEEP_AS_RESIDUAL = {ORTHODOX}

# What the tests select (before the cluster refusal below), asserted.
EXPECT_OBLAST_PASS = {ORTHODOX, NO_RELIGION, CATHOLIC, "Islam"}
EXPECT_MACRO_PASS = {NO_RELIGION, CATHOLIC, "Eastern religions"}

# spec §12 (Uzbekistan): a pass carried by one sampling cell is refused. The share of an answer's
# unweighted respondents in its largest (round, oblast) cell above which a pass is refused.
CLUSTER_REFUSE = 0.5

# Categories drawn at the oblast against the rank test, with the reason (sources/gt.py's convention).
OVERRIDE = {}

# The Catholic answer is split Eastern/Latin at the oblast where the card rounds give it at least this
# many Catholic respondents, else at its macro-region, else nationally.
CATH_SPLIT_MIN = 10

# §3.4 IS NOT APPLIED, AND THAT IS A DEPARTURE FROM NORWAY AND LATVIA WITH A REASON. Round 11 puts No
# religion 6.3 points above the pool on the 23 oblasts it sampled (Orthodox -3.4, Catholic -1.2), past
# the 3.5-point bar. But the only measurement of the change excludes Crimea, Sevastopol, Donetsk and
# Luhansk. Scaling the 23 alone draws a vintage step along the 2014 line of occupation; scaling all 27
# asserts a change in occupied territory that nobody measured. Neither is a call to make alone, so the
# map draws 2005-2013's level everywhere, `how` says so, and the question is in the §14 ask
# (sources/ua.md §6). ask 015 (open) is the general version for LAPOP countries.
RESCALE_TO_LATE = False
DRIFT_BAR = 0.035


def _key(s):
    return " ".join(str(s).split())


def _fold(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("’", "'")
    return " ".join(s.split())


def _p(name):
    return os.path.join(RAW, name)


_UNIT_MEMO = {}


def _unit_of(label):
    if label in _UNIT_MEMO:
        return _UNIT_MEMO[label]
    f = _fold(label)
    hits = [u for u, stems in UNIT_STEMS.items() if any(st in f for st in stems)]
    if len(hits) != 1:
        sys.exit(f"!! region label {label!r} matches {hits}, not exactly one oblast")
    _UNIT_MEMO[label] = hits[0]
    return hits[0]


# =======================================================================================
# fetch
# =======================================================================================

def _ess_try(rnd, bv, weight):
    """One tabulation for Ukraine, or None when a variable is not in this round (HTTP 400 E201)."""
    fid, ver = ESS_FILES[rnd]
    q = _no._TAB % (f' weightVariable:"{weight}",' if weight else "")
    body = json.dumps({"query": q, "variables": {"id": fid, "v": ver, "bv": bv}}).encode()
    req = urllib.request.Request(ESS_API, data=body, headers={
        "Content-Type": "application/json", **UA_HDR})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        if e.code == 400 and "E201VariableNotFound" in txt:
            return None
        sys.exit(f"!! ESS round {rnd} {bv}: HTTP {e.code} {txt[:300]}")
    if "errors" in r:
        if "E201VariableNotFound" in json.dumps(r["errors"]):
            return None
        sys.exit(f"!! ESS round {rnd} {bv}: {r['errors'][0].get('message')}")
    hit = [x for x in r["data"]["analysis"]["frequencyTabulationByVariables"]["responses"]
           if x["by"][0]["value"] == "UA"]
    if not hit:
        sys.exit(f"!! ESS round {rnd} has no UA response for {bv}")
    return hit[0]["response"]


def _get(rnd, bv, weight, name):
    dest = _p(name)
    if os.path.exists(dest):
        return True
    resp = _ess_try(rnd, bv, weight)
    if resp is None:
        return False
    _no._save(resp, dest)
    return True


def _download(url, dest, check):
    if os.path.exists(dest):
        print(f"  {os.path.basename(dest)}: already on disk")
        return
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(urllib.request.Request(url, headers=BROWSER), timeout=300,
                                context=ctx) as r:
        data = r.read()
    if not check(data):
        sys.exit(f"!! {url}: the download does not look like the file ({len(data):,} bytes)")
    with open(dest + ".tmp", "wb") as f:
        f.write(data)
    os.replace(dest + ".tmp", dest)
    print(f"  {os.path.basename(dest)}: {len(data):,} bytes")


def _meta():
    path = _p("ess_meta.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    meta = _meta()
    print("ESS…")
    for rnd in sorted(ESS_FILES):
        m = meta.setdefault(str(rnd), {})
        reg = REGION_VAR[rnd]
        main = [reg, "ctzcntr", "rlgblg", "rlgdnm"]
        if not _get(rnd, main, None, f"ess_r{rnd}_n.json"):
            sys.exit(f"!! round {rnd}: {main} not all present")
        if "weight" not in m:
            for w in WEIGHT_CANDIDATES:
                if _get(rnd, main, w, f"ess_r{rnd}_w.json"):
                    m["weight"] = w
                    break
            else:
                sys.exit(f"!! round {rnd}: none of {WEIGHT_CANDIDATES}")
        if rnd in CARD_VAR:
            card = [reg, "rlgblg", CARD_VAR[rnd]]
            _get(rnd, card, None, f"ess_r{rnd}_card_n.json") or sys.exit(f"!! {rnd} {card}")
            _get(rnd, card, m["weight"], f"ess_r{rnd}_card_w.json") or sys.exit(f"!! {rnd} {card}")
            _get(rnd, [CARD_VAR[rnd], "rlgdnm"], None, f"ess_r{rnd}_nest.json") or sys.exit("!! nest")
        _get(rnd, [reg, "domicil"], None, f"ess_r{rnd}_domicil.json") or sys.exit(f"!! {rnd} domicil")
        if "lang" not in m:
            for v in LANG_CANDIDATES:
                if _get(rnd, [reg, v], None, f"ess_r{rnd}_lang.json"):
                    m["lang"] = v
                    break
            else:
                m["lang"] = None
        if "year" not in m:
            m["year"] = None
            for v in YEAR_CANDIDATES:
                if _get(rnd, [v], None, f"ess_r{rnd}_year.json"):
                    m["year"] = v
                    break
        n = sum(c["count"] for c in json.load(open(_p(f"ess_r{rnd}_n.json"), encoding="utf-8"))["table"])
        print(f"  round {rnd}: {n:,.0f} respondents; weight {m['weight']}, language {m['lang']}, "
              f"year {m['year']}")
        _no._save(meta, _p("ess_meta.json"))

    print("Ukrstat population, DESS Form 1, Razumkov…")
    _download(UKRSTAT_2022, _p("ukrstat_present_population_2022-01-01.pdf"),
              lambda b: b[:5] == b"%PDF-" and b"%%EOF" in b[-2048:])
    _download(UKRSTAT_2014, _p("ukrstat_kn0114_2014-01-01.html"),
              lambda b: b"kn0114" in b or "Крим".encode("cp1251") in b)
    _download(DESS_2024, _p("dess_form1_2024-01-01.xlsx"), lambda b: b[:2] == b"PK")
    _download(RAZUMKOV_2025, _p("razumkov_2025-11.html"),
              lambda b: b"Greek Catholicism" in b)


# =======================================================================================
# reading
# =======================================================================================

def _read(name, rnd):
    """One saved response: region -> `unit` (pcode) by label, card -> `card`, language -> `lang`,
    everything else as a LABEL. `path` indexes into codeList (fi.py's trap)."""
    lang = _meta().get(str(rnd), {}).get("lang")
    d = json.load(open(_p(name), encoding="utf-8"))
    order = [v["name"] for v in d["variableValues"]]
    codes = {v["name"]: v["codeList"] for v in d["variableValues"]}
    rows = []
    for cell in d["table"]:
        if not cell["count"]:
            continue
        rec = {"round": rnd, "count": float(cell["count"])}
        for i, n in enumerate(order):
            c = codes[n][cell["path"][i]]
            if n == REGION_VAR[rnd]:
                rec["region_code"] = str(c["value"])
                rec["region_label"] = _key(c["label"])
                rec["region_miss"] = bool(c["isMissing"])
            else:
                k = "card" if n == CARD_VAR.get(rnd) else ("lang" if n == lang else n)
                rec[k] = _key(c["label"])
                rec[k + "_miss"] = bool(c["isMissing"])
        rows.append(rec)
    df = pd.DataFrame(rows)
    if "region_label" in df.columns:
        df = df[~df["region_miss"]].copy()
        df["unit"] = df["region_label"].map(_unit_of)
    return df


def _cat(df, col="rlgdnm"):
    return np.where(df["rlgblg"] == "No", NO_RELIGION,
                    np.where((df["rlgblg"] != "Yes") | df[col + "_miss"], REFUSAL, df[col]))


def _pool(rounds, tag):
    df = pd.concat([_read(f"ess_r{r}_{tag}.json", r) for r in rounds], ignore_index=True)
    df["cat"] = _cat(df)
    return df


def _card_pool(rounds, tag):
    df = pd.concat([_read(f"ess_r{r}_card_{tag}.json", r) for r in rounds], ignore_index=True)
    df["cat"] = _cat(df, "card")
    return df


# =======================================================================================
# checks
# =======================================================================================

def _check_regions():
    for rnd in sorted(ESS_FILES):
        d = _read(f"ess_r{rnd}_n.json", rnd)
        present = set(d["unit"])
        want = set(SURVEY_UNITS) - EXPECT_ABSENT[rnd]
        if present != want:
            sys.exit(f"!! round {rnd}: oblasts {sorted(present ^ want)} differ from EXPECT_ABSENT")
        if REGION_VAR[rnd] == "region":
            for code, u in set(zip(d["region_code"], d["unit"])):
                if "M" + code[2] != MACRO_OF[u]:
                    sys.exit(f"!! round {rnd}: {code} is {u}, which MACRO_OF puts in {MACRO_OF[u]}")
        print(f"  round {rnd} ({ESS_YEARS[rnd]}): {len(present)} of 26 survey units by label; "
              f"not sampled: {', '.join(UNIT_NAMES[u] for u in sorted(EXPECT_ABSENT[rnd])) or 'none'}")
    print(f"  the split-half runs on the {len(TEST_UNITS)} oblasts sampled in all of rounds "
          f"{POOL_ROUNDS}; macro-regions from the first digit of `region`, asserted in rounds 5, 6, 11")


def _check_labels():
    """EVERY ROUND'S OBLASTS AGAINST THE PEOPLE BEHIND THEM (Denmark's round 9, Latvia's round 10).

    Kyiv city is the most big-city; Lviv, Ternopil or Ivano-Frankivsk is the most Catholic (the
    harmonised Catholic answer is mostly Greek Catholic); in rounds 2-6 Crimea, Donetsk or Luhansk
    speaks the most Russian at home and Donetsk and Luhansk have under 2% Catholics.
    """
    from scipy.stats import spearmanr
    for rnd in sorted(ESS_FILES):
        dom = _read(f"ess_r{rnd}_domicil.json", rnd)
        dom = dom[~dom["domicil_miss"]]
        big = (dom[dom["domicil"] == "A big city"].groupby("unit")["count"].sum()
               / dom.groupby("unit")["count"].sum()).fillna(0.0)
        main = _read(f"ess_r{rnd}_n.json", rnd)
        main["cat"] = _cat(main)
        ans = main[main["cat"] != REFUSAL]
        cath = (ans[ans["cat"] == CATHOLIC].groupby("unit")["count"].sum()
                / ans.groupby("unit")["count"].sum()).reindex(sorted(set(ans["unit"]))).fillna(0.0)
        n_unit = main.groupby("unit")["count"].sum()
        pop = pd.Series({u: POP[u] + (POP[SEVASTOPOL] if u == CRIMEA else 0) for u in n_unit.index})
        rho = spearmanr(n_unit, pop.reindex(n_unit.index)).correlation
        line = (f"  round {rnd}: big city top {UNIT_NAMES[big.idxmax()]} {100 * big.max():.0f}%; "
                f"Catholic top {UNIT_NAMES[cath.idxmax()]} {100 * cath.max():.0f}%")
        ok = big.idxmax() == KYIV_CITY and cath.idxmax() in {"UA46", "UA61", "UA26"}
        lang_file = _p(f"ess_r{rnd}_lang.json")
        if os.path.exists(lang_file):
            lg = _read(f"ess_r{rnd}_lang.json", rnd)
            lg = lg[~lg["lang_miss"]]
            rus = (lg[lg["lang"].str.lower().str.contains("russian")].groupby("unit")["count"].sum()
                   / lg.groupby("unit")["count"].sum()).fillna(0.0)
            line += f"; Russian at home top {UNIT_NAMES[rus.idxmax()]} {100 * rus.max():.0f}%"
            if rnd in POOL_ROUNDS:
                ok = ok and rus.idxmax() in {CRIMEA, "UA14", "UA44"}
        if rnd in POOL_ROUNDS:
            east = cath.reindex(["UA14", "UA44"]).dropna()
            line += f"; Donetsk/Luhansk Catholic max {100 * east.max():.1f}%"
            ok = ok and (east < 0.02).all()
        print(line + f"; sample vs population Spearman {rho:+.2f}")
        if not ok:
            sys.exit(f"!! round {rnd}: the oblasts no longer look like Ukraine's; check the labels")


def _check_card():
    for rnd in CARD_ROUNDS:
        d = _read(f"ess_r{rnd}_nest.json", rnd)
        seen = set()
        for _, r in d.iterrows():
            a, b = r["card"], r["rlgdnm"]
            if r["card_miss"] or r["rlgdnm_miss"]:
                if r["card_miss"] != r["rlgdnm_miss"]:
                    sys.exit(f"!! round {rnd}: `{a}` / `{b}` missing on one card only")
                continue
            if b not in CARD_NEST.get(a, set()):
                sys.exit(f"!! round {rnd}: {CARD_VAR[rnd]} `{a}` sits in rlgdnm `{b}` for "
                         f"{r['count']:.0f}; CARD_NEST says {CARD_NEST.get(a)}")
            seen.add(a)
        print(f"  round {rnd}: {CARD_VAR[rnd]} nests in rlgdnm, {len(seen)} answers given")


# =======================================================================================
# the composition
# =======================================================================================

def _shares(df, key):
    tab = df.groupby([key, "cat"])["count"].sum().unstack(fill_value=0.0)
    return tab.div(tab.sum(axis=1), axis=0)


def _standouts(raw, cats, units, rounds):
    """Honduras's test (spec §12): for each category, how often the same unit tops both halves."""
    splits = stability.halvings(len(rounds))
    out = {}
    for c in cats:
        tops = []
        for a, b in splits:
            sides = []
            for half in (a, b):
                h = raw[raw["round"].isin([rounds[i] for i in half])]
                t = h.groupby("unit")["count"].sum()
                s = (h[h["cat"] == c].groupby("unit")["count"].sum() / t).reindex(t.index).fillna(0)
                sides.append(s)
            common = sides[0].index.intersection(sides[1].index).intersection(units)
            if not len(common) or sides[0][common].max() <= 0 or sides[1][common].max() <= 0:
                tops.append(None)
                continue
            ta, tb = sides[0][common].idxmax(), sides[1][common].idxmax()
            tops.append(ta if ta == tb else None)
        named = [t for t in tops if t]
        best = max(set(named), key=named.count) if named else None
        out[c] = (best, (named.count(best) / len(tops)) if best else 0.0, len(tops))
    return out


def _chi(raw, c, units):
    t = raw.groupby("unit")["count"].sum().reindex(units).fillna(0)
    a = raw[raw["cat"] == c].groupby("unit")["count"].sum().reindex(units).fillna(0)
    return stability.chi2_p(a.values, t.values)


def _citizen_like_shares():
    raw_all = _pool(POOL_ROUNDS, "n")
    wtd_all = _pool(POOL_ROUNDS, "w")
    answered = 1.0 - wtd_all.loc[wtd_all["cat"] == REFUSAL, "count"].sum() / wtd_all["count"].sum()
    noncit = wtd_all.loc[wtd_all["ctzcntr"] == "No", "count"].sum() / wtd_all["count"].sum()
    print(f"  {raw_all['count'].sum():,.0f} respondents with an oblast in rounds {POOL_ROUNDS}; "
          f"{100 * (1 - answered):.2f}% declined (weighted); non-citizens {100 * noncit:.2f}% (kept)")
    raw, wtd = raw_all[raw_all["cat"] != REFUSAL], wtd_all[wtd_all["cat"] != REFUSAL]
    n = int(round(raw["count"].sum()))
    if N_POOL is None:
        print(f"  !! N_POOL is unset; this build has {n:,}")
    elif n != N_POOL:
        sys.exit(f"!! {n:,} answered respondents, expected {N_POOL:,}")
    import ua2013
    unknown = sorted(set(wtd["cat"]) - ua2013.POOL_SOURCE)
    if unknown:
        sys.exit(f"!! pool categories with no mapping: {unknown}")

    nat = wtd.groupby("cat")["count"].sum()
    nat = nat / nat.sum()
    cats = sorted(nat.index, key=lambda c: -nat[c])
    print("  by round, weighted:")
    per = {r: _shares(wtd[wtd["round"] == r].assign(k=1), "k").iloc[0] for r in POOL_ROUNDS}
    print("    " + " " * 32 + "".join(f"{'r' + str(r):>9}" for r in POOL_ROUNDS) + f"{'pool':>9}")
    for c in cats:
        print(f"    {c[:30]:<32}" + "".join(f"{100 * per[r].get(c, 0):8.2f}%" for r in POOL_ROUNDS)
              + f"{100 * nat[c]:8.2f}%")

    t_raw = raw[raw["unit"].isin(TEST_UNITS)].rename(columns={"unit": "region"})
    passed_obl = _no._stability(t_raw, cats, POOL_ROUNDS, TEST_UNITS,
                                f"{len(TEST_UNITS)} oblasts sampled in every pooled round")
    m_raw = raw.assign(region=raw["unit"].map(MACRO_OF))
    passed_mac = _no._stability(m_raw, cats, POOL_ROUNDS, sorted(MACRO_NAME), "8 macro-regions")
    for label, got, want in (("oblast", passed_obl, EXPECT_OBLAST_PASS),
                             ("macro-region", passed_mac, EXPECT_MACRO_PASS)):
        if want is not None and set(got) != want:
            sys.exit(f"!! the {label} test now selects {sorted(got)}, not {sorted(want)}")

    cell = raw.groupby(["cat", "round", "unit"])["count"].sum()
    tot_c = raw.groupby("cat")["count"].sum()
    refused = set()
    print("\n  largest (round, oblast) cell of each passing answer (spec §12, Uzbekistan):")
    for c in sorted(set(passed_obl) | set(passed_mac), key=lambda x: -tot_c[x]):
        top, share = cell.loc[c].idxmax(), float(cell.loc[c].max() / tot_c[c])
        flag = share > CLUSTER_REFUSE
        print(f"    {c[:34]:<36} n {int(tot_c[c]):>5}  round {top[0]} {UNIT_NAMES[top[1]]:<30} "
              f"{100 * share:3.0f}%{'  REFUSED: one sampling cell is most of the answer' if flag else ''}")
        if flag:
            refused.add(c)
    passed_obl = [c for c in passed_obl if c not in refused]
    passed_mac = [c for c in passed_mac if c not in refused]
    for c, why in OVERRIDE.items():
        if c in passed_obl:
            sys.exit(f"!! `{c}` now passes at the oblast on its own; take it out of OVERRIDE")
        print(f"\n  OVERRIDE, drawn at the oblast against the test: `{c}`\n    {why}")

    fine_obl = [c for c in cats if (c in passed_obl or c in OVERRIDE) and c not in KEEP_AS_RESIDUAL]
    fine_mac = [c for c in cats if c in passed_mac and c not in fine_obl and c not in KEEP_AS_RESIDUAL]
    failing = [c for c in cats if c not in fine_obl and c not in fine_mac and c not in KEEP_AS_RESIDUAL]
    so = _standouts(raw, failing, SURVEY_UNITS, POOL_ROUNDS)
    standouts = {}
    print("\n  standouts among the categories that pass at neither level (spec §12, Honduras):")
    for c in failing:
        u, frac, k = so[c]
        chi = _chi(raw, c, SURVEY_UNITS)
        flag = u is not None and frac >= 0.95 and chi < 0.05
        print(f"    {c[:34]:<36} top in both halves: {UNIT_NAMES.get(u, '-'):<20} {frac:5.0%} of {k}"
              f"  chi2 p {chi:.1e}  {'STANDOUT' if flag else ''}")
        if flag:
            standouts[c] = u
    small = [c for c in cats if c not in fine_obl and c not in fine_mac and c not in standouts]

    s_obl = _shares(wtd, "unit").reindex(index=SURVEY_UNITS, columns=cats, fill_value=0.0)
    s_mac = _shares(wtd.assign(macro=wtd["unit"].map(MACRO_OF)), "macro") \
        .reindex(columns=cats, fill_value=0.0)
    comp = pd.DataFrame(0.0, index=SURVEY_UNITS, columns=cats)
    for c in fine_obl:
        comp[c] = s_obl[c]
    for c in fine_mac:
        comp[c] = [float(s_mac.loc[MACRO_OF[u], c]) for u in SURVEY_UNITS]
    for c, u0 in standouts.items():
        rest = wtd[wtd["unit"] != u0]
        comp[c] = float(rest.loc[rest["cat"] == c, "count"].sum() / rest["count"].sum())
        comp.loc[u0, c] = float(s_obl.loc[u0, c])
    fixed = fine_obl + fine_mac + list(standouts)
    residual = 1.0 - comp[fixed].sum(axis=1)
    for c in sorted(KEEP_AS_RESIDUAL):
        alt = residual - s_obl[c]
        print(f"  if `{c}` were fixed at its oblast share, the tail left would be negative in "
              f"{int((alt < -1e-9).sum())} of {len(alt)} (worst {alt.min():+.2%})")
    if (residual <= 0).any():
        sys.exit(f"!! units with no room for the tail: {sorted(residual[residual <= 0].index)}")
    tot = float(nat[small].sum())
    for c in small:
        comp[c] = residual * float(nat[c]) / tot

    # The 2x rule (spec §12): a residual category drawn at 2x its national share where nobody answered it.
    cnt = raw.groupby(["unit", "cat"])["count"].sum().unstack(fill_value=0.0).reindex(SURVEY_UNITS).fillna(0)
    worst = (0.0, None, None)
    for c in small:
        if c in KEEP_AS_RESIDUAL:
            continue
        for u in SURVEY_UNITS:
            if cnt.get(c, pd.Series(0, index=SURVEY_UNITS))[u] == 0 and nat[c] > 0:
                mult = comp.loc[u, c] / nat[c]
                if mult > worst[0]:
                    worst = (mult, c, u)
    print(f"  2x rule: worst residual multiple where the survey found none: {worst[0]:.2f}x "
          f"({worst[1]} in {UNIT_NAMES.get(worst[2], '-')})")
    if worst[0] >= 2.0:
        sys.exit("!! the 2x rule switches the small categories flat; build that construction")

    print("\n  levels: " + "; ".join([
        f"oblast {fine_obl}", f"macro-region {fine_mac}", f"standouts {standouts}",
        f"residual {small}"]))
    print("  drawn against the survey's own oblast share, residual categories (reversal check):")
    for c in small[:4]:
        d = (comp[c] - s_obl[c]).abs().sort_values(ascending=False).index[:4]
        print(f"    {c[:30]:<32}" + "  ".join(
            f"{UNIT_NAMES[u][:10]} {100 * comp.loc[u, c]:.2f}% ({100 * s_obl.loc[u, c]:.2f}%)" for u in d))
    if (comp.sum(axis=1) - 1).abs().max() > 1e-9:
        sys.exit("!! composition does not sum to 1")
    return comp, s_obl, nat, cats, fine_obl, fine_mac, standouts, small, answered, raw, wtd


def _late_check(comp, s_obl, cats):
    """Round 11 on the oblasts it sampled: the level (§3.4) and the ordering, out of sample."""
    from scipy.stats import spearmanr
    late = _pool([LATE_ROUND], "w")
    late = late[late["cat"] != REFUSAL]
    s11 = _shares(late, "unit").reindex(columns=cats, fill_value=0.0)
    units = sorted(s11.index)
    w = pd.Series({u: POP[u] for u in units})
    w = w / w.sum()
    print(f"\n  ROUND 11 ({ESS_YEARS[LATE_ROUND]}), {int(_pool([LATE_ROUND], 'n').query('cat != @REFUSAL')['count'].sum()):,} "
          f"answered, on its {len(units)} oblasts at 2022 population weights:")
    print(f"    {'category':<32}{'pool':>9}{'drawn':>9}{'r11':>9}{'drift':>9}{'rho':>8}")
    drift = {}
    for c in cats:
        p = float((s_obl.loc[units, c] * w).sum())
        dr = float((comp.loc[units, c] * w).sum())
        l11 = float((s11[c] * w).sum())
        rho = spearmanr(comp.loc[units, c], s11[c]).correlation
        drift[c] = l11 - dr
        print(f"    {c[:30]:<32}{100 * p:8.2f}%{100 * dr:8.2f}%{100 * l11:8.2f}%{100 * (l11 - dr):+8.2f}"
              f"{rho:+8.3f}")
    big = {c: d for c, d in drift.items() if abs(d) > DRIFT_BAR}
    print(f"  categories past the {100 * DRIFT_BAR:.1f}-point bar: {big or 'none'}")
    return drift, s11


def _jurisdictions():
    """Printed, not drawn: the Orthodox answer on the two Ukrainian cards."""
    print("\n  THE ORTHODOX JURISDICTIONS, PRINTED AND NOT DRAWN (sources/ua.md §7):")
    for rounds in ([4, 5, 6], [LATE_ROUND]):
        d = _card_pool(rounds, "w")
        d = d[d["cat"].map(lambda x: ORTHODOX in CARD_NEST.get(x, set()))]
        t = d.groupby("cat")["count"].sum()
        print(f"    rounds {rounds}, share of the Orthodox answer, nationally: " + ", ".join(
            f"{k} {100 * v / t.sum():.1f}%" for k, v in t.sort_values(ascending=False).items()))
        m = d.assign(macro=d["unit"].map(MACRO_OF))
        tab = m.groupby(["macro", "cat"])["count"].sum().unstack(fill_value=0)
        tab = tab.div(tab.sum(axis=1), axis=0)
        for mac in tab.index:
            print(f"      {MACRO_NAME[mac][:44]:<46}" + ", ".join(
                f"{k.split('(')[0][:26].strip()} {100 * v:.0f}%" for k, v in tab.loc[mac].items() if v > 0))


def _catholic_split():
    """Eastern share of the Catholic answer per survey unit, from the card rounds, at the finest level
    with CATH_SPLIT_MIN Catholic respondents."""
    raw = _card_pool(CARD_ROUNDS, "n")
    wtd = _card_pool(CARD_ROUNDS, "w")
    is_c = lambda s: s.isin(EASTERN_CARD | LATIN_CARD)
    raw, wtd = raw[is_c(raw["cat"])], wtd[is_c(wtd["cat"])]
    n_u = raw.groupby("unit")["count"].sum()
    e_u = wtd[wtd["cat"].isin(EASTERN_CARD)].groupby("unit")["count"].sum() / wtd.groupby("unit")["count"].sum()
    rm, wm = raw.assign(m=raw["unit"].map(MACRO_OF)), wtd.assign(m=wtd["unit"].map(MACRO_OF))
    n_m = rm.groupby("m")["count"].sum()
    e_m = wm[wm["cat"].isin(EASTERN_CARD)].groupby("m")["count"].sum() / wm.groupby("m")["count"].sum()
    e_nat = float(wtd.loc[wtd["cat"].isin(EASTERN_CARD), "count"].sum() / wtd["count"].sum())
    out, level = {}, {}
    for u in SURVEY_UNITS:
        if n_u.get(u, 0) >= CATH_SPLIT_MIN:
            out[u], level[u] = float(e_u.get(u, 0.0)), "oblast"
        elif n_m.get(MACRO_OF[u], 0) >= CATH_SPLIT_MIN:
            out[u], level[u] = float(e_m.get(MACRO_OF[u], 0.0)), "macro-region"
        else:
            out[u], level[u] = e_nat, "national"
    print(f"\n  CATHOLIC SPLIT (rounds {CARD_ROUNDS}, {int(raw['count'].sum())} Catholic respondents; "
          f"Eastern nationally {100 * e_nat:.1f}%):")
    for u in sorted(SURVEY_UNITS, key=lambda x: -n_u.get(x, 0)):
        print(f"    {UNIT_NAMES[u]:<32} n {int(n_u.get(u, 0)):>4}  Eastern {100 * out[u]:5.1f}%  [{level[u]}]")
    return out, level


def _razumkov(comp_units):
    from scipy.stats import spearmanr
    print("\n  WITNESS, Razumkov Centre November 2025 (four regions; a different question and answer list, "
          "so the ordering, not the level):")
    groups = {"Orthodoxy": [ORTHODOX], "Greek Catholicism": ["_eastern"], "Roman Catholicism": ["_latin"],
              "Protestant and Evangelical churches": ["Protestant"],
              "I do not identify myself with any religious denomination": [NO_RELIGION]}
    for rz, members in groups.items():
        drawn = []
        for reg, units in RAZUMKOV_REGION.items():
            w = pd.Series({u: POP[u] for u in units})
            drawn.append(float((comp_units.loc[units, members].sum(axis=1) * w).sum() / w.sum()))
        rzv = RAZUMKOV_2025_TABLE[rz][1:]
        rho = spearmanr(drawn, rzv).correlation
        print(f"    {rz[:40]:<42} drawn " + " ".join(f"{100 * x:5.1f}" for x in drawn)
              + "   Razumkov " + " ".join(f"{x:5.1f}" for x in rzv) + f"   rho {rho:+.2f}")


# =======================================================================================
# build
# =======================================================================================

# DESS Form 1, sheet `За областями`: one block per region, numbered rows from one template, registered
# communities (`Громади, Всього`) in column 4. Row numbers are the template's.
DESS_GROUPS = {
    "OCU": [1], "UOC": [2], "_latin": [10], "_eastern": [11, 12],
    "Protestant": list(range(16, 56)), "Islam": list(range(60, 67)), "Jewish": list(range(67, 73)),
}


def _dess_unit(title):
    f = _fold(title)
    if "києві" in f or "київ міськ" in f or "києва" in f:
        return KYIV_CITY
    if "київськ" in f:
        return "UA32"
    hits = [u for u, stems in UNIT_STEMS.items()
            if u not in (KYIV_CITY, "UA32") and any(st in f for st in stems if not st.isascii())]
    return hits[0] if len(hits) == 1 else None


def _dess_witness(cu, s_obl):
    """Registered communities per 100,000 people against the drawn shares. Congregations, not people,
    and 2024 against a 2005-2013 pattern: an ordering, not a validation (sources/ua.md §1)."""
    import openpyxl
    from scipy.stats import spearmanr
    ws = openpyxl.load_workbook(_p("dess_form1_2024-01-01.xlsx"), read_only=True,
                                data_only=True)["За областями"]
    cur, acc, titles = None, {}, []
    for row in ws.iter_rows(values_only=True):
        c0 = str(row[0] or "").strip()
        if "мережу релігійних організацій" in c0:
            cur = _dess_unit(c0)
            titles.append((c0, cur))
            continue
        m = re.fullmatch(r"(\d+)\.", c0)
        if cur and m and len(row) > 4:
            acc[(cur, int(m.group(1)))] = acc.get((cur, int(m.group(1))), 0.0) + float(row[4] or 0)
    unmatched = [t for t, u in titles if u is None]
    units = sorted({u for _, u in titles if u})
    print(f"\n  WITNESS, DESS Form 1 at 1 January 2024: {len(titles)} region blocks, {len(units)} matched"
          + (f"; unmatched {unmatched}" if unmatched else ""))
    per = {g: pd.Series({u: sum(acc.get((u, k), 0.0) for k in rows) for u in units})
           for g, rows in DESS_GROUPS.items()}
    for g in ("_eastern", "_latin", "Protestant", "Islam"):
        rate = per[g] / pd.Series({u: POP[u] for u in units}) * 1e5
        drawn = cu.loc[units, g]
        rho = spearmanr(rate, drawn).correlation
        top = rate.sort_values(ascending=False).index[:3]
        print(f"    {g.strip('_'):<11} communities {int(per[g].sum()):>6,}  Spearman with drawn share over "
              f"{len(units)}: {rho:+.3f}   most per head: " + ", ".join(
                  f"{UNIT_NAMES[u]} {rate[u]:.1f} ({100 * drawn[u]:.1f}%)" for u in top))
    per["Other Christian denomination"] = pd.Series(
        {u: sum(acc.get((u, k), 0.0) for k in range(56, 60)) for u in units})
    print("    against the SURVEY'S OWN oblast share (the evidence an OVERRIDE would need):")
    for g in ("Protestant", "Other Christian denomination", "Islam"):
        rate = per[g] / pd.Series({u: POP[u] for u in units}) * 1e5
        sv = s_obl.reindex(units)[g]
        r = spearmanr(rate, sv)
        print(f"      {g:<30} Spearman {r.correlation:+.3f} (p {r.pvalue:.4f}) over {len(units)}; survey top: "
              + ", ".join(f"{UNIT_NAMES[u]} {100 * sv[u]:.1f}% ({rate[u]:.0f})"
                          for u in sv.sort_values(ascending=False).index[:5]))
    ocu, uoc = per["OCU"], per["UOC"]
    mac = pd.DataFrame({"OCU": ocu, "UOC": uoc}).groupby(lambda u: MACRO_OF[u]).sum()
    print(f"    Orthodox communities, printed and not drawn: OCU {int(ocu.sum()):,}, UOC {int(uoc.sum()):,}; "
          "OCU share by macro-region: " + ", ".join(
              f"{k} {100 * r.OCU / (r.OCU + r.UOC):.0f}%" for k, r in mac.iterrows()))


def build():
    import ua2013

    print("checking the oblasts, their labels against the people, and the Ukrainian cards…")
    _check_regions()
    _check_labels()
    _check_card()

    print("\nESS, rounds 2-6…")
    (comp, s_obl, nat, cats, fine_obl, fine_mac, standouts, small, answered, raw,
     wtd) = _citizen_like_shares()

    drift, s11 = _late_check(comp, s_obl, cats)
    if RESCALE_TO_LATE:
        sys.exit("!! RESCALE_TO_LATE is set but the rescale is not built; see sources/ua.md §6")

    _jurisdictions()
    east, east_level = _catholic_split()

    full = comp.copy()
    full.loc[SEVASTOPOL] = comp.loc[CRIMEA]
    full = full.reindex(UNITS)
    cu = full.copy()
    cu["_eastern"] = [full.loc[u, CATHOLIC] * east[CRIMEA if u == SEVASTOPOL else u] for u in UNITS]
    cu["_latin"] = full[CATHOLIC] - cu["_eastern"]
    _razumkov(cu)
    _dess_witness(cu, s_obl)

    rows = []
    levels = {**{c: "oblast" for c in fine_obl}, **{c: "macro-region" for c in fine_mac},
              **{c: f"standout ({UNIT_NAMES[u]}), the rest at the national rate" for c, u in standouts.items()},
              **{c: "share of the oblast's residual at the national proportions (§9bi)" for c in small}}
    for u in UNITS:
        src = CRIMEA if u == SEVASTOPOL else u
        n_ans = POP[u] * answered
        for cat in cats:
            s = float(full.loc[u, cat])
            if s <= 0:
                continue
            note = (f"ESS rounds 2-6 (2005-2013), {levels[cat]}"
                    + ("; Sevastopol takes Crimea's composition" if u == SEVASTOPOL else ""))
            if cat == CATHOLIC:
                e = east[src]
                for sub, w in ((GREEK_CATHOLIC, e), (LATIN_CATHOLIC, 1.0 - e)):
                    if w > 0:
                        rows.append((u, "oblast", UNIT_NAMES[u], sub, s * w * n_ans, "self_id", 2013,
                                     "ess_r2-r6_oblast",
                                     note + f"; Eastern/Latin split from rounds 4-6 and 11 at the "
                                            f"{east_level[src]}"))
            else:
                rows.append((u, "oblast", UNIT_NAMES[u], cat, s * n_ans, "self_id", 2013,
                             "ess_r2-r6_oblast", note))
    df = pd.DataFrame(rows, columns=COLUMNS)
    unknown = sorted(set(df["source_category"]) - set(ua2013.MAP))
    if unknown:
        sys.exit(f"!! unmapped source categories: {unknown}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT + ".tmp", index=False)
    os.replace(OUT + ".tmp", OUT)

    total = sum(POP.values())
    drawn = df["count"].sum()
    print(f"\nwrote {OUT}  ({len(df):,} rows, {df['source_category'].nunique()} source categories)")
    print(f"drawn {drawn:,.0f} of {total:,.0f}, {100 * drawn / total:.2f}%")
    nodes = df.assign(node=df["source_category"].map(ua2013.resolve)).groupby("node")["count"].sum()
    for node, c in nodes.sort_values(ascending=False).items():
        print(f"  {c:>12,.0f}  {100 * c / drawn:5.2f}%  {node}")
    byu = df.assign(node=df["source_category"].map(ua2013.resolve)) \
        .groupby(["geo_id", "node"])["count"].sum().unstack(fill_value=0.0)
    byu = byu.div(byu.sum(axis=1), axis=0)
    cols = [c for c in ("christianity.orthodox", "christianity.catholic.eastern",
                        "christianity.catholic.latin", "christianity.protestant", "islam.sunni",
                        "unaffiliated") if c in byu.columns]
    print("\nper unit, drawn:")
    print(f"    {'unit':<32}" + "".join(f"{c.split('.')[-1][:10]:>11}" for c in cols))
    for u in sorted(UNITS, key=lambda x: -byu.loc[x, "unaffiliated"]):
        print(f"    {UNIT_NAMES[u]:<32}" + "".join(f"{100 * byu.loc[u, c]:10.2f}%" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    else:
        build()
