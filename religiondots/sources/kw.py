"""Kuwait: PACI's register counts religion (Muslim, Christian, other or not stated) by nationality
group and sex for the whole country; the 2021 census counts Kuwaitis and non-Kuwaitis by sex in
each of 157 areas. This joins the two.

Reads, all fetched into data/raw/kw/ by --fetch:

  * **PACI's own table builder, June 2014** (`stat.paci.gov.kw/englishreports/tableQuery`, view
    `ColumnChartEduAge`, rows `sf_relegion`, columns `sf_nationality_TD`, wafers year and sex), as
    the Wayback Machine captured its JSON answer on 15 October 2014. Muslim, Christian and
    `Other-Not Stated` for Kuwaitis and seven groups of non-Kuwaitis (Arab, Asian, African,
    European, North American, South American, Australian), each by sex: 4,039,445 people. The host
    times out from here on both ports (sources.md §11ao), so the archived answer is the only copy
    reached;
  * **2021 census** (Central Statistical Bureau, `census.csb.gov.kw`; source line "The Public
    Authority For Civil Information"), three workbooks: Table 1, population by governorate,
    nationality and sex; Table 6, by governorate, sex and nationality group; Table 52, population
    habitually residing by area, nationality (Kuwaiti / non-Kuwaiti) and sex. 4,385,717 people;
  * **GLMM's copies** (`gulfmigration.grc.net`) of PACI's December 2014 table of population by
    nationality group, sex and locality (183 localities), used only as the mix of nationality
    groups inside each area, and of PACI's mid-2018 non-Kuwaitis by country of citizenship and sex
    (the six Asian nationalities named), used only to split the Asian `Other-Not Stated`;
  * Pew Research Center, *Religious Composition 2010-2020* (data/raw/estimates/pew.zip), for that
    split only.

Writes data/normalized/kw.csv: one row per 2021 area and category, every row `derived`.
`sources/kw.md` is the record in prose.

## WHAT WAS COUNTED, AND WHAT IS CARRIED

PACI's civil-information register records a religion for everyone it holds, and its table builder
crosses that with nationality group and sex. That is a count, nationally. It is not published by
area (sources.md §11ao: no archived view pairs `sf_relegion` with `sf_governorate` or
`sf_regions`). So the religion shares of each nationality group and sex are carried to each area
by the area's own population of that group and sex. That is spec §7's `derived`: somebody counted
it, and a proxy carried it to a finer place.

The 2021 census gives each area's Kuwaitis and non-Kuwaitis by sex, but nationality groups only
per governorate (Table 6). Each area's non-Kuwaitis are split into groups by its own December 2014
mix (GLMM's copy of PACI's locality table), then raked (iterative proportional fitting, per
governorate and sex) so every area keeps its 2021 non-Kuwaiti count and every governorate its 2021
group totals. Areas with fewer than `PRIOR_MIN` non-Kuwaitis of a sex in 2014 start from their
governorate's 2021 mix.

## KUWAITIS ARE ALL DRAWN AS MUSLIM, SPLIT SHIA AND SUNNI AT ONE NATIONAL SHARE

PACI counted 1,257,977 Kuwaiti Muslims, 255 Christians and 22 other or not stated in June 2014
(99.978% Muslim). Spread by area, the 277 would be placed nowhere in particular, so every Kuwaiti
is drawn as Muslim.

**Sect, added 2026-10-03 (session fafd1067-kwsect) on Anita's ruling of that day** (`ask/RULINGS.md`:
Kuwaiti citizens may be split at one share for the whole country, "its basically just a city state
so one region fine"; a by-governorate split was not approved). The share is Arab Barometer wave III
(February-March 2014, citizens 18+, 1,021 interviews, 200 PSUs drawn by probability proportional to
size from the 2011 census frame, Kish selection, no quota: the wave's technical report), item
`q2005kw`, *"In the opinion of the field team, the respondent is a member of what sect?"*. Of the
1,018 Muslims, weighted: Sunni 66.62%, Shia 14.32%, cannot determine 19.06%. **The drawn share is
Shia of those the team could place, 17.69%** (`kuwaiti_sect`), so `cannot determine` is apportioned
at the placed ratio. That is right here and not for Iraq's `Just a Muslim`: this is the
interviewer failing to tell, not the respondent declining a sect, and the undetermined sit with the
placed mix on the items that separate the two (sources/kw.md §8). Post-stratified to the 2021
census's Kuwaitis by governorate it reads 17.67%, so the 2011 frame does not move it. Every area's
Kuwaitis go to `islam.shia` and `islam.sunni` at that one share; non-Kuwaiti Muslims stay `islam`.
Witnesses, printed only: the State Department's "about 30%" of citizens (NGOs and the media) and
Pew 2009's 20-25% of all Muslims (500,000-700,000) both read higher.

## THE ASIAN `Other-Not Stated` IS SPLIT, NOTHING ELSE IS

Asians' `Other-Not Stated` (234,265 in 2014) is the Hindus, Buddhists and Sikhs, and is the only
category split: by the six Asian nationalities PACI's mid-2018 table names by sex (India,
Bangladesh, Philippines, Pakistan, Sri Lanka, Nepal), each weighted by its own Pew 2020 count of
everyone who is neither Muslim nor Christian, through `taxonomy/origin_religion.py`. The
magnitude is PACI's; the split is a model, and its rows roll back to `other.kw`. Every other
group's `Other-Not Stated` stays on `other.kw`. Christians are not split into churches: PACI's
Asian Christians (593,751) are about ten times what Pew's national shares give the named Asian
nationalities, so a nationality weighting has nothing to stand on.

## CHECKS

  * PACI 2014: every group's three categories sum to its total, the sexes to the both-sexes block,
    the groups to the national row (4,039,445);
  * 2021: Table 52's areas, read in order, close on Table 1's six governorates in all four
    nationality-by-sex columns (this is how each area gets its governorate); Table 6's groups sum
    to Table 1's totals; Table 1's governorates to the national 4,385,717;
  * GLMM 2014: each locality's groups sum to its printed total; the national row is 4,091,993;
  * the crosswalk: every 2014 locality with `XWALK_MIN` or more non-Kuwaitis is named in
    `XWALK_2014`, and every target is a 2021 area;
  * the raking converges, and the share of non-Kuwaitis it moves between groups is printed;
  * **witness**: the non-Kuwaiti Muslim, Christian and other shares as drawn against PACI's June
    2023 figures as the US State Department's *2023 Report on International Religious Freedom:
    Kuwait* quotes them (62.7%, 24.5%, 12.8%), each within `WITNESS_BAND` points;
  * the split is printed against the same report's informal community estimates (about 250,000
    Hindus, 100,000 Buddhists, 10,000-12,000 Sikhs); not asserted.

Usage:
    python sources/kw.py --fetch    PACI JSON (Wayback), two GLMM pages, three census workbooks
    python sources/kw.py            rebuild data/normalized/kw.csv
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

import numpy as np
import pandas as pd

from afrobarometer import round_within_rows
import origin_religion as origin

RAW = os.path.join(ROOT, "data", "raw", "kw")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")
OUT = os.path.join(ROOT, "data", "normalized", "kw.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
PACI_URL = ("http://web.archive.org/web/20141015093707id_/http://stat.paci.gov.kw/englishreports/"
            "tableQuery?viewId=ColumnChartEduAge&wafers=mf_Year,sf_gender&rows=sf_relegion&"
            "columns=sf_nationality_TD&resultWriter=json%3Aformat%3Dstandard%3BformatOutput%3D"
            "false%3BjavaScriptContentType%3Dfalse")
PACI_JSON = os.path.join(RAW, "paci_religion_by_nationality_group_2014-06.json")
GLMM = "https://gulfmigration.grc.net/"
GLMM_PAGES = {
    "locality_group_sex_2014": "kuwait-population-by-nationality-group-sex-and-administrative-sub-"
                               "region-locality-of-residence-december-2014/",
    "nationality_sex_2018": "kuwait-non-kuwaiti-population-by-region-and-selected-countries-of-"
                            "origin-and-sex-2018/",
}
CENSUS = "https://census.csb.gov.kw/CensusData_EN?st_id={}&handler=ExportExcel"
CENSUS_TABLES = {"t01": 4, "t06": 26, "t52": 72}

GOVS = ["Capital", "Hawalli", "Ahmadi", "Jahra", "Farwaniya", "Mubarak Al-Kabeer"]
GROUPS = ["ARAB", "ASIA", "AFRICA", "EUROPE", "NAM", "SAM", "OCEANIA"]
SEXES = ["M", "F"]

# PACI 2014's labels, in its own order, then the key used here
PACI_GROUPS = [("Kuwaiti", "KW"), ("Arabian", "ARAB"), ("Asian", "ASIA"), ("African", "AFRICA"),
               ("European", "EUROPE"), ("N.American", "NAM"), ("S.American", "SAM"),
               ("Australian", "OCEANIA"), ("Total", "TOTAL")]
PACI_SEXES = [("Total", "T"), ("Male", "M"), ("Female", "F")]
PACI_RELIGIONS = ["Muslim", "Christian", "Other-Not Stated", "Total"]
PACI_TOTAL = 4_039_445

# Table 1, pinned: governorate -> (Kuwaiti M, F, non-Kuwaiti M, F)
T01 = {"Capital": (138639, 144940, 172822, 118438),
       "Hawalli": (119799, 124120, 402839, 279412),
       "Ahmadi": (162729, 166392, 433003, 161660),
       "Jahra": (103272, 108888, 217562, 137139),
       "Farwaniya": (118602, 125310, 657701, 208206),
       "Mubarak Al-Kabeer": (86430, 89314, 55278, 48644)}
T01_NOT_STATED = (167, 114, 2423, 1874)
CENSUS_TOTAL = 4_385_717
N_AREAS = 157

# GLMM 2014's group columns, in its order
GLMM_GROUPS = ["KW", "ARAB", "ASIA", "AFRICA", "EUROPE", "NAM", "SAM", "OCEANIA"]
GLMM_2014_TOTAL = 4_091_993

# Each 2014 locality with non-Kuwaitis, as GLMM prints it, to the 2021 area (Table 52's English
# name) it became. Written from a name match and read row by row; None where nothing in 2021
# corresponds (Magwa and Sikrab, 173 and 156 non-Kuwaitis).
XWALK_2014 = {
    "Dasman": "DASMAN", "Sharq": "AL-SHARQ", "Mirqab": "AL-MURGAB", "Qibla": "AL-QIBLA",
    "Bneid Al – Gar": "BNIED AL-GAR", "Dasma": "AL-DASMA", "Mansoriya": "AL-MANSOURIA",
    "Abdalla-Alsalim": "ABDULLAH AL-SALEM", "Shamiya": "AL-SHAMIYA", "Diya": "AL-DAIYA",
    "Qadisiya": "AL-QADISIYA", "Nuzha": "AL-NUZHA", "Faiha": "AL-FAIHA", "Kifan": "KAIFAN",
    "Rawda": "AL-RAWDA", "Idailiya": "AL-ADAILIYA", "Khaldiya": "AL-KHALIDIYA", "Surra": "AL-SURA",
    "Qurtuba": "QURTOBA", "Al.yamouk": "AL-YARMOUK", "Shuwaikh": "AL-SHUWAIKH",
    "Shuwaikh – Ind": "AL-SHUWAIKH INDUSTRIAL", "Garnada": "GARNATA",
    "Mubarakiya Comp": "AL-MUBARAKIYA CAMPS", "Health Reg": "AL-SHUWAIKH MEDICAL",
    "Sulaibekhat": "AL-SULAIBIKHAT", "Doha": "AL-DOHA RESIDENTIAL", "Doha Port": "AL-DOHA PORT",
    "Hawalli": "HAWALLI", "Salmiya": "AL-SALMIYA", "Shaab": "ALSHAAB",
    "Rumaythiya": "AL-RUMAITHIYA", "Salwa": "SALWA", "Bedi": "AL-BIDA", "Mushaif": "MISHRIEF",
    "Mubarak Al-Abdel-Allah": "MUBARAK AL-ABDULLAH", "Bayan": "BAYAN", "Jabriya": "AL-JABRIYA",
    "Al-Shohadaa": "AL-SHUHADA", "Al-Zahraa": "AL-ZAHRA", "Hetteen": "HATEEN",
    "Al-Siddeek": "AL-SIDDIQ", "Al-Salam": "AL-SALAM", "Anjafa": "ANJAFA",
    "Ahmadi city": "AL-AHMADI CITY", "Fahaheel": "AL-FAHAHEEL", "Sabahiya": "AL-SABAHIYA",
    "Rikka": "AL-RIQQA", "Hadiya": "HADIYA", "Fintas": "AL-FINTAS", "Jaber Al-Ali": "JABER AL-ALI",
    "Auqqila": "AL-AQILA", "Abu- Alhasniya": "ABU AL-HASANIYA", "Mahbula": "AL-MAHBULA",
    "Abu-Halifa": "ABU HALIFA", "Munkaf": "ALMANGAF", "Thaher": "AL-DHAHER",
    "Shuaiba": "AL-SHUAIBA", "Shuaiba-ind W": "AL-SHUAIBA INDUSTRIAL",
    "Abdulla -Port": "ABDULLAH PORT", "Abdulla Port-Resort": "ABDULLAH PORT CHALETS",
    "Al-Kayron resorts": "AL-KHAIRAN CHALETS", "Jlaiaa Resort": "JULAIA CHALETS",
    "Zoor": "AL-ZOOR", "Wafra": "AL-WAFRA", "New Wafra": "AL-WAFRA RESIDENTIAL",
    "Wafra -Agriculture": "WAFRA FARMS", "Muqwaa": None,
    "Ahmadi – Desert": "AL-AHMADI GOVERNORATE DESERT", "Fahd Al – Ahmad": "FAHAD AL-AHMAD AL-JABER",
    "Ali Sabah Alsalem": "ALI SABAH AL-SALIM", "Jahra": "AL-JAHRA", "Al – Kasser": "AL-QASR",
    "Al – Naim": "AL-NAEEEM", "Al – Naseem": "AL-NASSEEM", "Taimaa": "TAIMA", "Waha": "AL-WAHA",
    "Al – Auyon": "AL-OYOUN", "Sekrab – Reg": None, "Jahraa Ind": "AL-JAHRA INDUSTRIAL 1",
    "Sulaibiya -Shabiya": "AL-SULAIBIYA RESIDENTIAL",
    "Sulaibiya – Ind (1)": "AL-SULAIBIYA INDUSTRAIL 1",
    "Sulaibiya – Ind (2)": "AL-SULAIBIYA INDUSTRAIL 2",
    "Sulaibiya -Agriculture": "AL-SULAIBIYA AGRICULTURAL", "Abdelli": "AL-ABDALLI",
    "Amgara – Ind": "AMGHARA INDUSTRIAL", "Kathma": "KAZMAH", "Al – Salmi": "AL-SALMI",
    "Kabad": "KABAD", "Jahara – Desert": "AL-JAHRA GOVERNORATE DESERT",
    "Saad Al – Abdulla city": "SAAD AL-ABDULLAH", "Qayrawan": "AL-QAIRAWAN",
    "Jaber al-Ahmad": "JABER AL-AHMED CITY", "Farwaniya": "AL-FARAWANIYA", "Khitan": "KHAITAN",
    "AlRaay": "AL-RAI", "Omarya": "AL-OMARIYA", "Rabiya": "AL-RABIYA", "Rihab": "AL-RIHAB",
    "Jleeb Al -Shuyoukh": "JLEEB AL-SHUYOUKH", "Reggae": "AL-RIGGAE", "Andalus": "AL-ANDALUS",
    "Ardiya": "AL-ARDIYA", "Sabah Alnasir": "SABAH AL-NASSER", "Ishbiliya": "ASHBELYA",
    "Ardiya(6)": "AL-ARDIYA STORES", "Fordus": "AL-FORDOUS", "International Airport": "THE AIRPORT",
    "Al – Nahda": "AL-NAHDA", "Abdulla Mubarak AlSabah": "ABDULLAH AL-MUBARAK",
    "Mubarak Kabeer": "MUBARAK AL-KABEER", "Qurain": "AL-QURIAN", "Al – Adan": "AL-ADAN",
    "Qosoor": "AL-QUSOUR", "Misila": "AL-MISILA", "Subah Alsalim": "SABAH AL-SALEM",
    "Fanatees": "AL-FUNAITEES", "Sabhan Ind": "SABHAN INDUSTRAIL", "Mid – Reg": "CENTRAL AREA",
}
XWALK_MIN = 100           # a 2014 locality with this many non-Kuwaitis must be in XWALK_2014
PRIOR_MIN = 100           # non-Kuwaitis of one sex in 2014 below which the governorate mix is used
IPF_TOL = 1e-9

# GLMM 2018 (PACI): Asian nationalities named, by sex; the Asia row
ASIA_2018 = {"India": "IN", "Bangladesh": "BD", "Philippines": "PH", "Pakistan": "PK",
             "Sri Lanka": "LK", "Nepal": "NP"}
ASIA_2018_TOTAL = (1_364_546, 503_662, 1_868_208)
NON_KUWAITI_2018 = (2_253_768, 964_757, 3_218_525)

OTHER_NODE = "other.kw"
# PACI June 2023, as the State Department's 2023 report quotes it, for non-citizens
WITNESS_2023 = {"Muslim": 0.627, "Christian": 0.245, "Other-Not Stated": 0.128}
WITNESS_BAND = 0.05
INFORMAL = {"hinduism": 250_000, "buddhism": 100_000, "sikhism": 11_000}

# Kuwaiti citizens' sect: Arab Barometer wave III, `q2005kw` (the field team's opinion), Muslims.
# Unweighted counts pinned so a re-release that changes the file stops here; the share drawn is
# the weighted Shia share of those the team placed (module docstring, sources/kw.md §8).
SECT_PIN = {"Sunni": 677, "Shia": 149, "Cannot determine": 192}
SECT_SHARE = 0.1769            # asserted to 4 places against the file
# State Department 2023 ("about 30%" of citizens, NGOs and media); printed, not asserted
SECT_WITNESS = 0.30
AB_GOV = {"Amman": "Capital", "Hawalli": "Hawalli", "Ahmadi": "Ahmadi",
          "al-Farwaniyah": "Farwaniya", "Mubarak al-Sabah": "Mubarak Al-Kabeer",
          "al-Jahra": "Jahra"}   # wave III's `Amman` is the Capital: a label shared across countries

# note_public's figures, measured 2026-10-03 and asserted
NOTE = dict(kuwaitis=1488435, non_kuwaitis=2892704, muslim_nk=1918101, christian=719529,
            other=255074, hinduism=216323, buddhism=20379, sikhism=5268,
            kw_shia=263299, kw_sunni=1225136)


def fetch():
    os.makedirs(RAW, exist_ok=True)

    def get(url, dst, ok, minsize):
        if os.path.exists(dst) and os.path.getsize(dst) > minsize:
            return
        print("  GET", url)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            data = r.read()
        if not ok(data):
            raise SystemExit(f"{url} did not return the expected file ({len(data):,} bytes)")
        with open(dst + ".part", "wb") as fh:
            fh.write(data)
        os.replace(dst + ".part", dst)

    get(PACI_URL, PACI_JSON, lambda d: b'"cellData"' in d and b"sf_relegion" in d, 1_000)
    for key, path in GLMM_PAGES.items():
        get(GLMM + path, os.path.join(RAW, f"glmm_{key}.html"), lambda d: b"<table" in d, 10_000)
    for key, sid in CENSUS_TABLES.items():
        get(CENSUS.format(sid), os.path.join(RAW, f"census2021_{key}.xlsx"),
            lambda d: d[:2] == b"PK", 5_000)


# ---------------------------------------------------------------------------------------------
# PACI, June 2014
# ---------------------------------------------------------------------------------------------
def load_paci():
    with open(PACI_JSON, encoding="utf-8") as fh:
        d = json.load(fh)
    res = d["result"]
    dims = {x["id"]: x["labels"] for x in res["dimensions"]}
    if (dims["mf_Year"] != ["June - 2014"] or dims["sf_gender"] != [s for s, _ in PACI_SEXES]
            or dims["sf_nationality_TD"] != [g for g, _ in PACI_GROUPS]
            or dims["sf_relegion"] != PACI_RELIGIONS):
        raise SystemExit(f"PACI's dimensions are not the pinned ones: {dims}")
    cells = res["cellData"]
    if len(cells) != len(PACI_SEXES) * len(PACI_GROUPS) * len(PACI_RELIGIONS):
        raise SystemExit(f"PACI: {len(cells)} cells")
    tab, k = {}, 0
    for _s, sk in PACI_SEXES:                     # wafer, then column, then row (religion)
        for _g, gk in PACI_GROUPS:
            m, c, o, t = cells[k:k + 4]
            k += 4
            if abs(m + c + o - t) > 0.5:
                raise SystemExit(f"PACI {sk}/{gk}: {m}+{c}+{o} != {t}")
            tab[(sk, gk)] = dict(zip(PACI_RELIGIONS, (m, c, o, t)))
    for gk in [g for _, g in PACI_GROUPS]:
        for r in PACI_RELIGIONS:
            if abs(tab[("M", gk)][r] + tab[("F", gk)][r] - tab[("T", gk)][r]) > 0.5:
                raise SystemExit(f"PACI {gk}/{r}: the sexes do not sum to the total")
    for sk in "TMF":
        for r in PACI_RELIGIONS:
            if abs(sum(tab[(sk, g)][r] for _, g in PACI_GROUPS[:-1]) - tab[(sk, "TOTAL")][r]) > 0.5:
                raise SystemExit(f"PACI {sk}/{r}: the groups do not sum to the national row")
    if tab[("T", "TOTAL")]["Total"] != PACI_TOTAL:
        raise SystemExit(f"PACI national total {tab[('T', 'TOTAL')]['Total']:,}")
    print(f"  PACI June 2014: religion by nationality group and sex, {PACI_TOTAL:,} people; "
          f"Kuwaitis {tab[('T', 'KW')]['Muslim'] / tab[('T', 'KW')]['Total']:.3%} Muslim")
    shares = {(s, g): {r: tab[(s, g)][r] / tab[(s, g)]["Total"] for r in PACI_RELIGIONS[:3]}
              for s in SEXES for g in ["KW"] + GROUPS}
    return tab, shares


# ---------------------------------------------------------------------------------------------
# the 2021 census
# ---------------------------------------------------------------------------------------------
def xl(key):
    return pd.read_excel(os.path.join(RAW, f"census2021_{key}.xlsx"), header=None)


def ints(row, cols):
    return tuple(int(row[c]) for c in cols)


def load_census():
    # Table 1
    t = xl("t01")
    t01 = {}
    for _i, r in t.iterrows():
        lab = str(r[11]).strip()
        for g in GOVS:
            if lab.lower().replace("the ", "").startswith(("al-" + g.lower(), g.lower())):
                t01[g] = ints(r, (2, 3, 5, 6))
        if lab == "Not Stated":
            t01["NS"] = ints(r, (2, 3, 5, 6))
        if lab == "Total":
            total = int(r[10])
    if {g: t01.get(g) for g in GOVS} != T01 or t01.get("NS") != T01_NOT_STATED:
        raise SystemExit(f"census Table 1 is not the pinned one: {t01}")
    if sum(sum(v) for v in t01.values()) != CENSUS_TOTAL or total != CENSUS_TOTAL:
        raise SystemExit("census Table 1 does not close on 4,385,717")

    # Table 6: governorate x sex x group (Gulf, Arabic, Asian, African, European, NAm, SAm, Aus)
    t = xl("t06")
    t06, gov = {}, None
    for _i, r in t.iterrows():
        lab = str(r[13]).strip()
        for g in GOVS:
            if lab.lower().replace("the ", "").startswith(("al-" + g.lower(), g.lower())):
                gov = g
        if lab in ("Not Stated", "Total"):
            gov = None
        sex = str(r[12]).strip()
        if gov and sex in ("Male", "Female"):
            v = ints(r, range(3, 12))
            if sum(v[:8]) != v[8]:
                raise SystemExit(f"Table 6 {gov}/{sex}: groups do not sum to the total")
            t06[(gov, sex[0])] = v[:8]
    if len(t06) != 12:
        raise SystemExit(f"Table 6: {len(t06)} governorate-sex rows")
    groups21 = {}
    for g in GOVS:
        for i, s in enumerate(SEXES):
            v = t06[(g, s)]
            kw = T01[g][i]
            if sum(v) != T01[g][i] + T01[g][2 + i]:
                raise SystemExit(f"Table 6 {g}/{s} does not equal Table 1")
            if v[0] < kw:
                raise SystemExit(f"Table 6 {g}/{s}: fewer Gulf nationals than Kuwaitis")
            # Gulf less Kuwaitis are the other GCC nationals, Arabs as PACI's 2014 groups have them
            groups21[(g, s)] = dict(zip(GROUPS, (v[0] - kw + v[1],) + v[2:]))
    # Table 52: areas in governorate order
    t = xl("t52")
    rows = []
    for _i, r in t.iterrows():
        en = str(r[10]).strip()
        if pd.isna(r[1]) or not re.fullmatch(r"\d+", str(r[1]).split(".")[0]):
            continue
        rows.append(dict(en=en, ar=str(r[0]).strip(), kw_m=int(r[1]), kw_f=int(r[2]),
                         nk_m=int(r[4]), nk_f=int(r[5])))
        if int(r[1]) + int(r[2]) != int(r[3]) or int(r[4]) + int(r[5]) != int(r[6]):
            raise SystemExit(f"Table 52 {en}: sexes do not sum")
    a = pd.DataFrame(rows)
    a = a[~a["en"].isin(["NOT STATED", "TOTAL"])].reset_index(drop=True)
    if len(a) != N_AREAS or a["en"].duplicated().any():
        raise SystemExit(f"Table 52: {len(a)} areas, expected {N_AREAS} unique")
    gi, acc, gov_of = 0, np.zeros(4, dtype=int), []
    for _i, r in a.iterrows():
        gov_of.append(GOVS[gi])
        acc += (r["kw_m"], r["kw_f"], r["nk_m"], r["nk_f"])
        if tuple(acc) == T01[GOVS[gi]]:
            gi, acc = gi + 1, np.zeros(4, dtype=int)
            if gi == len(GOVS):
                break
    if gi != len(GOVS) or len(gov_of) != N_AREAS:
        raise SystemExit("Table 52's areas, read in order, do not close on Table 1's governorates")
    a["gov"] = gov_of
    print(f"  census 2021: {N_AREAS} areas in Table 52 close on Table 1's six governorates in all "
          f"four columns; Table 6's groups equal Table 1")
    return a, groups21


# ---------------------------------------------------------------------------------------------
# GLMM's copies
# ---------------------------------------------------------------------------------------------
def glmm_table(key):
    with open(os.path.join(RAW, f"glmm_{key}.html"), encoding="utf-8") as fh:
        return pd.read_html(StringIO(fh.read()), thousands=None)[0]


def load_2014_localities():
    t = glmm_table("locality_group_sex_2014")
    loc, total = {}, None
    for i in range(len(t)):
        name, sex = str(t.iat[i, 0]).strip(), str(t.iat[i, 1]).strip()
        if sex not in ("Males", "Females", "Total"):
            continue
        v = [int(str(x).replace(",", "")) for x in t.iloc[i, 2:11]]
        if sum(v[:8]) != v[8]:
            raise SystemExit(f"GLMM 2014 {name}/{sex}: groups do not sum")
        if name == "Total" and sex == "Total":
            total = v[8]
        if name.startswith("Total") or name in ("Not Stated", "Total"):
            continue
        loc.setdefault(name, {})[sex[0]] = dict(zip(GLMM_GROUPS, v[:8]))
    if total != GLMM_2014_TOTAL:
        raise SystemExit(f"GLMM 2014 national total {total}")
    return loc


def load_asia_2018():
    t = glmm_table("nationality_sex_2018")
    got = {}
    for i in range(len(t)):
        lab = re.sub(r"^of which ", "", str(t.iat[i, 0]).strip())
        try:
            v = tuple(int(str(t.iat[i, j]).replace(",", "")) for j in (1, 2, 3))
        except ValueError:
            continue
        got[lab] = v
    if got.get("Asia") != ASIA_2018_TOTAL or got.get("Total") != NON_KUWAITI_2018:
        raise SystemExit(f"GLMM 2018: Asia {got.get('Asia')}, Total {got.get('Total')}")
    named = {iso: got[lab] for lab, iso in ASIA_2018.items()}
    if any(m + f != tt for m, f, tt in named.values()):
        raise SystemExit("GLMM 2018: a nationality's sexes do not sum")
    return named


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
def ipf(prior, rows, cols, where):
    """Rake `prior` (areas x groups) to row totals `rows` and column totals `cols`."""
    m = prior.copy()
    for _ in range(10_000):
        rs = m.sum(axis=1)
        m = m.mul(np.where(rs > 0, rows / rs.where(rs > 0, 1), 0), axis=0)
        cs = m.sum(axis=0)
        m = m.mul(np.where(cs > 0, cols / cs.where(cs > 0, 1), 0), axis=1)
        if (m.sum(axis=1) - rows).abs().max() < IPF_TOL * max(1, rows.sum()):
            return m
    raise SystemExit(f"raking did not converge in {where}")


def pew_table():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    return t[t["Year"] == 2020].set_index("Country")


def other_split(asia18, pew):
    """{sex: {node: share}} of the Asian `Other-Not Stated`, and the residual share."""
    out = {}
    for i, s in enumerate(SEXES):
        w = {}
        for iso, v in asia18.items():
            row = {f: float(pew.loc[origin.PEW_BY_ISO[iso], f]) for f in origin.FAMILIES}
            for node, sh in origin.composition(iso, row, OTHER_NODE).items():
                if node.startswith(("islam", "christianity")):
                    continue
                w[node] = w.get(node, 0.0) + v[i] * sh
        tot = sum(w.values())
        out[s] = {n: x / tot for n, x in w.items()}
        print(f"  Asian Other-Not Stated, {'men' if s == 'M' else 'women'}: "
              + ", ".join(f"{n} {x:.3f}" for n, x in sorted(out[s].items(), key=lambda kv: -kv[1])
                          if x >= 0.001))
    return out


def kuwaiti_sect(areas):
    """Shia share of Kuwaiti Muslims, one figure for the whole country (Anita, 2026-10-03).

    Arab Barometer wave III's `q2005kw`, the field team's opinion of each respondent's sect; the
    weighted Shia share of the Muslims it placed. Prints the bounds the undetermined allow, the
    share post-stratified to the 2021 Kuwaitis by governorate, and the State Department's figure.
    """
    import arabbarometer as ab
    df = ab.load("Kuwait", expect_waves=["III", "VII", "VIII"], waves=["III"],
                 omit={"VII": "no sect item for Kuwait: the sect column is present and empty "
                              "(sources/estimates.md, Gulf section)",
                       "VIII": "no sect item for Kuwait, as VII"},
                 extra={"sect": ("q2005kw",)})
    m = df[df["category"] == "Muslim"]
    got = m["sect"].value_counts().to_dict()
    if got != SECT_PIN:
        raise SystemExit(f"wave III q2005kw among Kuwaiti Muslims is {got}, pinned {SECT_PIN}")
    w = m.groupby("sect")["w"].sum()
    share = w["Shia"] / (w["Shia"] + w["Sunni"])
    lo, hi = w["Shia"] / w.sum(), (w["Shia"] + w["Cannot determine"]) / w.sum()
    g = m.groupby(["geo_raw", "sect"])["w"].sum().unstack()
    unknown = sorted(set(g.index) - set(AB_GOV))
    if unknown:
        raise SystemExit(f"wave III Kuwait governorate labels not in AB_GOV: {unknown}")
    g.index = g.index.map(AB_GOV)
    kw21 = (areas.assign(k=areas["kw_m"] + areas["kw_f"]).groupby("gov")["k"].sum())
    post = ((g["Shia"] / (g["Shia"] + g["Sunni"])) * kw21.reindex(g.index)).sum() / kw21.sum()
    print(f"\n  Kuwaiti sect (Arab Barometer III, field team's opinion, {len(m)} Muslims): Shia "
          f"{share:.2%} of those placed; {lo:.2%} if every undetermined is Sunni, {hi:.2%} if "
          f"every one is Shia; post-stratified to 2021 Kuwaitis {post:.2%}; State Department "
          f"about {SECT_WITNESS:.0%} (not asserted)")
    if round(share, 4) != SECT_SHARE:
        raise SystemExit(f"Shia share {share:.4f}, SECT_SHARE says {SECT_SHARE}")
    if abs(post - share) > 0.01:
        raise SystemExit(f"post-stratified {post:.4f} is more than a point from {share:.4f}")
    return SECT_SHARE


def main():
    need = [PACI_JSON] + [os.path.join(RAW, f"glmm_{k}.html") for k in GLMM_PAGES] + \
           [os.path.join(RAW, f"census2021_{k}.xlsx") for k in CENSUS_TABLES]
    if "--fetch" in sys.argv or not all(os.path.exists(p) for p in need):
        fetch()
    tab, shares = load_paci()
    areas, groups21 = load_census()
    loc14 = load_2014_localities()
    asia18 = load_asia_2018()
    pew = pew_table()

    # ---- the crosswalk ----
    names21 = set(areas["en"])
    bad = sorted(v for v in XWALK_2014.values() if v is not None and v not in names21)
    if bad:
        raise SystemExit(f"XWALK_2014 targets not in Table 52: {bad}")
    unlisted = sorted(n for n, d in loc14.items()
                      if sum(d["T"][g] for g in GROUPS) >= XWALK_MIN and n not in XWALK_2014)
    if unlisted:
        raise SystemExit(f"2014 localities with {XWALK_MIN}+ non-Kuwaitis not in XWALK_2014: {unlisted}")
    stale = sorted(n for n in XWALK_2014 if n not in loc14)
    if stale:
        raise SystemExit(f"XWALK_2014 names not in GLMM's 2014 table: {stale}")
    prior14 = {}
    for n, d in loc14.items():
        tgt = XWALK_2014.get(n)
        if tgt is None:
            continue
        for s in SEXES:
            p = prior14.setdefault((tgt, s), dict.fromkeys(GROUPS, 0))
            for g in GROUPS:
                p[g] += d[s][g]

    # ---- non-Kuwaitis by area, sex and group: 2014 mix raked to 2021 ----
    raked = {}
    moved = 0.0
    n_prior = {"2014": 0, "governorate": 0}
    for gov in GOVS:
        sub = areas[areas["gov"] == gov]
        for i, s in enumerate(SEXES):
            col = pd.Series(groups21[(gov, s)])
            rows = sub.set_index("en")["nk_" + s.lower()].astype(float)
            pr = []
            for en in rows.index:
                p = prior14.get((en, s))
                if p and sum(p.values()) >= PRIOR_MIN:
                    pr.append(pd.Series(p, dtype=float) / sum(p.values()))
                    n_prior["2014"] += 1
                else:
                    pr.append(col / col.sum())
                    n_prior["governorate"] += 1
            prior = pd.DataFrame(pr, index=rows.index)[GROUPS].mul(rows, axis=0)
            if abs(rows.sum() - col.sum()) > 0.5:
                raise SystemExit(f"{gov}/{s}: Table 52 areas {rows.sum()} != Table 6 {col.sum()}")
            # a group the governorate has but no prior area holds gets a floor so it can be raked
            for g in GROUPS:
                if col[g] > 0 and prior[g].sum() == 0:
                    prior[g] = rows * 1e-6
            m = ipf(prior, rows, col[GROUPS].astype(float), f"{gov}/{s}")
            moved += 0.5 * (m - prior).abs().sum().sum()
            raked[(gov, s)] = m
    nk_total = int(areas["nk_m"].sum() + areas["nk_f"].sum())
    print(f"  raking: {n_prior['2014']} area-sex rows start from their 2014 mix, "
          f"{n_prior['governorate']} from the governorate's; it moves {moved:,.0f} of {nk_total:,} "
          f"non-Kuwaitis ({moved / nk_total:.1%}) between groups")

    # ---- religion ----
    split = other_split(asia18, pew)
    shia = kuwaiti_sect(areas)
    kwc = ["Kuwaiti citizens, Shia", "Kuwaiti citizens, Sunni"]
    cats = ["Kuwaitis", "Muslim", "Christian", "Other or not stated"] + \
           [f"Other or not stated, {n}" for n in sorted({n for s in split.values() for n in s})
            if n != OTHER_NODE]
    recs = []
    for _i, r in areas.iterrows():
        c = dict.fromkeys(cats, 0.0)
        c["Kuwaitis"] = r["kw_m"] + r["kw_f"]
        for s in SEXES:
            m = raked[(r["gov"], s)].loc[r["en"]]
            for g in GROUPS:
                n = m[g]
                sh = shares[(s, g)]
                c["Muslim"] += n * sh["Muslim"]
                c["Christian"] += n * sh["Christian"]
                o = n * sh["Other-Not Stated"]
                if g == "ASIA":
                    for node, x in split[s].items():
                        key = "Other or not stated" if node == OTHER_NODE else \
                              f"Other or not stated, {node}"
                        c[key] += o * x
                else:
                    c["Other or not stated"] += o
        recs.append(c)
    m = pd.DataFrame(recs, index=areas["en"])[cats]
    counts = round_within_rows(m)
    # Kuwaitis are a whole census count per area; the sect split is rounded inside it, so no
    # rounding moves anyone between Kuwaitis and non-Kuwaitis
    k = counts.pop("Kuwaitis")
    counts.insert(0, kwc[1], k - (k * shia).round().astype(k.dtype))
    counts.insert(0, kwc[0], (k * shia).round().astype(k.dtype))
    tot =areas.set_index("en")[["kw_m", "kw_f", "nk_m", "nk_f"]].sum(axis=1)
    if not (counts.sum(axis=1) == tot).all():
        raise SystemExit("rounded counts do not sum to Table 52's area totals")

    # ---- witness ----
    nk = counts.drop(columns=kwc)
    fam = {"Muslim": nk["Muslim"].sum(), "Christian": nk["Christian"].sum(),
           "Other-Not Stated": nk[[c for c in nk if c.startswith("Other")]].sum().sum()}
    n_nk = sum(fam.values())
    print(f"\n  witness: non-Kuwaitis as drawn against PACI June 2023 (State Department 2023):")
    fail = []
    for k, v in fam.items():
        print(f"      {k:<17} {v:>10,} {v / n_nk:6.1%}   PACI 2023 {WITNESS_2023[k]:6.1%}")
        if abs(v / n_nk - WITNESS_2023[k]) > WITNESS_BAND:
            fail.append(k)
    drawn_split = {n: int(nk.get(f"Other or not stated, {n}", pd.Series(0)).sum()) for n in INFORMAL}
    print("  the split against the report's informal community estimates (not asserted): "
          + ", ".join(f"{n} {drawn_split[n]:,} (informal {INFORMAL[n]:,})" for n in INFORMAL))
    if fail:
        raise SystemExit(f"outside the witness band ({WITNESS_BAND:.0%} points): {fail}")

    # ---- per governorate ----
    by_gov = counts.groupby(areas.set_index("en")["gov"]).sum()
    print("\n  per governorate, drawn:")
    for g in GOVS:
        r = by_gov.loc[g]
        t = r.sum()
        oth = r[[c for c in r.index if c.startswith("Other")]].sum()
        print(f"      {g:<18} {t:>9,}  Muslim {(r[kwc].sum() + r['Muslim']) / t:6.1%}  "
              f"Christian {r['Christian'] / t:6.1%}  other {oth / t:6.1%}")
    top = (counts["Christian"] / tot).sort_values(ascending=False)
    print("  most Christian areas (500+ people): " + ", ".join(
        [f"{a} {top[a]:.1%}" for a in top.index if tot[a] >= 500][:8]))

    # ---- write ----
    long = counts.stack().rename("count").reset_index()
    long.columns = ["geo_id", "source_category", "count"]
    long = long[long["count"] > 0].copy()
    long["geo_level"] = "area"
    long["geo_name"] = long["geo_id"]
    long["governorate"] = long["geo_id"].map(dict(zip(areas["en"], areas["gov"])))
    long["basis"] = "register"
    long["year"] = 2021
    long["tier"] = "derived"
    long["source_id"] = "census2021_t52_x_paci2014_religion"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    long[["geo_id", "geo_level", "geo_name", "governorate", "source_category", "count", "tier",
          "basis", "year", "source_id"]].to_csv(OUT, index=False, encoding="utf-8")
    fx = long.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"\nwrote {OUT}: {long['geo_id'].nunique()} areas, {int(long['count'].sum()):,} people")
    print("    " + ", ".join(f"{k} {int(v):,}" for k, v in fx.items()))
    got = dict(kuwaitis=int(counts[kwc].sum().sum()), non_kuwaitis=int(nk.sum().sum()),
               muslim_nk=int(fam["Muslim"]), christian=int(fam["Christian"]),
               other=int(fam["Other-Not Stated"]), **{n: drawn_split[n] for n in INFORMAL},
               kw_shia=int(counts[kwc[0]].sum()), kw_sunni=int(counts[kwc[1]].sum()))
    print(f"\n  note_public's figures: {got}")
    if "--no-note" not in sys.argv and got != NOTE:
        raise SystemExit(f"note_public's figures are {NOTE}; the build gives {got}")


if __name__ == "__main__":
    main()
