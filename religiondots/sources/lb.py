"""Lebanon: the 2022 electoral register by sect, carried to the 26 cazas, scaled to OCHA's 2026
count of resident Lebanese; Syrians and Palestinians by caza from the same OCHA package.

Anita's ruling, ask 049 (2026-10-03): draw Lebanon from the electoral register by caza, labelled as
where families are registered and not where they live. `sources/lb.md` §10 is the record in prose.

## THE FOUR INPUTS (all in data/raw/lb/)

  1. **The register by sect and electoral district, 2022.** Information International, *The
     Monthly* no. 187 (April-May 2022), "Lebanon's 2022 voters by sect and district", Tables 1-16,
     "based on the figures issued by the Ministry of Interior and Municipalities". Table 1 is the
     nation, Tables 2-16 the fifteen electoral districts of the 2017 law. Read off the PDF's text
     layer (`read_monthly`); the 2022 column is used, 2018 is printed beside it.
  2. **The register by sect and caza, 2014.** lub-anan.com, "per the official voter lists issued by
     the Lebanese Ministry of Interior for 2014" (its disclaimer page: the lists the Ministry issued
     on discs). One page per caza with every sect's count; 29 pages, because Beirut is split into the
     2008 law's three districts and Saida into city and villages. Only these aggregate pages are read.
  3. **Registered voters per minor district, 2022.** L'Orient Today's Tableau Public workbook
     "Registered Voters by District" (Richard Salame and Iva Kovic, 2022), the view's own CSV export:
     25 minor districts, three of them caza pairs (West Bekaa-Rashaya, Marjayoun-Hasbaya,
     Baalbek-Hermel). Only that export is used. THE PACKAGED WORKBOOK IS NOT TO BE DOWNLOADED: its
     extract is the voter roll itself, with names, parents' names, dates of birth and sect for each
     of 3.97 million people. It was fetched once by accident on 2026-10-03, its column list read,
     and deleted unopened (sources/lb.md §10.2).
  4. **OCHA Lebanon's 2026 LRP population package** (HDX), Lebanese, Syrians, Palestinians and
     migrants by caza. Its Lebanese total is CAS's 2018-19 Labour Force and Household Living
     Conditions Survey.

## THE CONSTRUCTION

The 2022 register is printed by electoral district, and nine of the fifteen hold two to four cazas.
So each district's 2022 sect counts are raked (IPF) onto its minor districts, with the 2014 caza
register as the seed and two margins: each published 2022 sect row of the district, and each minor
district's 2022 total from L'Orient. The three caza pairs L'Orient merges are then split per sect on
the 2014 register. Six cazas are whole electoral districts (Beirut is two: Beirut I and II), and
their 2022 counts are used as printed. Every caza total therefore equals the 2022 register, every
district's sect rows equal the 2022 register, and only the split of a sect between cazas of one
district comes from 2014.

The registered voters (aged 21 and over, emigrants included) are then scaled by one national factor,
OCHA's 3,864,296 resident Lebanese over the register's 3,967,507, so the map draws as many Lebanese as
live in Lebanon, each caza in proportion to how many are REGISTERED there. Nothing moves anyone to
where they live: there is no published conversion from caza of registration to caza of residence.

Syrians and Palestinians are drawn where OCHA counts them, at Pew Research Center's 2020 mix for
Syria and for the Palestinian territories (`origin_mix`). Migrants (164,097) are not drawn.

## CHECKS (each one stops the build; `PINNED` lists the printed misprints, each with its evidence)

  * every Monthly table's 2022 rows sum to its printed Total, and that Total to the district total
    printed above the table (`HEADER_2022`) and to L'Orient's minor districts summed;
  * the fifteen districts sum to Table 1's 3,967,507, sect by sect where Table 1 can be compared;
  * every lub-anan page's sects sum to its own grand total and its block subtotals, and the 29 pages
    to the site's national 3,514,588;
  * the 2014 seed summed to each 2022 district, against the 2022 table: growth per sect inside a band
    (`GROWTH_BAND`), which is what catches a sect mapped to the wrong row;
  * the rake closes on both margins; the 26 cazas sum to the register;
  * OCHA's caza rows sum to its governorate rows and its national totals.

Usage:
    python sources/lb.py --fetch    the PDF, 29 lub-anan pages, L'Orient's CSV, OCHA's package
    python sources/lb.py            rebuild data/normalized/lb.csv from data/raw/lb/
"""

import io
import os
import re
import sys
import time
import zipfile
from html import unescape

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

RAW = os.path.join(ROOT, "data", "raw", "lb")
OUT = os.path.join(ROOT, "data", "normalized", "lb.csv")
PEW = os.path.join(ROOT, "data", "raw", "estimates", "pew.zip")

MONTHLY_PDF = os.path.join(RAW, "monthly_2022_elections.pdf")
MONTHLY_URL = "https://monthlymagazine.com/cms/upload/magazine/630f55c7ec72b382_file.pdf"
MONTHLY_SIZE = 4_609_114
OLJ_CSV = os.path.join(RAW, "olj_registered_voters_by_district.csv")
OLJ_URL = ("https://public.tableau.com/views/RegisteredVotersbyDistrict/"
           "RegisteredVotersbyDistrict.csv?:showVizHome=no")
OCHA_XLSX = os.path.join(RAW, "05.-2026-lrp-population-package.xlsx")
OCHA_URL = ("https://data.humdata.org/dataset/d0f7980a-4b72-4744-9065-a036c89dc3b5/resource/"
            "feef36a8-cbf8-48fc-9bd3-758e03a7b85a/download/05.-2026-lrp-population-package.xlsx")
LUBANAN_DIR = os.path.join(RAW, "lubanan2014")
LUBANAN_BASE = "https://lub-anan.com/المحافظات/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

VOTERS_2022 = 3_967_507
LUBANAN_NATIONAL = 3_514_588
OCHA = dict(lebanese=3_864_296, syrian=1_120_000, palestinian=224_791, migrant=164_097)
YEAR = 2022
SOURCE_ID = "lb_moim2022_register"

# ---------------------------------------------------------------------------------------------
# The 26 cazas, COD-AB `cod-ab-lbn` adm2 (2026-01 release) names as OCHA's package also spells them.
# ---------------------------------------------------------------------------------------------
PCODE = {
    "Beirut": "LB11", "Baalbek": "LB21", "El Hermel": "LB22", "Rachaya": "LB23",
    "West Bekaa": "LB24", "Zahle": "LB25", "Aley": "LB31", "Baabda": "LB32", "Chouf": "LB33",
    "Jbeil": "LB34", "Kesrwane": "LB35", "El Meten": "LB36", "Bent Jbeil": "LB41",
    "Hasbaya": "LB42", "Marjaayoun": "LB43", "El Nabatieh": "LB44", "Akkar": "LB51",
    "El Batroun": "LB52", "Bcharre": "LB53", "El Koura": "LB54", "El Minieh-Dennie": "LB55",
    "Tripoli": "LB56", "Zgharta": "LB57", "Saida": "LB61", "Jezzine": "LB62", "Sour": "LB63",
}

# lub-anan's 29 pages: (governorate path, caza path) -> the COD caza it belongs to. Beirut's three
# 2008-law districts are only a seed for nothing (Beirut is two whole electoral districts), but are
# read for the national check.
LUBANAN = {
    ("البقاع", "البقاع-الغربي"): "West Bekaa", ("البقاع", "الهرمل"): "El Hermel",
    ("البقاع", "بعلبك"): "Baalbek", ("البقاع", "راشيا"): "Rachaya", ("البقاع", "زحلة"): "Zahle",
    ("الجنوب", "جزين"): "Jezzine", ("الجنوب", "صور"): "Sour",
    ("الجنوب", "صيدا-قرى"): "Saida (villages)", ("الجنوب", "صيدا-مدينة"): "Saida (city)",
    ("الشمال", "البترون"): "El Batroun", ("الشمال", "الكورة"): "El Koura",
    ("الشمال", "المنية-الضنية"): "El Minieh-Dennie", ("الشمال", "بشري"): "Bcharre",
    ("الشمال", "زغرتا"): "Zgharta", ("الشمال", "طرابلس"): "Tripoli", ("الشمال", "عكار"): "Akkar",
    ("النبطية", "النبطية"): "El Nabatieh", ("النبطية", "بنت-جبيل"): "Bent Jbeil",
    ("النبطية", "حاصبيا"): "Hasbaya", ("النبطية", "مرجعيون"): "Marjaayoun",
    ("بيروت", "بيروت-الأولى"): "Beirut (2008 I)", ("بيروت", "بيروت-الثانية"): "Beirut (2008 II)",
    ("بيروت", "بيروت-الثالثة"): "Beirut (2008 III)",
    ("جبل-لبنان", "الشوف"): "Chouf", ("جبل-لبنان", "المتن-الشمالي"): "El Meten",
    ("جبل-لبنان", "بعبدا"): "Baabda", ("جبل-لبنان", "جبيل"): "Jbeil", ("جبل-لبنان", "عاليه"): "Aley",
    ("جبل-لبنان", "كسروان"): "Kesrwane",
}

# lub-anan's sect labels -> the row of The Monthly's Table 1 they are counted in. Armenian
# Evangelicals are part of the Evangelical community, one of the eighteen recognised sects;
# `نسطوري` (Nestorian) is the Assyrian Church of the East. Everything a 2022 table prints as
# `Others` is the last group.
SECT_2014 = {
    "سني": "Sunni", "شيعي": "Shia", "علوي": "Alawite", "درزي": "Druze", "اسماعيليي": "Others",
    "ماروني": "Maronite", "روم ارثوذكس": "Greek Orthodox", "روم كاثوليك": "Greek Catholic",
    "ارمن ارثوذكس": "Armenian Orthodox", "ارمن كاثوليك": "Armenian Catholic",
    "انجيلي (بروتستانت)": "Evangelical", "ارمن بروتستانت": "Evangelical",
    "سريان ارثوذكس": "Syriac Orthodox", "سريان كاثوليك": "Syriac Catholic", "لاتيني": "Latin",
    "اسرائيلي": "Israeli", "اشوري": "Assyrian Orthodox", "نسطوري": "Assyrian Orthodox",
    "كلدان": "Chaldean", "كلدان كاثوليك": "Chaldean", "كلدان ارثوذكس": "Chaldean",
    "قبطي": "Others", "قبطي ارثوذكس": "Others", "قبطي كاثوليك": "Others", "مسيحي": "Others",
    "شهود يهوه": "Others", "بهائي": "Others", "بوذي": "Others", "هندوسي": "Others",
    "غير مذكور": "Others", "لا طائفي": "Others", "للتدقيق": "Others", "مختلف": "Others",
}
SUBTOTALS = {"مجموع مسيحي", "مجموع مسلم", "مجموع مختلف"}
GRAND = "المجموع العام"

FINE = ["Sunni", "Shia", "Maronite", "Greek Orthodox", "Druze", "Greek Catholic",
        "Armenian Orthodox", "Alawite", "Armenian Catholic", "Evangelical", "Syriac Orthodox",
        "Syriac Catholic", "Latin", "Israeli", "Assyrian Orthodox", "Chaldean", "Others"]

# The Monthly's Tables 2-16 -> (district, the L'Orient minor districts in it).
DISTRICTS = {
    2: ("Beirut I", ["Beirut I"]),
    3: ("Beirut II", ["Beirut II"]),
    4: ("Mount Lebanon I", ["Keserwan", "Jbeil"]),
    5: ("Mount Lebanon II", ["Metn"]),
    6: ("Mount Lebanon III", ["Baabda"]),
    7: ("Mount Lebanon IV", ["Chouf", "Aley"]),
    8: ("Bekaa I", ["Zahle"]),
    9: ("Bekaa II", ["West Bekaa - Rashaya"]),
    10: ("Bekaa III", ["Baalbek-Hermel"]),
    11: ("North I", ["Akkar"]),
    12: ("North II", ["Tripoli", "Minyeh-Dinnieh"]),
    13: ("North III", ["Bsharri", "Zgharta", "Batroun", "Koura"]),
    14: ("South I", ["Saida", "Jezzine"]),
    15: ("South II", ["Zahrani", "Sour"]),
    16: ("South III", ["Nabatieh", "Bint Jbeil", "Marjayoun-Hasbaya"]),
}
# L'Orient's minor districts -> the lub-anan seed units (one, or a caza pair) and COD cazas.
MINOR = {
    "Beirut I": [], "Beirut II": [],
    "Keserwan": ["Kesrwane"], "Jbeil": ["Jbeil"], "Metn": ["El Meten"], "Baabda": ["Baabda"],
    "Chouf": ["Chouf"], "Aley": ["Aley"], "Zahle": ["Zahle"],
    "West Bekaa - Rashaya": ["West Bekaa", "Rachaya"], "Baalbek-Hermel": ["Baalbek", "El Hermel"],
    "Akkar": ["Akkar"], "Tripoli": ["Tripoli"], "Minyeh-Dinnieh": ["El Minieh-Dennie"],
    "Bsharri": ["Bcharre"], "Zgharta": ["Zgharta"], "Batroun": ["El Batroun"],
    "Koura": ["El Koura"], "Saida": ["Saida (city)"], "Jezzine": ["Jezzine"],
    "Zahrani": ["Saida (villages)"], "Sour": ["Sour"], "Nabatieh": ["El Nabatieh"],
    "Bint Jbeil": ["Bent Jbeil"], "Marjayoun-Hasbaya": ["Marjaayoun", "Hasbaya"],
}
SEED_TO_CAZA = {"Saida (city)": "Saida", "Saida (villages)": "Saida"}

# Each district's total, printed above its table ("Total registered voters 2022"), transcribed.
HEADER_2022 = {2: 134_825, 3: 370_862, 4: 182_103, 5: 183_441, 6: 171_746, 7: 346_451,
               8: 183_425, 9: 153_975, 10: 341_263, 11: 309_517, 12: 377_111, 13: 257_964,
               14: 129_229, 15: 328_064, 16: 497_531}

# Printed misprints, each with the evidence that names it.
PINNED = {
    # Table 9 prints Total 2022 153,974; its rows sum to 153,975, the header and L'Orient both give
    # 153,975. (Its 2018 Syriac Catholic, 439, is also a misprint for 139: the 2018 rows then sum to
    # the printed 143,812 and the difference column says +1. Only 2022 is used.)
    ("total", 9): 153_974,
}
# Nine tables' 2022 rows do not sum to the district's total, by 1 to 44 voters, while the total
# printed above the table, the table's own Total row and L'Orient's count from the roll all agree.
# The difference column finds three of the slips (Table 4 `Others` +99 for +9; Table 12 `Others`
# +13 for +15; Table 13 `Alawite` +57 for +58) and none of them closes its table, so the rows are
# kept as printed and each district's rows are scaled to its total: at most 0.017% (North III).
# Pinned, so a re-read that parses a different number stops.
ROW_SUM_OFF = {4: 2, 5: 1, 6: -5, 8: 1, 11: 1, 12: 10, 13: 44, 14: 2, 15: 1}
# Table 1 against the fifteen districts summed. Ten sects agree to the voter. Greek Orthodox is
# 2,554 higher in the districts, and the Syriac, Chaldean and `Others` rows are grouped differently
# in Table 1 (its Syriac Orthodox is 15,672 where the districts print 6,530 plus 13,047 merged
# `Syriac`). The districts are what is drawn; Table 1 is the witness, pinned as printed.
NATIONAL_OFF = {"Greek Orthodox": 2_554, "Greek Catholic": -3, "Israeli": -11,
                "Syriac Orthodox": -9_142, "Syriac Catholic": -3_000, "Chaldean": -997,
                "Others": -2_391}

# A sect's 2022 count over its 2014 count in one district. The national register grew 12.9%
# (3,514,588 to 3,967,507); the band is wide because small sects move by tens of people.
GROWTH_BAND = (0.70, 1.45)
GROWTH_MIN = 2_000          # only sects of this size in the district are held to the band
# Outside the band, each explained. lub-anan counts the PERSONAL sect (`مذهب`) and the 2022 tables the
# REGISTER sect (`مذهب السجل`, the family record's): lub-anan's minority sects are mostly women (El Koura's
# Greek Catholics 593 women, 98 men; nationally 12,126 of the 13,857 `not stated` are women), who keep their
# own sect on marrying into a family registered under another. So in Maronite Kesrouan-Jbeil the 2014 seed
# has about twice the Greek Orthodox and Greek Catholics the 2022 register files there; the 2018 column of
# the same table (3,361 and 2,328) agrees with 2022. North II's Alawites: the 2018 column (20,227) is
# already 35% above 2014, so the jump is in the register, not the parse. sources/lb.md §10.3.
GROWTH_PINNED = {("Mount Lebanon I", "Greek Orthodox"): 0.49,
                 ("Mount Lebanon I", "Greek Catholic"): 0.41,
                 ("North II", "Alawite"): 1.47}


def _get(url, dst, minsize=1_000, pause=0.0):
    import requests

    if os.path.exists(dst) and os.path.getsize(dst) >= minsize:
        return
    print("GET", url)
    r = requests.get(url, timeout=300, headers={"User-Agent": UA})
    r.raise_for_status()
    if len(r.content) < minsize:
        raise SystemExit(f"{url}: {len(r.content)} bytes, expected at least {minsize}")
    with open(dst + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(dst + ".part", dst)
    time.sleep(pause)


def fetch():
    os.makedirs(LUBANAN_DIR, exist_ok=True)
    _get(MONTHLY_URL, MONTHLY_PDF, minsize=MONTHLY_SIZE)
    if os.path.getsize(MONTHLY_PDF) != MONTHLY_SIZE:
        raise SystemExit(f"{MONTHLY_PDF} is not {MONTHLY_SIZE:,} bytes")
    _get(OLJ_URL, OLJ_CSV, minsize=1_500)
    _get(OCHA_URL, OCHA_XLSX, minsize=100_000)
    for (gov, caza) in LUBANAN:
        _get(f"{LUBANAN_BASE}{gov}/{caza}/المذاهب/", _lubanan_path(gov, caza), minsize=8_000,
             pause=1.0)


def _lubanan_path(gov, caza):
    return os.path.join(LUBANAN_DIR, f"{gov}__{caza}.html")


# ---------------------------------------------------------------------------------------------
# 1. The Monthly, 2022 by electoral district
# ---------------------------------------------------------------------------------------------
_NUM = re.compile(r"^[+\-]?\s*[\d,]+$")


def _label(s):
    s = re.sub(r"\s+", " ", s.replace("‘", "").replace("’", "")).strip()
    s = {"Shiaa": "Shia", "Chaldean Catholic": "Chaldean"}.get(s, s)
    return s


def read_monthly():
    """{table no: {row label: 2022 count}} with each table's printed 2022 total under 'Total'."""
    import fitz

    with open(MONTHLY_PDF, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            raise SystemExit(f"{MONTHLY_PDF} is not a PDF")
    doc = fitz.open(MONTHLY_PDF)
    text = "\n".join(p.get_text() for p in doc)
    out = {}
    for no in range(1, 17):
        m = re.search(rf"Table No\. {no}:", text)
        if not m:
            raise SystemExit(f"Table No. {no} not found")
        start = text.index("Difference", m.end())
        lines = [ln.strip() for ln in text[start:].split("\n")[1:]]
        if no == 1:                       # Table 1 adds Percentage and % to the header
            while lines[0] in ("Percentage", "%"):
                lines.pop(0)
        rows, label, nums = {}, None, []
        for ln in lines:
            if not ln:
                continue
            if _NUM.match(ln) or ln == "-":
                nums.append(ln)
                continue
            if re.match(r"^[+\-]?\d+(\.\d+)?$", ln):
                nums.append(ln)
                continue
            if label is not None:
                if not nums or nums[0] == "-":
                    raise SystemExit(f"Table {no}: no 2022 figure for {label!r}")
                rows[label] = int(nums[0].replace(",", "").replace(" ", "").lstrip("+"))
                if label == "Total":
                    break
            label, nums = _label(ln), []
        else:
            raise SystemExit(f"Table {no}: no Total row")
        out[no] = rows
    return out


def check_monthly(tab, olj):
    bad = []
    say = lambda ok, msg: (print(("  ok   " if ok else "  FAIL ") + msg), ok or bad.append(msg))
    for no, rows in tab.items():
        total = rows["Total"]
        s = sum(v for k, v in rows.items() if k != "Total")
        printed = PINNED.get(("total", no), total)
        if total != printed:
            raise SystemExit(f"Table {no}: Total {total} but PINNED says {printed}")
        expect = HEADER_2022.get(no, VOTERS_2022)
        off = ROW_SUM_OFF.get(no, 0)
        say(s - expect == off, f"Table {no:>2}: rows sum {s:,}, header {expect:,}, printed Total "
                               f"{total:,}" + (f", rows off by {off:+} (pinned)" if off else "")
                               + (" (Total a pinned misprint)" if ("total", no) in PINNED else ""))
        if ("total", no) not in PINNED:
            say(total == expect, f"          printed Total equals the header")
        if no in DISTRICTS:
            o = sum(olj[m] for m in DISTRICTS[no][1])
            say(o == expect, f"          L'Orient's minor districts sum {o:,}")
    # the districts against Table 1, row by row
    nat = {k: v for k, v in tab[1].items() if k != "Total"}
    summed = {}
    for no in DISTRICTS:
        for k, v in tab[no].items():
            if k != "Total":
                summed[k] = summed.get(k, 0) + v
    s_all = sum(summed.values())
    say(s_all - VOTERS_2022 == sum(ROW_SUM_OFF.values()),
        f"fifteen districts' rows sum {s_all:,} against {VOTERS_2022:,} (the pinned row offsets)")
    syr = summed.pop("Syriac", 0)
    print("  sect by sect, Table 1 against the fifteen districts (Syriac printed merged in four "
          f"districts, {syr:,}):")
    diffs = {}
    for k in sorted(set(nat) | set(summed), key=lambda x: -nat.get(x, 0)):
        d = summed.get(k, 0) - nat.get(k, 0)
        if d:
            diffs[k] = d
        print(f"      {k:<18} Table 1 {nat.get(k, 0):>10,}   districts {summed.get(k, 0):>10,}"
              f"   {d:+,}")
    say(diffs == NATIONAL_OFF, f"the differences are NATIONAL_OFF's, and the other "
                               f"{len(set(nat) - set(diffs))} sects agree to the voter")
    so_sc = summed.get("Syriac Orthodox", 0) + summed.get("Syriac Catholic", 0) + syr
    print(f"      Syriac, both     Table 1 {nat['Syriac Orthodox'] + nat['Syriac Catholic']:>10,}"
          f"   districts {so_sc:>10,}")
    return bad, diffs


# ---------------------------------------------------------------------------------------------
# 2. lub-anan, 2014 by caza
# ---------------------------------------------------------------------------------------------
_AR = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _ar_int(s):
    return int(s.translate(_AR).replace(",", "").replace("٬", ""))


def read_lubanan():
    """{seed unit: {Table-1 row: count}}, with the page's own totals asserted."""
    out, bad = {}, []
    for (gov, caza), unit in LUBANAN.items():
        with open(_lubanan_path(gov, caza), encoding="utf-8") as fh:
            html = fh.read()
        toks = [t.strip() for t in re.sub(r"<[^>]+>", "\n", unescape(html)).split("\n")]
        toks = [t for t in toks if t]
        i = toks.index("%") + 1
        rows, subs, grand = {}, {}, None
        while i + 4 <= len(toks):
            lab, a, b, c, pct = toks[i:i + 5]
            if not re.match(r"^\(.*%\)$", pct):
                break
            n = _ar_int(c)
            if _ar_int(a) + _ar_int(b) != n:
                raise SystemExit(f"lub-anan {unit}: {lab} {a} + {b} != {c}")
            if lab == GRAND:
                grand = n
                break
            if lab in SUBTOTALS:
                subs[lab] = n
            else:
                if lab not in SECT_2014:
                    raise SystemExit(f"lub-anan {unit}: sect label {lab!r} has no row")
                rows[lab] = n
            i += 5
        if grand is None:
            raise SystemExit(f"lub-anan {unit}: no grand total")
        if sum(rows.values()) != grand or sum(subs.values()) != grand:
            bad.append(f"lub-anan {unit}: sects {sum(rows.values()):,}, blocks "
                       f"{sum(subs.values()):,}, grand {grand:,}")
        fine = {}
        for lab, n in rows.items():
            fine[SECT_2014[lab]] = fine.get(SECT_2014[lab], 0) + n
        out[unit] = fine
    nat = sum(sum(v.values()) for v in out.values())
    print(f"  lub-anan 2014: {len(out)} pages, {nat:,} voters against the site's {LUBANAN_NATIONAL:,}")
    if nat != LUBANAN_NATIONAL:
        bad.append(f"lub-anan pages sum {nat:,}")
    for b in bad:
        print("  FAIL " + b)
    return out, bad


# ---------------------------------------------------------------------------------------------
# 3. L'Orient's minor-district totals, and OCHA's package
# ---------------------------------------------------------------------------------------------
def read_olj():
    t = pd.read_csv(OLJ_CSV, thousands=",", encoding="utf-8")
    t = t[t["admin2Name1"].notna() & (t["admin2Name1"] != "Conflict")]
    olj = dict(zip(t["admin2Name1"], t["Count of Regsect"].astype(int)))
    if set(olj) != set(MINOR):
        raise SystemExit(f"L'Orient minor districts: {sorted(set(olj) ^ set(MINOR))}")
    return olj


def read_ocha():
    import openpyxl

    wb = openpyxl.load_workbook(OCHA_XLSX, read_only=True, data_only=True)
    ws = wb["ALL POPULATION SUMMARY"]
    rows = [r for r in ws.iter_rows(values_only=True)]
    hdr = [i for i, r in enumerate(rows) if r[0] == "Governorate" and r[1] == "District"]
    if len(hdr) != 1:
        raise SystemExit("OCHA package: the district header row moved")
    h = [str(x or "") for x in rows[hdr[0]]]
    col = {}
    for key, pat in (("lebanese", "TOTAL LEBANESE"), ("palestinian", "TOTAL PALESTINIANS"),
                     ("syrian", "TOTAL SYRIANS"), ("migrant", "Migrants")):
        hits = [i for i, x in enumerate(h) if x.startswith(pat) and "2026" in x]
        if len(hits) != 1:
            raise SystemExit(f"OCHA package: column {pat} (2026) found {len(hits)} times: {h}")
        col[key] = hits[0]
    recs = []
    for r in rows[hdr[0] + 1:]:
        if r[1] is None or r[0] is None:
            break
        rec = {"governorate": r[0], "caza": r[1]}
        rec.update({k: float(r[c] or 0) for k, c in col.items()})
        recs.append(rec)
    t = pd.DataFrame(recs)
    if set(t["caza"]) != set(PCODE):
        raise SystemExit(f"OCHA cazas: {sorted(set(t['caza']) ^ set(PCODE))}")
    for k, v in OCHA.items():
        got = t[k].sum()
        if abs(got - v) > 0.5:
            raise SystemExit(f"OCHA {k}: cazas sum {got:,.1f}, expected {v:,}")
    return t.set_index("caza")


def pew_rows():
    with zipfile.ZipFile(PEW) as z:
        name = [n for n in z.namelist() if n.endswith("(unrounded counts).csv")][0]
        t = pd.read_csv(io.BytesIO(z.read(name)), thousands=",")
    out = {}
    for iso, country in (("SY", "Syria"), ("PS", "Palestinian territories")):
        r = t[(t["Country"] == country) & (t["Year"] == 2020)]
        if len(r) != 1:
            raise SystemExit(f"Pew has {len(r)} {country} rows for 2020")
        out[iso] = {k: float(r[k].iloc[0]) for k in
                    ["Christians", "Muslims", "Religiously_unaffiliated", "Buddhists", "Hindus",
                     "Jews", "Other_religions"]}
    return out


def origin_mix(iso, row):
    """{node: share} for refugees from `iso`, from Pew's 2020 row. Christians are not split into
    churches and Syrians' Muslims not into sects: nothing measures either for the people in
    Lebanon (sources/lb.md §10.6). Palestinians' Muslims are Sunni, as origin_religion's default."""
    muslim = "islam.sunni" if iso == "PS" else "islam"
    node = {"Christians": "christianity", "Muslims": muslim,
            "Religiously_unaffiliated": "unaffiliated", "Buddhists": "buddhism",
            "Hindus": "hinduism", "Jews": "judaism", "Other_religions": "other.lb"}
    tot = sum(row.values())
    out = {}
    for fam, v in row.items():
        if v > 0:
            out[node[fam]] = out.get(node[fam], 0.0) + v / tot
    return out


# ---------------------------------------------------------------------------------------------
# 4. The rake
# ---------------------------------------------------------------------------------------------
def _group(fine, published):
    if fine in published:
        return fine
    if fine in ("Syriac Orthodox", "Syriac Catholic") and "Syriac" in published:
        return "Syriac"
    if "Others" in published:
        return "Others"
    return None


def ipf(seed, rows, cols, iters=5000, tol=1e-10):
    x = seed.copy()
    for _ in range(iters):
        x *= (rows / x.sum(axis=1).replace(0, np.nan)).fillna(0).to_numpy()[:, None]
        x *= (cols / x.sum(axis=0).replace(0, np.nan)).fillna(0).to_numpy()[None, :]
        err = max((x.sum(axis=1) - rows).abs().max(), (x.sum(axis=0) - cols).abs().max())
        if err < tol * rows.sum():
            return x
    raise SystemExit(f"IPF did not converge: {err}")


def build(tab, seed14, olj):
    """{COD caza: {Table-1 row: (2022 voters, tier)}}."""
    cell = {}

    def add(caza, fine, v, tier):
        c = cell.setdefault(caza, {})
        old_v, old_t = c.get(fine, (0.0, "measured"))
        c[fine] = (old_v + v, "derived" if "derived" in (old_t, tier) else "measured")

    def seed_of(minor):
        s = {}
        for u in MINOR[minor]:
            for k, v in seed14[u].items():
                s[k] = s.get(k, 0) + v
        return s

    print("\n  growth 2014 to 2022 per sect and district (seed summed against the 2022 table):")
    bad = []
    for no, (dname, minors) in DISTRICTS.items():
        pub = {k: v for k, v in tab[no].items() if k != "Total"}
        if dname.startswith("Beirut"):
            for k, v in pub.items():
                add("Beirut", k, float(v), "measured")
            continue
        # seed matrix over the district's published rows
        groups = list(pub)
        seed = pd.DataFrame(0.0, index=minors, columns=groups)
        dropped = 0
        for mnr in minors:
            for fine, v in seed_of(mnr).items():
                g = _group(fine, pub)
                if g is None:
                    dropped += v
                else:
                    seed.loc[mnr, g] += v
        ssum = seed.sum(axis=0)
        for g in groups:
            if pub[g] >= GROWTH_MIN:
                r = pub[g] / ssum[g] if ssum[g] else float("inf")
                ok = GROWTH_BAND[0] <= r <= GROWTH_BAND[1]
                pin = GROWTH_PINNED.get((dname, g))
                if pin is not None:
                    print(f"    pinned: {dname} {g} 2014 {ssum[g]:,.0f}, 2022 {pub[g]:,} ({r:.2f})")
                    if abs(r - pin) > 0.005:
                        bad.append(f"{dname} {g}: pinned {pin}, now {r:.3f}")
                elif not ok:
                    bad.append(f"{dname} {g}: 2014 {ssum[g]:,.0f}, 2022 {pub[g]:,} ({r:.2f})")
        rows = pd.Series({m: float(olj[m]) for m in minors})
        cols = pd.Series({g: float(v) for g, v in pub.items()})
        if abs(cols.sum() - rows.sum() - ROW_SUM_OFF.get(no, 0)) > 0.5:
            raise SystemExit(f"{dname}: minor districts {rows.sum():,} != table {cols.sum():,}")
        cols = cols * rows.sum() / cols.sum()      # ROW_SUM_OFF: rows scaled to the total
        for g in groups:                 # a published row the 2014 seed never saw
            if seed[g].sum() == 0:
                seed[g] = rows / rows.sum() * 1e-6
        x = ipf(seed, rows, cols) if len(minors) > 1 else pd.DataFrame(
            [cols.to_numpy()], index=minors, columns=groups)
        g_moved = sum(abs(x.loc[m].sum() - seed.loc[m].sum() / seed.values.sum() * cols.sum())
                      for m in minors) / 2 if len(minors) > 1 else 0
        print(f"    {dname:<17} {len(minors)} minor district(s); 2014 seed dropped {dropped:,.0f}"
              f" (sects the 2022 table does not print); {g_moved:,.0f} voters reallocated "
              f"between its minor districts by the 2022 totals")
        whole = len(minors) == 1 and len(MINOR[minors[0]]) == 1
        # expand: merged rows to fine rows, minor districts to cazas
        for mnr in minors:
            units = MINOR[mnr]
            for g in groups:
                v = float(x.loc[mnr, g])
                if v == 0:
                    continue
                fines = [f for f in FINE if _group(f, pub) == g]
                # split the group over (seed unit, fine row) by the 2014 seed
                parts = {(u, f): float(seed14[u].get(f, 0)) for u in units for f in fines}
                tot = sum(parts.values())
                if tot == 0:             # nothing in 2014: by the units' totals, first fine row
                    ut = {u: sum(seed14[u].values()) for u in units}
                    parts = {(u, fines[0]): ut[u] for u in units}
                    tot = sum(parts.values())
                exact = whole and fines == [g]
                for (u, f), w in parts.items():
                    if w:
                        caza = SEED_TO_CAZA.get(u, u)
                        add(caza, f, v * w / tot, "measured" if exact else "derived")
    for b in bad:
        print("  FAIL growth " + b)
    return cell, bad


def main():
    if "--fetch" in sys.argv or not os.path.exists(MONTHLY_PDF):
        fetch()
    olj = read_olj()
    print(f"L'Orient minor districts: {len(olj)}, {sum(olj.values()):,} voters")
    tab = read_monthly()
    bad, diffs = check_monthly(tab, olj)
    seed14, b2 = read_lubanan()
    bad += b2
    cell, b3 = build(tab, seed14, olj)
    bad += b3
    if bad:
        raise SystemExit(f"{len(bad)} check(s) failed")

    # the 26 cazas
    if set(cell) != set(PCODE):
        raise SystemExit(f"cazas built: {sorted(set(cell) ^ set(PCODE))}")
    m = pd.DataFrame({c: {f: cell[c].get(f, (0.0, ""))[0] for f in FINE} for c in PCODE}).T
    tier = {(c, f): cell[c][f][1] for c in cell for f in cell[c]}
    tot = m.values.sum()
    print(f"\n  26 cazas: {tot:,.1f} registered voters against {VOTERS_2022:,}")
    if abs(tot - VOTERS_2022) > 1:
        raise SystemExit("the cazas do not sum to the register")

    ocha = read_ocha()
    from afrobarometer import round_within_rows

    factor = OCHA["lebanese"] / VOTERS_2022
    leb = round_within_rows(m * factor)
    pew = pew_rows()
    recs = []
    for c in PCODE:
        for f in FINE:
            n = int(leb.loc[c, f])
            if n:
                recs.append(dict(geo_id=PCODE[c], geo_name=c, source_category=f"Lebanese, {f}",
                                 count=n, registered_2022=round(float(m.loc[c, f]), 1),
                                 tier=tier[(c, f)], basis="register, scaled"))
        for who, iso, key in (("Syrian", "SY", "syrian"), ("Palestinian", "PS", "palestinian")):
            people = ocha.loc[c, key]
            if people <= 0:
                continue
            mix = origin_mix(iso, pew[iso])
            row = round_within_rows(pd.DataFrame([{k: people * s for k, s in mix.items()}]))
            for node, n in row.iloc[0].items():
                if n:
                    recs.append(dict(geo_id=PCODE[c], geo_name=c,
                                     source_category=f"{who}, {node}", count=int(n),
                                     registered_2022=None, tier="modelled",
                                     basis="OCHA 2026 count, Pew 2020 origin mix"))
        n = int(round(ocha.loc[c, "migrant"]))
        recs.append(dict(geo_id=PCODE[c], geo_name=c, source_category="Migrants (not drawn)",
                         count=n, registered_2022=None, tier="measured",
                         basis="OCHA 2026 count"))
    out = pd.DataFrame(recs)
    out["geo_level"] = "caza"
    out["year"] = YEAR
    out["source_id"] = SOURCE_ID
    out["note"] = ("Lebanese: 2022 electoral register by caza of registration (where families are "
                   "registered), scaled to OCHA's resident Lebanese; others: OCHA 2026 by caza")
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
            "source_id", "note", "tier", "registered_2022"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[cols].to_csv(OUT + ".part", index=False, encoding="utf-8")
    os.replace(OUT + ".part", OUT)

    # the report
    lebrows = out[out["source_category"].str.startswith("Lebanese")]
    print(f"\nwrote {OUT} ({len(out)} rows)")
    print(f"  Lebanese drawn {lebrows['count'].sum():,} (factor {factor:.5f}); "
          f"measured {lebrows.loc[lebrows['tier'] == 'measured', 'count'].sum():,}, "
          f"derived {lebrows.loc[lebrows['tier'] == 'derived', 'count'].sum():,}")
    nat = m.sum(axis=0)
    for f in FINE:
        print(f"      {f:<18} {nat[f]:>11,.0f}  {nat[f] / tot:7.3%}")
    print("\n  registration against residence, per caza: 2022 register share / OCHA resident "
          "Lebanese share")
    reg = m.sum(axis=1) / tot
    res = ocha["lebanese"] / OCHA["lebanese"]
    for c in sorted(PCODE, key=lambda c: -(reg[c] / res[c])):
        print(f"      {c:<17} register {reg[c]:6.2%}  residents {res[c]:6.2%}  "
              f"{reg[c] / res[c]:5.2f}")
    gb = ["Beirut", "Baabda", "El Meten"]
    print(f"  Beirut, Baabda and El Meten: {reg[gb].sum():.1%} of the register, "
          f"{res[gb].sum():.1%} of resident Lebanese (OCHA)")
    for who in ("Syrian", "Palestinian"):
        r = out[out["source_category"].str.startswith(who)]
        print(f"  {who}s drawn {r['count'].sum():,}: " + ", ".join(
            f"{k.split(', ')[1]} {v:,}" for k, v in
            r.groupby('source_category')['count'].sum().sort_values(ascending=False).items()))


if __name__ == "__main__":
    main()
