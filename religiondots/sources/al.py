"""Albania — religion by qark, 2023 census, INSTAT table 1.13.

Writes data/normalized/al.csv.

**THE HOST WAS NOT DEAD AND THE DATABASE IS ON PORT 8083.** `queue.md` closed Albania three
days running on *"`instat.gov.al` did not resolve"*; on 2026-09-08 `https://www.instat.gov.al/`
answers 200 with a 119 KB page to an ordinary browser User-Agent, and its census microsite
links `https://databaza.instat.gov.al:8083/pxweb/en/DST/`, which is a live PxWeb 21.1 with a
whole Census 2023 branch in it. Nothing about the office changed; see `sources/al.md` §1.

**RELIGION IS PUBLISHED AT QARK AND AT NOTHING SMALLER, AND THAT WAS ESTABLISHED THREE WAYS
RATHER THAN ASSUMED** (`sources/al.md` §2). The PxWeb Census 2023 branch has four folders:
national, 12 qarqe, 61 bashki and 373 njësi administrative. `Besimi fetar` appears in the
first two and in neither of the other two, whose 21 and 7 tables are age, marital status,
education, disability, household and dwelling. The published XLS set has the same shape: 33
`_qarqe` files, of which 1.13 is religion, and no religion file in the bashki series. And the
2011 census's own prefecture booklets carry a whole second section of tables *sipas
bashkisë/komunës* which likewise stops before religion.

**WHAT DOES EXIST FINER IS FOUR CATEGORIES OF THE 2011 CENSUS**, as ArcGIS feature layers in
INSTAT's own ArcGIS Online organisation: Muslim, Bektashi, Catholic and Orthodox at all 373
administrative units. It is not drawn here and `sources/al.md` §3 says why, in full, because
it is the obvious thing for the next session to reach for.

**BEKTASHI ARE COUNTED APART FROM SUNNIS AND NO OTHER SOURCE ON THIS MAP DOES THAT.** The
world headquarters of the order is in Tirana and the census gives it its own row, `Mysliman -
Bektashi`, 115,644 people.

The file INSTAT serves as `.xls` is SpreadsheetML 2003 XML and not a BIFF workbook, so xlrd
and pandas both refuse it; it is parsed as XML. One worksheet per qark, eleven rows each.

Usage:
    python sources/al.py --fetch     one ~120 KB workbook
    python sources/al.py             rebuild from data/raw/al/
"""

import csv
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "al")
XLS = os.path.join(RAW, "tab_1_13_religion_qarqe_2023.xls")
OUT = os.path.join(ROOT, "data", "normalized", "al.csv")

# INSTAT media library. The filename carries the Albanian table title; the id is stable.
URL = ("https://www.instat.gov.al/media/14411/"
       "tab_1_13_popullsia-banuese-sipas-besimit-fetar-dhe-gjinis%C3%AB_qarqe.xls")

SS = "{urn:schemas-microsoft-com:office:spreadsheet}"

# INSTAT's own qark codes, as they appear as CODE_PREFECTURE on every one of the office's
# ArcGIS layers. The key is the folded worksheet name. NOTHING HERE IS TAKEN ON TRUST:
# sources/al_geo.py checks each code against the polygon layer's own P_DISTRIB, which is the
# 2023 census population of that qark, and all twelve are distinct, so a code swapped between
# two qarqe fails on the population rather than passing quietly.
QARK_CODE = {
    "berat": ("01", "Berat"),
    "diber": ("02", "Dibër"),
    "durres": ("03", "Durrës"),
    "elbasan": ("04", "Elbasan"),
    "fier": ("05", "Fier"),
    "gjirokaster": ("06", "Gjirokastër"),
    "korce": ("07", "Korçë"),
    "kukes": ("08", "Kukës"),
    "lezhe": ("09", "Lezhë"),
    "shkoder": ("10", "Shkodër"),
    "tirane": ("11", "Tiranë"),
    "vlore": ("12", "Vlorë"),
}

# The ten row labels, in the order the worksheet prints them under `Gjithsej Total`. Matched
# on the English half of the bilingual cell, which is the stable half: the Albanian text
# carries soft line breaks and the diacritics the office's own exports mangle.
CATEGORIES = [
    ("Muslim", r"^Mysliman\b(?!.*Bektashi)"),
    ("Muslim - Bektashism", r"Bektashi"),
    ("Christian - Catholicism", r"Katolik"),
    ("Christian - Orthodoxy", r"Ortodoks"),
    ("Christian - Evangelists (Protestant)", r"Ungjillore"),
    ("Other religion or faith", r"Tjet.r besim"),
    ("Believers without denomination", r"pacil.suar"),
    ("Atheists", r"^Ateist"),
    ("Prefer not to answer", r"Preferoj"),
    ("Not available", r"Nuk disponohet"),
]

# UNSD Demographic Yearbook table 28, as INSTAT forwarded it: the 2023 national figures, used
# as an independent check on the twelve worksheets. `tools/oracle.py Albania` prints them.
NATIONAL_2023 = {
    "Muslim": 1_101_718,
    "Believers without denomination": 332_155,
    "Prefer not to answer": 244_331,
    "Christian - Catholicism": 201_530,
    "Christian - Orthodoxy": 173_645,
    "Not available": 134_451,
    "Muslim - Bektashism": 115_644,
    "Atheists": 85_311,
    "Christian - Evangelists (Protestant)": 9_658,
    "Other religion or faith": 3_670,
}
CENSUS_POPULATION_2023 = 2_402_113


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(XLS) and os.path.getsize(XLS) > 50_000:
        print("already have", XLS)
        return
    print("GET", URL)
    r = requests.get(URL, timeout=600, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if not r.content.lstrip().startswith(b"<?xml"):
        raise SystemExit("INSTAT served something that is not SpreadsheetML -- the media id "
                         "may have moved; look for tab 1.13 on the census theme page")
    with open(XLS, "wb") as fh:
        fh.write(r.content)
    print(f"  {len(r.content):,} bytes")


def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def celltext(cell):
    """A cell's text, including the styled runs INSTAT wraps its labels in.

    Every label cell is `<ss:Data>` with an html40 namespace and `<B>`/`<I>` children holding
    the Albanian and English halves, so `.text` on the Data element is None and a parser that
    reads it sees a table of numbers with no row names at all.
    """
    out = []
    for d in cell:
        if d.tag.endswith("Data"):
            out.append("".join(d.itertext()))
    return " ".join(" ".join(out).split())


def sheets():
    """{worksheet name: [[cell, ...], ...]} for the workbook."""
    if not os.path.exists(XLS):
        raise SystemExit(f"missing {XLS} -- run with --fetch first")
    tree = ET.parse(XLS)
    out = {}
    for ws in tree.getroot().findall(SS + "Worksheet"):
        table = ws.find(SS + "Table")
        if table is None:
            continue
        out[ws.get(SS + "Name")] = [[celltext(c) for c in row.findall(SS + "Cell")]
                                    for row in table.findall(SS + "Row")]
    return out


def read_sheet(name, rows):
    """(printed total, {category: count}) for one qark's worksheet.

    The anchor is the row whose first cell is `Gjithsej Total` and which carries three
    numbers; the ten category rows follow it in the printed order. Each is matched to a
    category by REGEX ON ITS OWN LABEL rather than by position, so a re-ordered or extended
    release fails loudly here instead of shifting every count by one.
    """
    body = [r for r in rows if len(r) >= 2 and re.fullmatch(r"-?[\d ,]+", r[1] or "")]
    if len(body) != 11:
        raise SystemExit(f"{name}: {len(body)} numeric rows, expected 11 "
                         "(a Total and ten categories)")
    head, cats = body[0], body[1:]
    if not head[0].lower().startswith("gjithsej"):
        raise SystemExit(f"{name}: first numeric row is {head[0]!r}, expected the Total")
    total = int(head[1].replace(" ", "").replace(",", ""))

    got = {}
    for label, pat in CATEGORIES:
        hits = [r for r in cats if re.search(pat, r[0])]
        if len(hits) != 1:
            raise SystemExit(f"{name}: {len(hits)} rows match {label!r} ({pat}); "
                             f"labels are {[r[0][:40] for r in cats]}")
        got[label] = int(hits[0][1].replace(" ", "").replace(",", ""))
    if len(got) != len(cats):
        raise SystemExit(f"{name}: {len(cats)} category rows but {len(got)} matched")

    s = sum(got.values())
    if s != total:
        raise SystemExit(f"{name}: the ten categories sum to {s:,}, the printed Total is "
                         f"{total:,}")
    return total, got


def main():
    if "--fetch" in sys.argv:
        fetch()

    data = sheets()
    if len(data) != 12:
        raise SystemExit(f"{len(data)} worksheets, expected one per qark")

    rows, report, national = [], [], {}
    for sheet, table in data.items():
        key = fold(sheet)
        if key not in QARK_CODE:
            raise SystemExit(f"worksheet {sheet!r} is not one of the twelve qarqe")
        code, qname = QARK_CODE[key]
        total, got = read_sheet(sheet, table)
        for cat, n in got.items():
            national[cat] = national.get(cat, 0) + n
            rows.append(dict(
                geo_id=code, geo_level="qark", geo_name=qname,
                source_category=cat, count=n, basis="self_id", year=2023,
                source_id="al_phc_2023_t113",
                note=f"qark={qname}; worksheet={sheet}; INSTAT table 1.13"))
        report.append((code, qname, total, got))
    if len({r[0] for r in report}) != 12:
        raise SystemExit("two worksheets claimed the same qark code")

    # ---- the check: the twelve worksheets against UNSD table 28, to the person ----
    print(f"{'code':>4}  {'qark':<12}{'population':>11}  {'Muslim':>9}{'Bektashi':>9}"
          f"{'Catholic':>9}{'Orthodox':>9}{'no denom':>9}{'atheist':>8}{'undecl.':>9}")
    for code, qname, total, got in sorted(report):
        undecl = got["Prefer not to answer"] + got["Not available"]
        print(f"{code:>4}  {qname:<12}{total:>11,}  "
              f"{got['Muslim']:>9,}{got['Muslim - Bektashism']:>9,}"
              f"{got['Christian - Catholicism']:>9,}{got['Christian - Orthodoxy']:>9,}"
              f"{got['Believers without denomination']:>9,}{got['Atheists']:>8,}"
              f"{undecl:>9,}")

    print("\n  summed over the twelve worksheets, against UNSD table 28's national row:")
    bad = []
    for cat, nat in sorted(NATIONAL_2023.items(), key=lambda kv: -kv[1]):
        got = national.get(cat, 0)
        flag = "" if got == nat else "   <-- DISAGREES"
        print(f"    {cat:<40}{got:>10,}{nat:>10,}{flag}")
        if got != nat:
            bad.append(cat)
    tot = sum(national.values())
    print(f"    {'TOTAL':<40}{tot:>10,}{CENSUS_POPULATION_2023:>10,}")
    if bad or tot != CENSUS_POPULATION_2023:
        raise SystemExit(f"the qark file and the Yearbook disagree on {bad or 'the total'} "
                         "-- do not write a file that fails this")
    print("    every category and the total agree to the person.")

    gap = national["Prefer not to answer"] + national["Not available"]
    print(f"\n  §3.5 residual: {national['Prefer not to answer']:,} preferred not to answer "
          f"and {national['Not available']:,} are `not available`,\n  together {gap:,}, "
          f"{gap / CENSUS_POPULATION_2023:.2%} of the census.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["geo_id", "geo_level", "geo_name", "source_category", "count", "basis",
            "year", "source_id", "note"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows):,} rows, 12 qarqe x 10 categories)")


if __name__ == "__main__":
    main()
