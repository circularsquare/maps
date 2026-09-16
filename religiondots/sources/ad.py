"""Andorra: World Values Survey wave 7 (2018) shares, on the Department of Statistics' 2018
population estimate, drawn as one unit. 2026-09-15, session cb8b206e-ad.

    python sources/ad.py --fetch     IHSN variable pages and documents, the population series,
                                     Kontur's AD extract
    python sources/ad.py             normalise + build the hex layer

Writes:
    data/normalized/ad.csv          one row per Q289 answer, geo_level `country`,
                                    count = respondents / 1,004 x 76,177
    data/geo/ad/ad_hexes.gpkg       Kontur 400 m hexes, unit AD (sources/micro.py::geometry)

**NO CENSUS HAS EVER ASKED, AND THE SURVEY FILE IS BEHIND A FORM, BUT ITS FREQUENCIES ARE NOT.**
The WVS-7 Andorra data file needs the WVS download form. The International Household Survey
Network's catalogue entry for the same file (`catalog.ihsn.org/catalog/11550`,
`AND_2018_WVS-W7_v01_M`, data file `WVS_Wave_7_Andorra_Stata_v5.0`) prints every variable's
unweighted category counts with no form and no login. For a country drawn as one unit that is
the whole source. Read here: `Q289` (the card), `Q289CS9` (the archive's detailed code, checked
one to one against `Q289`), `W_WEIGHT` (one value for all 1,004) and `N_REGION_ISO` (interviews
per parish, printed, not drawn).

**UNWEIGHTED IS THE SURVEY TEAM'S OWN ESTIMATE.** The sample design note (F00010377) says no
weighting was applied because quotas held sex, age and nationality to the population, and the
methodology report (F00008607, Q39) prints the sample against the population: Andorran 38.97%
of the population against 39.1% of the sample, Spanish 30.00 / 30.2, Portuguese 14.49 / 15.3,
French 5.87 / 5.4, other 10.67 / 10.7.

**THE POPULATION BASE IS 2018, THE FIELDWORK YEAR** (6 January to 22 September 2018, IHSN study
description). The Department of Statistics' estimated resident population was 76,177 in 2018
and 89,058 in 2025. The series' 2024 value equals the year-end figure El Periodic d'Andorra
reported from the office ("L'any 2024 es va tancar amb una poblacio resident integrada per
87.097 persones"), so the series reads as 31 December; 2025 was not checked the same way. The growth since 2018 is mostly the `other
nationalities` group (10.7% of the 2018 sample's quota, 16.9% of residents in February 2025),
people the survey's quotas barely reached, so a later base would lay 2018 shares on them. The
same reasoning as sources/bq.py. The series is read off the Observatori Social's PDF, which
cites the Department.

**PARISHES ARE NOT DRAWN.** The file has all seven parishes, but the catalogue gives no
religion by parish, and the interviews are not spread as the population is: La Massana and
Ordino are 8.2% of the interviews and 20.0% of residents in the office's February 2025 note,
Encamp and Escaldes-Engordany 42.4% against 33.2%. The design note says the allocation was
proportional; the file does not show it. Anita's microstate ruling (2026-09-08) covers one
unit; sources/ad.md §4 has the reasoning.
"""

import csv
import gzip
import html
import os
import re
import shutil
import ssl
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import micro  # noqa: E402  (Kontur paths and the one-unit hex layer)

RAW = os.path.join(ROOT, "data", "raw", "ad")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

IHSN_VAR = "https://catalog.ihsn.org/catalog/11550/variable/F1/{vid}?name={name}"
PAGES = {"Q289": "V338", "Q289CS9": "V339", "W_WEIGHT": "V30", "N_REGION_ISO": "V16"}
IHSN_DOC = "https://catalog.ihsn.org/catalog/11550/download/{did}"
DOCS = {                                   # saved for the record; not parsed
    "F00006597-WVS7_Questionnaire_Andorra_2018_Catalan.pdf": 101859,
    "F00006598-WVS7_Questionnaire_Andorra_2018_English.pdf": 101860,
    "F00008607-WVS7_Methodology_Report_Andorra_2018.pdf": 101863,
    "F00010377-WVS7_Sample_Design_Andorra_2018.pdf": 101865,
}
POP_URL = ("https://observatorisocial.ad/files/153/Poblacio-dAndorra/2021/"
           "1-Evolucio-de-la-poblacio-d-%27Andorra-2025.pdf")
POP_PDF = os.path.join(RAW, "observatori_evolucio_poblacio_2025.pdf")

N = 1_004
POP_YEAR, POP = 2018, 76_177
# Two later years of the same series, pinned because they equal the office's December
# announcements and so date the series to 31 December.
POP_WITNESS = {2024: 87_097, 2025: 89_058}

# Q289 as the catalogue prints it: code -> (value label without its {short form}, respondents).
Q289 = {
    -2: ("No answer/refused", 2),
    0: ("Do not belong to a denomination", 302),
    1: ("Catholic (Roman/Greek/etc)", 641),
    2: ("Protestant", 10),
    3: ("Orthodox (Russian/Greek/etc.)", 17),
    4: ("Jew", 1),
    5: ("Muslim", 11),
    6: ("Hindu", 9),
    7: ("Buddhist", 6),
    9: ("Other", 5),
}
# Q289 code -> the archive's Q289CS9 code for the same answer. One to one in this file.
CS9_OF = {-2: -2, 0: 100000020, 1: 10100000, 2: 20000000, 3: 30100000, 4: 40000000,
          5: 50000000, 6: 60000000, 7: 70000000, 9: 90000000}

# N_REGION_ISO: code -> respondents. Printed beside the office's February 2025 estimate
# (NP_A001_A003_20250313, table 1.1), which is the nearest parish table read, not 2018's.
PARISH_SAMPLE = {20002: 63, 20003: 197, 20004: 51, 20005: 31, 20006: 138, 20007: 295,
                 20008: 229}
PARISH_POP_2025_02 = {20002: ("Canillo", 6_107), 20003: ("Encamp", 13_255),
                      20004: ("La Massana", 11_896), 20005: ("Ordino", 5_571),
                      20006: ("Sant Julià de Lòria", 10_158),
                      20007: ("Andorra la Vella", 24_698),
                      20008: ("Escaldes-Engordany", 15_801)}

ROW = re.compile(r"<tr>\s*<td>(-?\d+)</td>\s*<td>([^<]*)</td>\s*<td>(\d+)</td>", re.S)
TITLE = re.compile(r"<h2>([^<]*)\((\w+)\)</h2>")


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _save(url, dest, pdf=False):
    body = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180,
                                  context=_ctx()).read()
    if pdf and not body.startswith(b"%PDF-"):
        raise SystemExit(f"{url}: not a PDF ({body[:40]!r})")
    with open(dest + ".part", "wb") as fh:
        fh.write(body)
    os.replace(dest + ".part", dest)
    print(f"  got {os.path.basename(dest)} ({len(body):,} bytes)")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for name, vid in PAGES.items():
        _save(IHSN_VAR.format(vid=vid, name=name), os.path.join(RAW, f"ihsn_11550_{name}.html"))
    for fname, did in DOCS.items():
        _save(IHSN_DOC.format(did=did), os.path.join(RAW, fname), pdf=True)
    _save(POP_URL, POP_PDF, pdf=True)

    gz, gpkg = micro._kontur_paths("ad")
    os.makedirs(os.path.dirname(gz), exist_ok=True)
    if not os.path.exists(gz):
        _save(micro.KONTUR_URL.format(CC="AD"), gz)
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)


def read_page(name):
    """One IHSN variable page -> {code: (label, respondents)}, with the page's identity asserted."""
    path = os.path.join(RAW, f"ihsn_11550_{name}.html")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch")
    text = open(path, encoding="utf-8").read()
    m = TITLE.search(text)
    if not m or m.group(2) != name:
        raise SystemExit(f"{path}: the page is for {m.group(2) if m else 'no variable'}, "
                         f"expected {name}")
    if "WVS_Wave_7_Andorra" not in text:
        raise SystemExit(f"{path}: not the WVS-7 Andorra data file")
    rows = {}
    for code, label, n in ROW.findall(text):
        label = re.sub(r"\{[^}]*\}", "", html.unescape(label)).strip()
        rows[int(code)] = (label, int(n))
    if not rows:
        raise SystemExit(f"{path}: no category table found")
    return rows


def check_survey():
    q = read_page("Q289")
    if q != Q289:
        raise SystemExit(f"Q289 changed on the catalogue page:\n  got {q}\n  pinned {Q289}")
    if sum(n for _, n in q.values()) != N:
        raise SystemExit(f"Q289 sums to {sum(n for _, n in q.values())}, expected {N}")

    cs = read_page("Q289CS9")
    if len(cs) != len(Q289):
        raise SystemExit(f"Q289CS9 has {len(cs)} codes, Q289 {len(Q289)}")
    for code, cs_code in CS9_OF.items():
        if cs.get(cs_code, (None, None))[1] != Q289[code][1]:
            raise SystemExit(f"Q289 {code} ({Q289[code][1]}) against Q289CS9 {cs_code} "
                             f"({cs.get(cs_code)})")
    print(f"  Q289 = Q289CS9 one to one over {len(Q289)} codes; Other is "
          f"`{cs[90000000][0]}`, Catholic `{cs[10100000][0]}`")

    w = read_page("W_WEIGHT")
    if w != {1: ("No weighting", N)}:
        raise SystemExit(f"W_WEIGHT is not one constant: {w}")
    print(f"  W_WEIGHT: one value, `No weighting`, for all {N:,}")

    p = read_page("N_REGION_ISO")
    got = {c: n for c, (_, n) in p.items()}
    if got != PARISH_SAMPLE:
        raise SystemExit(f"N_REGION_ISO changed: {got}")
    pop_total = sum(v for _, v in PARISH_POP_2025_02.values())
    print("  parish                 interviews  share   residents Feb 2025  share")
    for code, (pname, pop) in PARISH_POP_2025_02.items():
        n = PARISH_SAMPLE[code]
        print(f"    {pname:<22}{n:>6}  {100 * n / N:5.1f}%   {pop:>10,}        "
              f"{100 * pop / pop_total:5.1f}%")


def check_population():
    import fitz  # PyMuPDF

    if not os.path.exists(POP_PDF):
        raise SystemExit(f"missing {POP_PDF}; run with --fetch")
    text = "\n".join(page.get_text() for page in fitz.open(POP_PDF))
    for year, want in {POP_YEAR: POP, **POP_WITNESS}.items():
        m = re.search(rf"\b{year}\s+(\d{{2}}\.\d{{3}})", text)
        got = int(m.group(1).replace(".", "")) if m else None
        if got != want:
            raise SystemExit(f"{os.path.basename(POP_PDF)}: {year} reads {got}, pinned {want:,}")
    print(f"  estimated resident population {POP_YEAR}: {POP:,} "
          f"(2024 {POP_WITNESS[2024]:,}, 2025 {POP_WITNESS[2025]:,})")


def normalise():
    check_survey()
    check_population()
    rows = []
    for code, (label, n) in sorted(Q289.items(), key=lambda kv: -kv[1][1]):
        rows.append(dict(
            geo_id="AD", geo_level="country", geo_name="Andorra", source_category=label,
            count=round(n / N * POP, 2), basis="self_id", year=2018, source_id="wvs7_ad_2018",
            note=f"{n} of {N:,} respondents aged 18+, unweighted, WVS-7 Q289 code {code} "
                 f"(IHSN catalogue 11550); x {POP:,}, estimated resident population "
                 f"{POP_YEAR} (Departament d'Estadistica)"))
    out = os.path.join(ROOT, "data", "normalized", "ad.csv")
    with open(out + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".part", out)
    total = sum(r["count"] for r in rows)
    print(f"  ad.csv: {len(rows)} rows, {total:,.0f} people "
          f"(Catholic {100 * 641 / N:.1f}%, none {100 * 302 / N:.1f}%)")


def main():
    if "--fetch" in sys.argv:
        fetch()
    normalise()
    micro.geometry("ad", POP)


if __name__ == "__main__":
    main()
