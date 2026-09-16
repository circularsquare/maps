"""Jersey: Statistics Jersey's Opinions and Lifestyle Survey, the 2023 level with the 2018 round's
Christian churches, on the office's revised end-2023 population estimate, drawn as one unit.
2026-09-15, session cb8b206e-je.

    python sources/je.py --fetch     three reports, two forms, the 2023 results table, the
                                     population report, Kontur's JE extract
    python sources/je.py             check, normalise, build the hex layer

Writes:
    data/normalized/je.csv          five drawn shares, `Not sure`, and the population the
                                    shares are laid on; geo_level `country`
    data/geo/je/je_hexes.gpkg       Kontur 400 m hexes, unit JE (sources/micro.py::geometry)

**NO CENSUS HAS ASKED.** The 2021 census report never mentions religion, and Jersey is absent
from UNSD table 28. Statistics Jersey's annual survey asked in 2015 (as the Jersey Annual Social
Survey, JASS) and in 2018 and 2023 (renamed the Jersey Opinions and Lifestyle Survey, JOLS); the
office's contents workbook (`D-JOLS-Contents-2005-2024`, TOPICS sheet) marks religion in those
three years only. Every round asks the same two questions, "Do you regard yourself as having a
religion?" (Yes, No, Not sure) and "If yes, which?", a write-in the office codes.

**THE LEVEL IS 2023, READ FROM THE OFFICE'S OWN RESULTS TABLE.** opendata.gov.je's
`jols_2023_results.csv` gives Q17.5 as weighted proportions for all adults and by parish type:
Yes 0.39, No 0.50, Not sure 0.11. The report's Figure 9.6 prints 39/49/11, which sums to 99; the
table sums to 1.00 and is used. 1,514 respondents aged 16 and over in private households, June
and July 2023, weighted by age, sex and tenure to the 2021 census (report Annex, Table A1).

**THE CHURCHES ARE 2018's.** The 2023 report says only that 93% of those who named a religion
named Christianity or a denomination of it. The 2018 report (p.74) gives, of those who stated a
specific denomination, Catholic 50%, Church of England 39%, other 12%; they sum to 101 by
rounding and are divided by 101 here so the three keep 2023's Christian share. The 2015 report
(p.8) had Catholic 43%, Anglican 44%, other 13%. `witness()` prints both beside a third reading
built from 2023's religion by place of birth and 2015's churches by place of birth
(sources/je.md §3). Respondents who wrote plain "Christian" are split as the named ones were.

**ONE UNIT.** The results table's parish types are St Helier (`4 Urban`), St Brelade, St Clement
and St Saviour (`3 Suburban`) and the eight others (`2 Rural`). Yes reads 0.39, 0.40 and 0.38,
inside the report's ±4 to ±5 points for those groups, and nothing splits the churches by parish.
Anita's microstate ruling (2026-09-08) covers one unit.

**THE BASE IS THE END-2023 ESTIMATE, REVISED.** Statistics Jersey's *Population and migration:
Total population, December 2024* (24 September 2025) revised end-2023 from 103,650 to 104,030.
The survey's fieldwork year, as sources/ad.py chose for its survey.
"""

import csv
import gzip
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

RAW = os.path.join(ROOT, "data", "raw", "je")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

OPENDATA = ("https://opendata.gov.je/dataset/2e6e2978-0680-4b37-ada6-46586262c273/resource/")
GOVJE = "https://www.gov.je/SiteCollectionDocuments/Government%20and%20administration/"
WAYBACK = "https://web.archive.org/web/{ts}id_/" + GOVJE
FILES = {
    "jols_2023_results.csv":
        OPENDATA + "6d30dc15-4752-44e8-9983-31a1b0fe72fb/download/jols_2023_results.csv",
    "jols_2023_sectionlookup.csv":
        OPENDATA + "15e128fe-ddce-42e4-9492-6a59e09b0193/download/jols_2023_sectionlookup.csv",
    "jols_2023_report.pdf":
        GOVJE + "Opinions%20and%20Lifestyle%20Survey%202023%20Report.pdf",
    "jols_2018_report.pdf":
        GOVJE + "R%20Opinions%20and%20Lifestyle%20Survey%202018%20Report%2020181205%20SU.pdf",
    "jass_2015_report.pdf":
        GOVJE + "R%20JASS%202015%2020151202%20SU.pdf",
    # Both forms answer 404 on gov.je (2026-09-15); the Wayback Machine's copies.
    "jols_2023_questionnaire.pdf":
        WAYBACK.format(ts="20240609214940")
        + "Opinions%20and%20Lifestyle%20Survey%202023%20questionnaire.pdf",
    "jass_2015_questionnaire.pdf":
        WAYBACK.format(ts="20220120100922")
        + "F%20JASS%202015%20questionnaire%2020151201%20SU.pdf",
    "population_2024_total.pdf":
        "https://stats.je/wp-content/uploads/2025/09/"
        "R-Population-and-migration-2024-total-population-SJ20250924.pdf",
}

POP_YEAR, POP, POP_PROVISIONAL = 2023, 104_030, 103_650

# Q17.5 "Do you regard yourself as having a religion?", weighted proportions as the results
# table prints them: column -> (Yes, No, Not sure).
Q175 = {"1 Overall": (0.39, 0.50, 0.11), "2 Rural": (0.38, 0.50, 0.11),
        "3 Suburban": (0.40, 0.50, 0.10), "4 Urban": (0.39, 0.48, 0.13)}
N_2023, N_2018 = 1_514, 1_074

CHRISTIAN_2023 = 93                     # % of those who named a religion (report p.79)
CHURCHES_2018 = (50, 39, 12)            # Catholic, Church of England, other (report p.74)
CHURCHES_2015 = (43, 44, 13)            # Catholic, Anglican, other (report p.8)

# The witness (sources/je.md §3). Q1.4 place of birth, all adults 2023, from the results table.
BORN_2023 = {"Jersey": 0.45, "British Isles": 0.35, "Portugal or Madeira": 0.06,
             "Other European": 0.06, "Elsewhere": 0.09}
# % with a religion by place of birth, 2023 (report p.80, Figure 9.8, chart labels).
YES_BY_BIRTH_2023 = {"Jersey": 30, "British Isles": 38, "Portugal or Madeira": 68,
                     "Other European": 37, "Elsewhere": 73}
# Church of England or Anglican, Catholic, other, by place of birth, 2015 (report p.8, Figure
# 1.3, chart labels; each triple sums to 99-101). 2015's form had a Poland box (100% Catholic)
# and 2023's does not, so 2023's Other European includes Poles and this uses 2015's Other
# European without them, which understates Catholics.
CHURCH_BY_BIRTH_2015 = {"Jersey": (52, 34, 14), "British Isles": (58, 29, 14),
                        "Portugal or Madeira": (0, 100, 0), "Other European": (9, 37, 54),
                        "Elsewhere": (29, 65, 5)}
# A tripwire for a mistyped transcription, not a test of the witness: set at 8 points before
# the build ran, when a hand calculation put the largest gap at about 4. The build reads 5.2
# (Catholic 49.5 against 44.3).
WITNESS_TOL = 8.0


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
    for name, url in FILES.items():
        _save(url, os.path.join(RAW, name), pdf=name.endswith(".pdf"))
    gz, gpkg = micro._kontur_paths("je")
    os.makedirs(os.path.dirname(gz), exist_ok=True)
    if not os.path.exists(gz):
        _save(micro.KONTUR_URL.format(CC="JE"), gz)
    with gzip.open(gz, "rb") as src, open(gpkg + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(gpkg + ".part", gpkg)


def _raw(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run with --fetch")
    return path


def _page(name, page):
    """One PDF page's text with every run of whitespace collapsed to one space."""
    import fitz  # PyMuPDF

    doc = fitz.open(_raw(name))
    return " ".join(doc[page - 1].get_text().split())


def _need(text, pattern, what):
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"{what}: expected text not found: {pattern}")
    return m


def check_documents():
    # Each answer code is followed by a checkbox glyph (U+F0A1) in both PDFs, hence `\S?`.
    q = (r"Do you regard yourself as having a religion\? Please leave blank if you do not wish "
         r"to answer 01\S? Yes 02\S? No 03\S? Not sure {n} If yes, which\?")
    _need(_page("jols_2023_questionnaire.pdf", 24), r"17\.5 " + q.format(n=r"17\.6"),
          "2023 form Q17.5-17.6")
    _need(_page("jass_2015_questionnaire.pdf", 5), r"1\.7 " + q.format(n=r"1\.8"),
          "2015 form Q1.7-1.8")
    print("  forms 2015 and 2023: the same question, Yes / No / Not sure, then a write-in")

    p79 = _page("jols_2023_report.pdf", 79)
    got = int(_need(p79, r"majority \((\d+)%\) specified .Christian. or a denomination of "
                         r"Christianity", "2023 report p.79").group(1))
    if got != CHRISTIAN_2023:
        raise SystemExit(f"2023 report p.79: {got}% Christian, pinned {CHRISTIAN_2023}")
    _need(p79, r"39% 49% 11% Yes No Not sure", "2023 report Figure 9.6")
    fig98 = " ".join(f"{v}%" for v in YES_BY_BIRTH_2023.values())
    _need(_page("jols_2023_report.pdf", 80), re.escape(fig98), "2023 report Figure 9.8")
    _need(_page("jols_2023_report.pdf", 101), rf"Total {N_2023} 100 84,742 100",
          "2023 report Table A1")

    m = _need(_page("jols_2018_report.pdf", 74),
              r"(\d+)% specified that they were Catholic, (\d+)% specified Church of England, "
              r"and (\d+)% specified other denominations", "2018 report p.74")
    if tuple(int(x) for x in m.groups()) != CHURCHES_2018:
        raise SystemExit(f"2018 report p.74: {m.groups()}, pinned {CHURCHES_2018}")
    _need(_page("jols_2018_report.pdf", 76), rf"Total {N_2018} 100 79,806 100",
          "2018 report Table A1")

    p8 = _page("jass_2015_report.pdf", 9)          # printed page 8
    m = _need(p8, r".Catholic. or .Roman Catholic. \((\d+)%\) as were .Anglican. or .Church of "
                  r"England. \((\d+)%\)\. The remaining eighth \((\d+)%\)", "2015 report p.8")
    if tuple(int(x) for x in m.groups()) != CHURCHES_2015:
        raise SystemExit(f"2015 report p.8: {m.groups()}, pinned {CHURCHES_2015}")
    # Figure 1.3's labels run series by series, skipping empty bars: Anglican for the four
    # places with any, Catholic for all six (Poland between Portugal and Other European), then
    # other for the four.
    ce = [CHURCH_BY_BIRTH_2015[b][0] for b in CHURCH_BY_BIRTH_2015 if CHURCH_BY_BIRTH_2015[b][0]]
    ca = [CHURCH_BY_BIRTH_2015[b][1] for b in CHURCH_BY_BIRTH_2015]
    ca.insert(3, 100)                                              # Poland
    ot = [CHURCH_BY_BIRTH_2015[b][2] for b in CHURCH_BY_BIRTH_2015 if CHURCH_BY_BIRTH_2015[b][2]]
    fig13 = " ".join(f"{v}%" for v in ce + ca + ot)
    _need(p8, re.escape(fig13), "2015 report Figure 1.3")

    _need(_page("population_2024_total.pdf", 3),
          rf"The updated estimate for the population size at the end of 2023 is {POP:,}\. In "
          rf"the previous report, the 2023 population was provisionally estimated to be "
          rf"{POP_PROVISIONAL:,}", "population report p.3")
    print(f"  reports: 2023 {CHRISTIAN_2023}% Christian of those naming a religion "
          f"(n={N_2023:,}); 2018 churches {CHURCHES_2018} (n={N_2018:,}); 2015 {CHURCHES_2015}")
    print(f"  end-{POP_YEAR} population {POP:,} (revised from {POP_PROVISIONAL:,})")


def check_table():
    with open(_raw("jols_2023_results.csv"), encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    def question(qn, options):
        got = {r["Options"]: r for r in rows if r["QuestionNumber"] == qn and r["Year"] == "2023"}
        if list(got) != options:
            raise SystemExit(f"results table {qn}: options {list(got)}, expected {options}")
        return got

    rel = question("Q17.5", ["Yes", "No", "Not sure"])
    if rel["Yes"]["Question"] != "Do you regard yourself as having a religion?":
        raise SystemExit(f"results table Q17.5 is `{rel['Yes']['Question']}`")
    got = {col: tuple(float(rel[o][col]) for o in rel) for col in Q175}
    if got != Q175:
        raise SystemExit(f"results table Q17.5 changed:\n  got {got}\n  pinned {Q175}")
    born = question("Q1.4", list(BORN_2023))
    got = {o: float(born[o]["1 Overall"]) for o in born}
    if got != BORN_2023:
        raise SystemExit(f"results table Q1.4 changed: {got}")

    # The lookup is Windows-1252 (a curly apostrophe is byte 0x92); the results table is UTF-8.
    with open(_raw("jols_2023_sectionlookup.csv"), encoding="cp1252", newline="") as fh:
        look = [line for line in fh if ",Q17.5," in line]
    if len(look) != 1 or "Who asked - everyone" not in look[0]:
        raise SystemExit(f"section lookup for Q17.5: {look}")

    print("  Q17.5 by parish type        yes    no  not sure")
    for col, (y, n, s) in Q175.items():
        print(f"    {col:<24}{y:>6.2f}{n:>6.2f}{s:>9.2f}")


def witness():
    """2023 religion by place of birth x 2015 churches by place of birth, against 2018 and 2015."""
    rel = {b: BORN_2023[b] * YES_BY_BIRTH_2023[b] / 100 for b in BORN_2023}
    tot = sum(rel.values())
    raw = [sum(rel[b] * CHURCH_BY_BIRTH_2015[b][i] for b in rel) for i in (1, 0, 2)]
    wit = [100 * x / sum(raw) for x in raw]                        # Catholic, CoE, other
    s18 = [100 * x / sum(CHURCHES_2018) for x in CHURCHES_2018]
    print(f"  yes rebuilt from place of birth {tot:.3f} against the table's "
          f"{Q175['1 Overall'][0]:.2f}")
    print("  churches, % of those naming one   Catholic  Anglican  other")
    print(f"    2015 report                   {CHURCHES_2015[0]:>9}{CHURCHES_2015[1]:>10}"
          f"{CHURCHES_2015[2]:>7}")
    print(f"    2018 report (drawn)           {s18[0]:>9.1f}{s18[1]:>10.1f}{s18[2]:>7.1f}")
    print(f"    2023 by birth x 2015 churches {wit[0]:>9.1f}{wit[1]:>10.1f}{wit[2]:>7.1f}")
    worst = max(abs(a - b) for a, b in zip(s18, wit))
    if worst > WITNESS_TOL:
        raise SystemExit(f"witness is {worst:.1f} points from 2018, over {WITNESS_TOL}; check "
                         "the transcriptions before the model")


def normalise():
    yes, no, unsure = Q175["1 Overall"]
    christian = yes * CHRISTIAN_2023 / 100
    k = sum(CHURCHES_2018)
    lvl = (f"JOLS 2023 Q17.5 (opendata.gov.je results table, weighted, n={N_2023:,}, aged 16+ "
           f"in private households)")
    base = f"x {POP:,}, end-{POP_YEAR} resident population (Statistics Jersey, revised 2025)"
    def church(share):
        return (f"Yes {yes:.2f} x {CHRISTIAN_2023}% naming Christianity (2023 report p.79) x "
                f"{share}/{k} of those naming a church in JOLS 2018 (report p.74); "
                + lvl + "; " + base)

    shares = [
        ("Catholic", christian * CHURCHES_2018[0] / k, "jols2023_je+jols2018_je",
         church(CHURCHES_2018[0])),
        ("Church of England", christian * CHURCHES_2018[1] / k, "jols2023_je+jols2018_je",
         church(CHURCHES_2018[1])),
        ("Other Christian denomination", christian * CHURCHES_2018[2] / k,
         "jols2023_je+jols2018_je", church(CHURCHES_2018[2])),
        ("Religion other than Christianity", yes * (100 - CHRISTIAN_2023) / 100, "jols2023_je",
         f"Yes {yes:.2f} x {100 - CHRISTIAN_2023}% naming another religion (2023 report p.79); "
         + lvl + "; " + base),
        ("No religion", no, "jols2023_je", f"No {no:.2f}; " + lvl + "; " + base),
        ("Not sure", unsure, "jols2023_je", f"Not sure {unsure:.2f}; " + lvl + "; " + base),
    ]
    total = sum(s for _, s, _, _ in shares)
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"shares sum to {total}, not 1")

    rows = [dict(geo_id="JE", geo_level="country", geo_name="Jersey", source_category=cat,
                 count=round(share * POP, 2), basis="self_id", year=POP_YEAR, source_id=sid,
                 note=note)
            for cat, share, sid, note in shares]
    rows.append(dict(geo_id="JE", geo_level="country", geo_name="Jersey",
                     source_category=f"Resident population, end of {POP_YEAR}", count=POP,
                     basis="population", year=POP_YEAR, source_id="sj_population_2024",
                     note="Statistics Jersey, Population and migration: Total population, "
                          "December 2024 (24 September 2025), p.3: the revised end-2023 "
                          "estimate. The universe the shares are laid on, not a category."))
    out = os.path.join(ROOT, "data", "normalized", "je.csv")
    with open(out + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".part", out)
    print(f"  je.csv: {len(rows)} rows")
    for r in rows:
        print(f"    {r['source_category']:<36}{r['count']:>12,.0f}")


def main():
    if "--fetch" in sys.argv:
        fetch()
    check_documents()
    check_table()
    witness()
    normalise()
    micro.geometry("je", POP)


if __name__ == "__main__":
    main()
