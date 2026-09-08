"""Bulgaria — Census 2021 religion by obshtina, from NSI's own ethnocultural workbook.

Reads (or fetches) data/raw/bg/ and writes data/normalized/bg.csv.

**THE QUEUE HAD BULGARIA AS "hosts answer now; not chased further" AND BOTH HOSTS IT NAMED
ARE DEAD ENDS.** `nsi.bg/en/content/6704/population-religion` returns 200 with the NSI
HOMEPAGE, which is the false positive §11o's "it answers" was reading; and
`censusresults.nsi.bg` is the **2011** census portal, which publishes religion by **oblast**
and nothing finer. Neither is the route. The route is NSI's site search:

    https://www.nsi.bg/search?q=<вероизповедание>            -> the press release
    https://www.nsi.bg/statistical-data/151/1349             -> Census 2021 results, 9 xlsx
    https://www.nsi.bg/file/download/<40-hex>                -> the workbook itself

`Census2021_Ethnocultural characteristics_BG.xlsx` is 78 KB and holds four sheets. Sheet 4
is **religion by municipality**: 265 obshtini, 28 oblasti, the six NUTS aggregates and the
country, each with nine columns. That is the whole country at LAU level, which is two
administrative tiers finer than anything the 2011 portal offers.

**THE PARTITION IS EXACT AND DOUBLY WITNESSED.** The eight category columns sum to `Общо`
sums to **6,519,789**, and UNSD Demographic Yearbook table 28 carries the same census and
agrees **to the person** on every category it shares. The oracle also folds NSI's two
declining answers into one `Not Specified`: 259,235 + 472,606 = 731,841, exactly UNSD's
figure, which is what confirms the two files are the same tabulation and not two vintages.

**THE ORACLE RETURNS VALUES, NOT ONLY AN INDEX — sources.md §11r documents the wrong column
set.** §11r's `c=0,1,2,3,4,5,6` gives country/year/area and no numbers, which is why it was
only ever used to answer *"has this office ever tabulated religion"*. **`c=0,2,3,6,8,10,15,16`
returns the counts.** Column 16 is `Value`; asking for 17 or more returns a zero-byte body
with a 200, so the failure is silent and looks like a network fault. This is how the check
above was done and it is worth having for every country in the file.

**CHRISTIANITY AND ISLAM ARRIVE UNDIVIDED AND THAT IS THE COUNTRY'S ONE REAL LIMIT.** The
municipal sheet has one `Християнско` column, though the census asked more finely: NSI's own
press release breaks the national figure into Eastern Orthodox 4,091,780 (97.0% of
Christians), Protestant 69,852 (1.7%), Catholic 38,709 (0.9%), Armenian Apostolic 5,002
(0.1%) and other Christian 13,927 (0.3%). **That split is published at national level only.**
The 2011 census publishes it per oblast, and the 2011 portal also separates Sunni from Shia
(546,004 and 27,407), which matters because Bulgaria holds the largest Alevi population in
Europe outside Türkiye.

**Deriving the 2021 municipal split from the 2011 oblast one was considered and refused**, on
spec §14.4 rule 1: never estimate a magnitude a source does not publish. Spreading a national
2021 magnitude over 265 municipalities on a ten-year-old oblast shape would put Catholics
evenly across Plovdiv oblast when they are almost all in Rakovski, and nothing published at
any geography could contradict it ([[feedback_dont_draw_unsourced_breakdowns]]). It is a
genuine question rather than a closed one, and `sources/bg.md` §4 states it for Anita.

**79.3% OF THE COUNTRY IS DRAWN.** Three columns are not religions and none of them is a
residual with religions hidden in it, so unlike Slovakia's `ostatné` there is nothing here to
recover:

    Не мога да определя   259,235   4.0%   an offered box: cannot determine
    Не желая да отговоря  472,606   7.2%   an offered box: do not wish to answer
    Непоказано            616,681   9.5%   never asked — added from administrative registers

The last is North Macedonia's case exactly (`mk2021.py`, 132,260 people) and is excluded for
the same reason: it is a coverage residual, not an answer. The first two are Croatia's
`Ne izjašnjavaju se`. The religion question has been voluntary at every census since 1992.

Usage:
    python sources/bg.py --fetch    one 78 KB workbook, plus the press release PDFs
    python sources/bg.py            normalise from data/raw/bg/
"""

import csv
import os
import re
import ssl
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bg")
OUT = os.path.join(ROOT, "data", "normalized", "bg.csv")

SOURCE_ID = "bg_census_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The page that lists every Census 2021 product. fetch() scrapes it rather than hard-coding
# the download hash, because the hash is opaque and a re-publication would change it while
# the label stays put.
RESULTS_PAGE = "https://www.nsi.bg/statistical-data/151/1349"
WORKBOOK_LABEL = "Census2021_Ethnocultural characteristics_BG.xlsx"
WORKBOOK_FALLBACK = ("https://www.nsi.bg/file/download/"
                     "d6bebedae9d8dc7824e050bfc47124b402d9129b")
WORKBOOK = os.path.join(RAW, "Census2021_Ethnocultural_BG.xlsx")

# The 1 km census population grid, fetched here so sources/bg_geo.py has no network step.
GRID_URL = "https://www.nsi.bg/uploads/manager/source/GRID2021.zip"
GRID_ZIP = os.path.join(RAW, "GRID2021.zip")

# The press release, kept because it is the only publication of the national Christian
# split and the docstring above cites its figures.
PRESS_URL = "https://www.nsi.bg/file/24016/Census2021-ethnos.pdf"
PRESS_PDF = os.path.join(RAW, "Census2021-ethnos.pdf")

SHEET = "4"                      # НАСЕЛЕНИЕ ПО ВЕРОИЗПОВЕДАНИЕ
HEADER_ROW = 3                   # 0-based, the row carrying the category names
FIRST_DATA_ROW = 4

TOTAL_CAT = "Общо"
# Verbatim, footnote marker included (§2.4): the mapping keys on what this file writes.
CATEGORIES = ["Общо", "Християнско", "Мюсюлманско", "Юдейско", "Друго", "Нямам",
              "Не мога да определя", "Не желая да отговоря", "Непоказано1"]

NATIONAL = 6_519_789
EXPECTED_OBSHTINI = 265
EXPECTED_OBLASTI = 28

# NSI's own code shapes. An oblast is three letters, an obshtina three letters and two
# digits, and everything else on the sheet is a NUTS aggregate or a footnote.
OBLAST_CODE = re.compile(r"^[A-Z]{3}$")
OBSHTINA_CODE = re.compile(r"^[A-Z]{3}\d{2}$")
NUTS_CODE = re.compile(r"^BG\d*$")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept-Language": "bg,en;q=0.8"}


def _ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _get(url, timeout=240):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout, context=_ctx()).read()


def _save(url, path, what):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  have {what} ({os.path.getsize(path):,} bytes)")
        return
    body = _get(url)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(body)
    os.replace(tmp, path)          # never open the target 'w+b' — [[reference_wb_truncates]]
    print(f"  got {what} ({len(body):,} bytes) from {url}")


def _workbook_url():
    """The download link whose label is the ethnocultural workbook, off the results page."""
    try:
        page = _get(RESULTS_PAGE, timeout=90).decode("utf-8", "replace")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  !! could not read {RESULTS_PAGE} ({exc}); using the recorded hash")
        return WORKBOOK_FALLBACK
    for href, label in re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
        if WORKBOOK_LABEL in re.sub(r"<[^>]+>", "", label):
            return "https://www.nsi.bg" + href if href.startswith("/") else href
    print(f"  !! {RESULTS_PAGE} no longer labels {WORKBOOK_LABEL!r}; using the recorded hash")
    return WORKBOOK_FALLBACK


def fetch():
    os.makedirs(RAW, exist_ok=True)
    _save(_workbook_url(), WORKBOOK, "the ethnocultural workbook")
    _save(GRID_URL, GRID_ZIP, "the 1 km census population grid")
    _save(PRESS_URL, PRESS_PDF, "the ethnocultural press release")


def _rows():
    import openpyxl

    if not os.path.exists(WORKBOOK):
        raise SystemExit(f"missing {WORKBOOK} — run with --fetch first")
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    if SHEET not in wb.sheetnames:
        raise SystemExit(f"no sheet {SHEET!r} in the workbook; sheets are {wb.sheetnames}")
    return list(wb[SHEET].iter_rows(values_only=True))


def _num(v):
    """NSI writes an empty cell as '-' and a suppressed one as '..'; both are zero here."""
    if v is None:
        return 0
    s = str(v).strip().replace("\xa0", "").replace(" ", "")
    if s in ("", "-", "..", "…"):
        return 0
    return int(float(s))


def normalise():
    rows = _rows()
    header = [("" if c is None else str(c)).strip() for c in rows[HEADER_ROW]]
    got = [h for h in header if h]
    if got != CATEGORIES:
        raise SystemExit("the sheet's category headers moved — expected\n"
                         f"  {CATEGORIES}\ngot\n  {got}")
    col_of = {h: i for i, h in enumerate(header) if h}

    out, levels = [], {}
    for row in rows[FIRST_DATA_ROW:]:
        code = ("" if row[0] is None else str(row[0])).strip()
        name = ("" if row[1] is None else str(row[1])).strip()
        if not code or not name:
            continue                                    # the legend and footnote block
        if OBSHTINA_CODE.match(code):
            level = "obshtina"
        elif OBLAST_CODE.match(code):
            level = "oblast"
        elif NUTS_CODE.match(code):
            level = "country" if code == "BG" else "nuts"
        else:
            continue                                    # a footnote that starts with a digit
        levels[level] = levels.get(level, 0) + 1

        total = _num(row[col_of[TOTAL_CAT]])
        parts = 0
        for cat in CATEGORIES:
            count = _num(row[col_of[cat]])
            if cat != TOTAL_CAT:
                parts += count
            note = (f"NSI Census 2021, ethnocultural workbook sheet {SHEET}"
                    + ("; universe total, not a religion category" if cat == TOTAL_CAT
                       else ""))
            out.append(dict(geo_id=code, geo_level=level, geo_name=name,
                            source_category=cat, count=count, basis=BASIS, year=YEAR,
                            source_id=SOURCE_ID, note=note))
        # §5a: every published row is its own partition check, not just the national one.
        if parts != total:
            raise SystemExit(f"{code} {name}: categories sum to {parts:,}, "
                             f"`{TOTAL_CAT}` says {total:,}")

    _check(out, levels)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".part"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, OUT)
    print(f"  wrote {OUT} ({len(out):,} rows)")


def _check(out, levels):
    print(f"  units: " + ", ".join(f"{k} {v}" for k, v in sorted(levels.items())))
    if levels.get("obshtina") != EXPECTED_OBSHTINI:
        raise SystemExit(f"{levels.get('obshtina')} obshtini, expected {EXPECTED_OBSHTINI}")
    if levels.get("oblast") != EXPECTED_OBLASTI:
        raise SystemExit(f"{levels.get('oblast')} oblasti, expected {EXPECTED_OBLASTI}")

    def total_at(level):
        return sum(r["count"] for r in out
                   if r["geo_level"] == level and r["source_category"] == TOTAL_CAT)

    nation = total_at("country")
    if nation != NATIONAL:
        raise SystemExit(f"national total {nation:,}, expected {NATIONAL:,}")
    for level in ("oblast", "obshtina"):
        got = total_at(level)
        if got != NATIONAL:
            raise SystemExit(f"the {level} rows sum to {got:,}, not {NATIONAL:,}")
        print(f"  OK  the {level} rows sum to the national total, {got:,}")

    # The three columns that are not religions, reported rather than assumed.
    undrawn = ("Не мога да определя", "Не желая да отговоря", "Непоказано1")
    off = sum(r["count"] for r in out
              if r["geo_level"] == "country" and r["source_category"] in undrawn)
    print(f"  {NATIONAL - off:,} of {NATIONAL:,} people are drawn "
          f"({100.0 * (NATIONAL - off) / NATIONAL:.1f}%); {off:,} are in the three "
          f"non-answer columns")


def main():
    if "--fetch" in sys.argv:
        fetch()
    normalise()


if __name__ == "__main__":
    main()
