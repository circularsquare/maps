"""Saint Lucia — CSO, 2022 Population and Housing Census, Table D.2.

Reads (or fetches) data/raw/lc/ and writes data/normalized/lc.csv.

**Twenty-three categories on 10 districts, for 171,834 people** — ~17,200 per unit, and the
list is the census's own answer set rather than a classification: `sources/lc.md` §2.

**THE `Mennonite` ROW IS ALMOST CERTAINLY `Evangelical`, AND THAT IS 2.2% OF THE COUNTRY.**
This file does NOT rename it — `lc.csv` carries the label the report prints, because a
normalised file is a record of what the source says (§12). The evidence, and the decision to
resolve it to `christianity.evangelical` rather than to a Mennonite node, live in
`taxonomy/lc2022.py`. Three things establish it and `check()` asserts the first:

  * **The census's own questionnaire.** CSO publishes the Survey Solutions instrument —
    *St Lucia Census 2022, Version 4* — and its question 1.5 offers 22 options. Table D.2's
    23 rows are those 22 options **in the same order**, plus `Not reported`. Twenty-two of
    the twenty-three match one for one. The one that does not is **option 6, which the
    questionnaire calls `Evangelical` and the report calls `Mennonite`**.
  * **The 2010 census.** Its Table 40 has `Evangelical` at **2.2%** — the same share the
    2022 report gives `Mennonite` — in the same position, and no Mennonite row at all.
  * There is no Mennonite community of 3,760 people in Saint Lucia.

`--fetch` therefore downloads the questionnaire as well as the report, and the check runs on
every invocation, exactly as `sources/bb.py` downloads Barbados's rejected 2021 workbook in
order to read its 2010 one.

**THE DRAWN TABLE IS THE SAME IN BOTH LIVE REVISIONS.** CSO's own site serves *Release 2
Rev 2.5*; the CARICOM mirror (`sources.md` §11v) serves *Rev 2.7*, which is newer than
anything the publisher lists. Their Table D.2 pages are **textually identical** — the two
revisions differ on the employment chapter and on where Table 14 breaks across a page — so
the choice of revision cannot affect the map. This file reads the publisher's own copy and
downloads the mirror's to prove that sentence rather than assert it.

**THE UNIVERSE IS THE HOUSEHOLD POPULATION, 171,834**, which is not the country:

    171,834   in private households   <- every table here, including D.2
      1,114   in institutions (194 hospitals, 684 prisons, 236 other)
    172,948   total RESIDENT population
      5,859   visitors in hotels and guesthouses
    178,807   total population on census night
    182,289   mid-year estimate for 2022, which the report contrasts with its own count

Nothing here is scaled up to any of those (§14.4).

Usage:
    python sources/lc.py --fetch    three PDFs, ~11.9 MB, seconds
    python sources/lc.py            normalise from data/raw/lc/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lc")
OUT = os.path.join(ROOT, "data", "normalized", "lc.csv")

SOURCE_ID = "lc_phc_2022_r2"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# CSO's own host. Release 2, Rev 2.5 -- the newest revision stats.gov.lc serves.
REPORT_URL = ("https://www.stats.gov.lc/wp-content/uploads/2024/08/"
              "StLucia-Provisional-Census-Report2022-Release-2Rev-2.5.pdf")
REPORT_NAME = "lc_provisional_census_2022_r2rev2.5.pdf"

# The census instrument, also CSO's own. NOT a source of counts -- read only to
# establish what question 1.5's option 6 actually was. See the module docstring.
FORM_URL = "https://www.stats.gov.lc/wp-content/uploads/2024/11/St-Lucia-Census-2022.pdf"
FORM_NAME = "lc_census_2022_questionnaire.pdf"

# The CARICOM mirror's Rev 2.7. NOT a source -- read only to show that the drawn table
# does not move between revisions.
MIRROR_URL = ("https://statistics.caricom.org/wp-content/uploads/2025/11/"
              "StLucia-Provisional-Census-Report2022-Release-2Rev-2.7.pdf")
MIRROR_NAME = "lc_provisional_census_2022_r2rev2.7.pdf"

# Two needles, not one: the caption alone also matches the table of contents, and a
# contents page carries no figures at all, so a caption-only search silently reads an
# empty page (§12, Chile -- a read that succeeds is not a read that returned data).
D2_NEEDLES = ("Table D.2 Population: Religion by District", "Roman Catholic")
A2_NEEDLES = ("Table A.2 Household Population by District", "Total resident population")

# The one word on the Table D.2 page that appears exactly once and sits on the caption
# line. Everything above it -- the running page header, the page number -- is not part
# of the table, and lands in whichever column it happens to be over.
CAPTION_TOKEN = "D.2"

# The group heading printed once, centred over the ten district columns. It belongs to
# none of them, so it is dropped rather than filed under whichever one it sits above.
SPANNER = "District"

# Everything left of this is the stub column; everything right of it is a figure. The
# `-` in `Atheist - Do not believe in God` is why the split cannot be by token shape.
VALUE_X0 = 200.0
NUM = re.compile(r"^(?:-|–|[\d,]+)$")

# In sheet order, top to bottom. Position 5 (0-based) is the one the questionnaire
# calls `Evangelical`; see the module docstring and taxonomy/lc2022.py.
CATEGORIES = [
    "Anglican",
    "Baptist",
    "Bahai Faith",
    "Brethren",
    "Buddhism",
    "Mennonite",                     # THE MISLABELLED ROW. Questionnaire: `Evangelical`.
    "Hindu",
    "Jehovah Witnesses",
    "Methodist",
    "Mormon",
    "Islam",
    "Pentecostal",
    "Nazarene",
    "Rastafarian",
    "Roman Catholic",
    "Salvation Army",
    "Seventh Day Adventist",
    "Universal Church",
    "Hinduism",
    "Atheist - Do not believe in God",
    "None - No religion but believe in God",
    "Other",
    "Not reported",
]
MISLABELLED = "Mennonite"
QUESTIONNAIRE_SAYS = "Evangelical"
TOTAL_CAT = "Total"

# Question 1.5's options, in the questionnaire's own numbering. Asserted against
# CATEGORIES in confirm_questionnaire(); the ONE disagreement is the finding.
FORM_OPTIONS = [
    "Anglican", "Baptist", "Bahai Faith", "Brethren", "Buddhism", "Evangelical",
    "Hindu", "Jehovah's Witnesses", "Methodist", "Mormon", "Islam (Muslim)",
    "Pentecostal", "Nazarene", "Rastafarian", "Roman Catholic", "Salvation Army",
    "Seventh-Day Adventist", "Universal Church", "Hinduism",
    "Atheist - Do not believe in God", "None - No Religion but believe in God", "Other",
]
# Spelling only, and held explicitly rather than solved by fuzzy matching: the point of
# the check is that the two lists are the SAME list, so anything that would let two
# different lists pass defeats it (§12).
FORM_SPELLINGS = {
    "Jehovah's Witnesses": "Jehovah Witnesses",
    "Islam (Muslim)": "Islam",
    "Seventh-Day Adventist": "Seventh Day Adventist",
    "None - No Religion but believe in God": "None - No religion but believe in God",
}

# The ten districts, left to right as Table D.2 prints them, with the pcode COD-AB uses.
# Asserted from the boundary side in sources/lc_geo.py -- CSO publishes no code of its
# own, and COD's numbering is neither alphabetical nor the census's order.
DISTRICTS = [
    ("Castries",     "LC13"),
    ("Anse La Raye", "LC04"),
    ("Canaries",     "LC05"),
    ("Soufriere",    "LC06"),
    ("Choiseul",     "LC07"),
    ("Laborie",      "LC08"),
    ("Vieux Fort",   "LC09"),
    ("Micoud",       "LC10"),
    ("Dennery",      "LC11"),
    ("Gros Islet",   "LC12"),
]

HOUSEHOLD_POPULATION = 171_834     # Table A.1, `In private households`
INSTITUTIONAL = 1_114              # not published as a line; the three below sum to it
RESIDENT_POPULATION = 172_948
VISITORS = 5_859
TOTAL_POPULATION = 178_807
MIDYEAR_ESTIMATE_2022 = 182_289    # narrative, p.12; not a census count

# Table A.1's own rows, asserted rather than transcribed. `In private households` is
# the universe of every other table in the report, Table D.2 included.
A1_LADDER = {
    "In private households": HOUSEHOLD_POPULATION,
    "In public hospitals, Mental homes,": 194,
    "In prisons": 684,
    "Other institutional": 236,
    "Total resident population": RESIDENT_POPULATION,
    "Total population": TOTAL_POPULATION,
}

# **SAINT LUCIA'S FIGURES ARE ALREADY WEIGHTED UP FOR THE UNDERCOUNT, AND THAT IS THE
# PUBLISHER'S DECISION RATHER THAN THIS MAP'S.** The report, §*Reading these tables*:
#
#   *"the Household Population is the sum of all people recorded on visitation records
#   and/or electronic census questionnaires, ON WHICH A SET OF GEOGRAPHICALLY DEPENDENT
#   WEIGHT FACTORS HAS BEEN APPLIED to arrive at estimated full values"*
#
# and Table 2 prints the factors. **The national undercount was 23.3%** and the
# per-district weights run 1.107 (Anse La Raye) to 1.507 (Laborie) — so 171,834 is an
# estimate of Saint Lucia, not a count of it, and the district cells were scaled by
# different amounts before publication.
#
# This is the exact inverse of Barbados (§9au), where BSS publishes the raw tabulable
# count, warns that its area tables are understated, and this project declines to scale
# them (§14.4). Nothing here scales anything either — the difference is entirely on the
# publisher's side of the line, and it is why `Estimated Household Population` below
# equals Table A.2's and Table D.2's district totals to the person.
UNDERCOUNT_WEIGHTS = {
    "Anse La Raye": (9.7, 1.107, 5_841),
    "Canaries": (20.8, 1.263, 2_171),
    "Castries": (25.2, 1.337, 60_614),
    "Choiseul": (25.3, 1.338, 7_122),
    "Dennery": (19.2, 1.238, 12_943),
    "Gros Islet": (18.8, 1.231, 29_953),
    "Laborie": (33.7, 1.507, 8_507),
    "Micoud": (30.9, 1.446, 16_693),
    "Soufriere": (16.3, 1.195, 8_322),
    "Vieux Fort": (22.4, 1.288, 19_669),
}
NATIONAL_UNDERCOUNT_PCT = 23.3
NATIONAL_WEIGHT = 1.304

# **TABLE D.2 DOES NOT INTERNALLY RECONCILE, AND THE WEIGHTING IS WHY.** Measured, not
# assumed, and check() prints the whole spread:
#
#   * a category row's ten districts miss the row's own Total by -2 to +2;
#   * a district column's categories miss the column's own Total by -3 to +3;
#   * and every drawn cell added together is 171,829 against a published 171,834 —
#     **five people in 0.003%**.
#
# Two-sided, single digits, on weighted estimates that were each rounded on their own.
# That is what independent rounding looks like and is nothing a parse error looks like:
# a dropped row or a mis-assigned column would be out by hundreds.
ROW_TOL = 2
COL_TOL = 3
GRAND_TOL = 5


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, name, floor in ((REPORT_URL, REPORT_NAME, 4_000_000),
                             (FORM_URL, FORM_NAME, 500_000),
                             (MIRROR_URL, MIRROR_NAME, 4_000_000)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > floor:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, timeout=900, stream=True, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(dest + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(dest + ".part", dest)
        size = os.path.getsize(dest)
        # §5a: a 200 is not a download, and a PDF can be truncated at source with a
        # Content-Length that agrees with the damage -- so check both ends of the file.
        with open(dest, "rb") as fh:
            head = fh.read(5)
            fh.seek(max(0, size - 4096))
            tail = fh.read()
        if head != b"%PDF-":
            raise SystemExit(f"{dest} is not a PDF -- starts {head!r}")
        if b"%%EOF" not in tail:
            raise SystemExit(f"{dest} has no %%EOF trailer -- truncated at source")
        print(f"  {size:,} bytes")


def _open(name):
    import fitz

    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    return fitz.open(path)


def _page(doc, needles):
    hits = [i for i in range(doc.page_count)
            if all(n in doc[i].get_text() for n in needles)]
    if len(hits) != 1:
        raise SystemExit(f"{needles[0]!r} matched {len(hits)} pages ({hits}); the report "
                         "has been re-laid out and this file must be rechecked against it")
    return doc[hits[0]]


def _grid(page, ncols):
    """Numeric tokens on `page`, grouped into rows of exactly `ncols`.

    Returns (rows, rejected). A row is a list of (x0, text) left to right. Anything
    that is not `ncols` wide -- the running header's `- 2022`, the page number -- is
    rejected, and `check()` proves nothing real was rejected by summing the columns.
    """
    toks = [(x0, (y0 + y1) / 2, w) for x0, y0, x1, y1, w, *_ in page.get_text("words")
            if x0 >= VALUE_X0 and NUM.match(w)]
    rows = []
    for x0, mid, w in sorted(toks, key=lambda t: (t[1], t[0])):
        for r in rows:
            if abs(r[0] - mid) <= 4.0:
                r[1].append((x0, w))
                break
        else:
            rows.append((mid, [(x0, w)]))
    rows.sort()
    full = [(mid, sorted(ts)) for mid, ts in rows if len(ts) == ncols]
    rejected = [(mid, sorted(ts)) for mid, ts in rows if len(ts) != ncols]
    return full, rejected


def _cell(w):
    return 0 if w in ("-", "–") else int(w.replace(",", ""))


def _stub(page, y_floor):
    """The label column, in reading order, from `y_floor` down."""
    ws = [((y0 + y1) / 2, x0, w) for x0, y0, x1, y1, w, *_ in page.get_text("words")
          if x0 < VALUE_X0 and (y0 + y1) / 2 >= y_floor]
    return [w for _, _, w in sorted(ws)]


def _squash(parts):
    """Join and drop every space, so a label that WRAPS still compares equal.

    `Atheist - Do not believe in God` is printed on two lines and `Seventh Day
    Adventist` on one; squashing makes the comparison independent of where the
    typesetter broke each one, while still being exact about the sequence.
    """
    return "".join("".join(str(p).split()) for p in parts)


def confirm_questionnaire():
    """Prove that Table D.2's rows ARE question 1.5's options, and find the one that is not.

    This is the whole reason the questionnaire is downloaded.
    """
    doc = _open(FORM_NAME)
    text = " ".join(doc[i].get_text() for i in range(doc.page_count))
    doc.close()
    text = " ".join(text.split())

    if "St Lucia Census 2022" not in text:
        raise SystemExit(f"{FORM_NAME} does not identify itself as the Saint Lucia 2022 "
                         "instrument -- do not use it as evidence for anything")

    m = re.search(r"p1_5:.*?Categories:(.*?)\[\d+\]", text)
    if not m:
        raise SystemExit("no `p1_5: ... Categories:` block in the questionnaire -- the "
                         "religion question's option list cannot be read, so the "
                         "Mennonite/Evangelical finding is UNSUPPORTED. Stop and re-check "
                         "sources/lc.md §2 by hand rather than trusting lc2022.py.")
    opts = [re.sub(r"^\d+\s*:\s*", "", o.strip())
            for o in m.group(1).split(",") if o.strip()]
    if opts != FORM_OPTIONS:
        raise SystemExit("question 1.5's option list has changed:\n"
                         f"  got  {opts}\n  want {FORM_OPTIONS}")

    normalised = [FORM_SPELLINGS.get(o, o) for o in opts]
    printed = [c for c in CATEGORIES if c != "Not reported"]
    if len(normalised) != len(printed):
        raise SystemExit(f"{len(normalised)} questionnaire options against "
                         f"{len(printed)} printed rows")
    disagree = [(i + 1, a, b) for i, (a, b) in enumerate(zip(normalised, printed))
                if a != b]
    if disagree != [(6, QUESTIONNAIRE_SAYS, MISLABELLED)]:
        raise SystemExit(
            "the questionnaire and the report no longer disagree in exactly one place, "
            f"which is what taxonomy/lc2022.py's `{MISLABELLED}` call rests on.\n"
            f"  disagreements now: {disagree}")
    print(f"  OK  question 1.5 offers {len(opts)} options and Table D.2's rows are those "
          f"options\n      in order, on {len(printed) - 1} of {len(printed)}. The "
          f"exception is option 6: the\n      questionnaire says "
          f"{QUESTIONNAIRE_SAYS!r}, the report prints {MISLABELLED!r}.")


def confirm_revision():
    """Prove that Rev 2.5 and Rev 2.7 print the same Table D.2."""
    a, b = _open(REPORT_NAME), _open(MIRROR_NAME)
    ta = _page(a, D2_NEEDLES).get_text()
    tb = _page(b, D2_NEEDLES).get_text()
    a.close()
    b.close()
    if ta != tb:
        raise SystemExit("Table D.2 DIFFERS between Release 2 Rev 2.5 (the publisher's) "
                         "and Rev 2.7 (the CARICOM mirror's). sources/lc.md §1 says they "
                         "are the same and the newer one must now be read instead.")
    print("  OK  Table D.2 is textually identical in Rev 2.5 and Rev 2.7, so the "
          "revision\n      this file reads cannot affect the map")


def _stub_line(page, mid, tol=6.0):
    """The stub cell sitting on one row of figures, in reading order.

    Sorted by line and then by x, because a stub that wraps would otherwise come back
    with its two lines interleaved.
    """
    ws = [(round((y0 + y1) / 2, 1), x0, w)
          for x0, y0, x1, y1, w, *_ in page.get_text("words")
          if x0 < VALUE_X0 and abs((y0 + y1) / 2 - mid) <= tol]
    return " ".join(w for _, _, w in sorted(ws))


def read_counts_page():
    """Tables A.1 and A.2, which share a page — the universe ladder and the district totals.

    A.1 is the universe this file's docstring prints; reading it rather than trusting a
    transcription means the ladder cannot go stale in silence. A.2 is the INDEPENDENT
    district table used to identify which column of Table D.2 is which district.

    Both tables' percentage columns carry decimal points and so do not match NUM; only
    the count columns do, 2022 then 2010 left to right. Rows are looked up by their stub
    label, and every expected label must be present exactly once — a header line that
    happens to hold two four-digit years is not one of them.
    """
    doc = _open(REPORT_NAME)
    page = _page(doc, A2_NEEDLES)
    rows, _ = _grid(page, 2)
    found = {}
    for mid, ts in rows:
        label = _stub_line(page, mid)
        if label:
            found.setdefault(label, []).append(_cell(ts[0][1]))
    doc.close()

    def one(label):
        got = found.get(label)
        if not got or len(got) != 1:
            raise SystemExit(f"the counts page has {label!r} {len(got or [])} times, "
                             f"expected once. Rows read: {sorted(found)}")
        return got[0]

    a1 = {k: one(k) for k in A1_LADDER}
    a2 = {name: one(name) for name, _ in DISTRICTS}
    return a1, a2


def read():
    """Table D.2 -> {category: {district label: count}}, plus the `Total` column."""
    doc = _open(REPORT_NAME)
    page = _page(doc, D2_NEEDLES)
    rows, rejected = _grid(page, len(DISTRICTS) + 1)

    want = len(CATEGORIES) + 1              # the `Total` row, then every category
    if len(rows) != want:
        raise SystemExit(f"Table D.2 gave {len(rows)} full rows of "
                         f"{len(DISTRICTS) + 1}, expected {want}. Rejected: "
                         f"{[[w for _, w in ts] for _, ts in rejected]}")

    # ---- the stub column must be the expected labels, in order ----
    got = _squash(_stub(page, rows[0][0] - 8))
    exp = _squash([TOTAL_CAT] + CATEGORIES)
    if got != exp:
        raise SystemExit(f"Table D.2's stub column is not the expected list.\n"
                         f"  got  {got}\n  want {exp}")

    # ---- and the column headings must be the expected districts, in order ----
    header = _header(page, [x for x, _ in rows[0][1]],
                     _caption_y(page, CAPTION_TOKEN), rows[0][0] - 8)
    exp_hdr = [TOTAL_CAT] + [n for n, _ in DISTRICTS]
    if [_squash([h]) for h in header] != [_squash([h]) for h in exp_hdr]:
        raise SystemExit(f"Table D.2's column headings are not the expected districts.\n"
                         f"  got  {header}\n  want {exp_hdr}")

    out = {}
    for (mid, ts), name in zip(rows, [TOTAL_CAT] + CATEGORIES):
        cells = [_cell(w) for _, w in ts]
        out[name] = {TOTAL_CAT: cells[0]}
        out[name].update({d: v for (d, _), v in zip(DISTRICTS, cells[1:])})
    doc.close()
    return out


def _caption_y(page, token):
    """The y of the caption line, which is the top of the header band.

    Above it sit the running page header and the page number, which are not part of
    the table and which land in whichever column they happen to be over.
    """
    ys = [(y0 + y1) / 2 for x0, y0, x1, y1, w, *_ in page.get_text("words")
          if w == token]
    if len(ys) != 1:
        raise SystemExit(f"the caption token {token!r} appears {len(ys)} times on the "
                         "page; the header band cannot be located")
    return ys[0]


def _header(page, col_x, y_floor, y_ceiling):
    """Column headings, each word filed under the column its x0 is nearest to.

    The headings wrap over up to three lines and break mid-word (`Canari` / `es`),
    so they are reassembled per column in reading order rather than per line, and
    compared with the spaces squashed out.

    `SPANNER` is dropped: it is the group heading printed once, centred over all ten
    district columns, and it belongs to none of them. Nothing else on the page between
    the caption and the first figure is anything but a column heading.
    """
    cols = {x: [] for x in col_x}
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        mid = (y0 + y1) / 2
        if not (y_floor < mid < y_ceiling) or x0 < VALUE_X0 - 20 or w == SPANNER:
            continue
        cols[min(col_x, key=lambda c: abs(c - x0))].append((mid, x0, w))
    return [" ".join(w for _, _, w in sorted(cols[x])) for x in col_x]


def rows_from(table):
    rows = []
    for name, pcode in DISTRICTS:
        for cat in [TOTAL_CAT] + CATEGORIES:
            note = ("level=district; universe is the household population, 171,834 of "
                    "a resident 172,948")
            if cat == TOTAL_CAT:
                note += "; district total, not a religion category"
            elif cat == MISLABELLED:
                note += ("; the report's own label -- the census questionnaire calls this "
                         "option `Evangelical` (sources/lc.md §2)")
            rows.append({"geo_id": pcode, "geo_level": "district", "geo_name": name,
                         "source_category": cat, "count": table[cat][name],
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": note})
    return rows


def check(rows, table, a1, a2):
    ok = True

    def result(label, bad, n, extra=""):
        nonlocal ok
        good = not bad
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} ({n} checks){extra}")
        for b in bad[:6]:
            print(f"        {b}")

    n_units = len({r["geo_id"] for r in rows})
    result(f"district {n_units} units (expected {len(DISTRICTS)})",
           [] if n_units == len(DISTRICTS) else ["wrong unit count"], n_units)

    nat = table[TOTAL_CAT][TOTAL_CAT]
    result(f"the national Total is {HOUSEHOLD_POPULATION:,}",
           [] if nat == HOUSEHOLD_POPULATION else [f"got {nat:,}"], 1)

    # ---- the ten districts sum to the Total column, on every row ----
    bad, row_spread = [], []
    for cat in [TOTAL_CAT] + CATEGORIES:
        got = sum(table[cat][n] for n, _ in DISTRICTS)
        d = got - table[cat][TOTAL_CAT]
        if d:
            row_spread.append((cat, got, table[cat][TOTAL_CAT], d))
        if abs(d) > ROW_TOL:
            bad.append(f"{cat}: {got:,} vs {table[cat][TOTAL_CAT]:,} ({d:+,})")
    result(f"the 10 districts sum to the row's own Total (tolerance {ROW_TOL})", bad,
           len(CATEGORIES) + 1)

    # ---- the categories sum to each district's own Total ----
    bad, col_spread = [], []
    for name in [TOTAL_CAT] + [n for n, _ in DISTRICTS]:
        got = sum(table[c][name] for c in CATEGORIES)
        d = got - table[TOTAL_CAT][name]
        col_spread.append((name, got, table[TOTAL_CAT][name], d))
        if abs(d) > COL_TOL:
            bad.append(f"{name}: {got:,} vs {table[TOTAL_CAT][name]:,} ({d:+,})")
    result(f"the categories sum to each district's own Total (tolerance {COL_TOL})", bad,
           len(DISTRICTS) + 1,
           "  <- a dropped row or a mis-assigned column would be out by hundreds")

    # ---- and every drawn cell together, which is the number that matters ----
    drawn = sum(table[c][n] for c in CATEGORIES for n, _ in DISTRICTS)
    d = drawn - HOUSEHOLD_POPULATION
    result(f"every drawn cell sums to {HOUSEHOLD_POPULATION:,} (tolerance {GRAND_TOL})",
           [] if abs(d) <= GRAND_TOL else [f"{drawn:,} ({d:+,})"], 1,
           f"   got {drawn:,}, {d:+,}, {abs(d) / HOUSEHOLD_POPULATION:.5%}")

    print("\n  where the table misses its own margins, and it is two-sided "
          f"({len(row_spread)} of {len(CATEGORIES) + 1} rows):")
    for cat, got, want, d in row_spread:
        print(f"      {cat:<40} {got:>8,} vs {want:>8,}   {d:+,}")
    print("  and per district:")
    for name, got, want, d in col_spread:
        print(f"      {name:<40} {got:>8,} vs {want:>8,}   {d:+,}")

    # ---- the census's OWN undercount weights, which explain all of the above ----
    bad = [f"{n}: Table 2 says {pop:,}, Table D.2 says {table[TOTAL_CAT][n]:,}"
           for n, (_, _, pop) in UNDERCOUNT_WEIGHTS.items()
           if table[TOTAL_CAT][n] != pop]
    result("Table 2's weighted `Estimated Household Population` is Table D.2's district "
           "total", bad, len(UNDERCOUNT_WEIGHTS),
           "\n        <- so what is drawn is the census's own estimate, already scaled "
           "up for\n           a 23.3% undercount by CSO. Nothing here scales it further "
           "(§14.4).")
    print("\n  CSO's Table 2 — the undercount it corrected, and by how much per district:")
    for n, (pct, w, pop) in sorted(UNDERCOUNT_WEIGHTS.items(),
                                   key=lambda kv: -kv[1][1]):
        print(f"      {n:<16} {pct:>5.1f}% undercount   weight {w:.3f}   -> {pop:>7,}")
    print(f"      {'Saint Lucia':<16} {NATIONAL_UNDERCOUNT_PCT:>5.1f}% undercount   "
          f"weight {NATIONAL_WEIGHT:.3f}   -> {HOUSEHOLD_POPULATION:>7,}")

    # ---- and an INDEPENDENT table, elsewhere in the report, agrees district by
    #      district. Every one of the ten figures is distinct, so this pins which
    #      column is which and not merely that the arithmetic closes.
    bad = []
    for name, _ in DISTRICTS:
        if name not in a2:
            bad.append(f"{name}: absent from Table A.2")
        elif a2[name] != table[TOTAL_CAT][name]:
            bad.append(f"{name}: D.2 {table[TOTAL_CAT][name]:,} vs A.2 {a2[name]:,}")
    result("Table A.2 agrees with Table D.2 district by district", bad, len(DISTRICTS),
           "  <- an independent table, and all ten figures differ")

    bad = [f"{k}: read {a1[k]:,}, this file says {v:,}"
           for k, v in A1_LADDER.items() if a1[k] != v]
    result("Table A.1's universe ladder is what this file says it is", bad,
           len(A1_LADDER))
    inst = a1["Total resident population"] - a1["In private households"]
    bad = ([] if inst == INSTITUTIONAL
           else [f"resident - household = {inst:,}, expected {INSTITUTIONAL:,}"])
    result(f"the institutional population is {INSTITUTIONAL:,}", bad, 1)

    print(f"\n  the universe, and what is outside it (Table A.1):")
    print(f"      {HOUSEHOLD_POPULATION:>8,}  in private households — THIS MAP")
    print(f"      {INSTITUTIONAL:>8,}  in institutions")
    print(f"      {RESIDENT_POPULATION:>8,}  total resident population")
    print(f"      {VISITORS:>8,}  visitors in hotels and guesthouses")
    print(f"      {TOTAL_POPULATION:>8,}  total population on census night")
    print(f"      {MIDYEAR_ESTIMATE_2022:>8,}  mid-year estimate for 2022, which the "
          "report contrasts\n                with its own count. Nothing here is scaled "
          "to it (§14.4).")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES:
        v = table[cat][TOTAL_CAT]
        flag = "   <- questionnaire: `Evangelical`" if cat == MISLABELLED else ""
        print(f"    {v:>8,}  {100.0 * v / HOUSEHOLD_POPULATION:6.2f}%  {cat}{flag}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    confirm_questionnaire()
    confirm_revision()
    table = read()
    a1, a2 = read_counts_page()
    rows = rows_from(table)
    check(rows, table, a1, a2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
