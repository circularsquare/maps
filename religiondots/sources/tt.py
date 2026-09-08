"""Trinidad and Tobago — CSO, 2011 Population and Housing Census Demographic Report, Table 8.

Reads (or fetches) data/raw/tt/ and writes data/normalized/tt.csv.

**Fourteen named religions plus Other/None/Not Stated, on 15 municipalities, for 1,322,546
people** — ~88,000 per unit, between Guyana's 74,700 and Jamaica's 191,000. Table 8 is
*Non-institutional population by sex, age group, religion and municipality*.

**THREE OF ITS CATEGORIES EXIST NOWHERE ELSE ON THIS MAP AND THEY ARE WHY THE COUNTRY IS
DRAWN.** `Orisha` (11,918) and `Baptist-Spiritual Shouter` (75,002) are Afro-Caribbean
traditions no other census here counts, and the Shouter Baptists are **5.7% of the country —
more numerous than the Anglicans**. `Rastafarian` (3,615) is the third census count of
Rastafari the project has, after Jamaica's and Saint Vincent's. And Trinidad is the second
place in the hemisphere where a census finds a large Hindu population: **240,100, 18.2%**,
which with Guyana and Suriname is the Indo-Caribbean geography this map was missing.

**THE SOURCE PDF THE SITE LINKS IS TRUNCATED AND THE ONE THIS FILE FETCHES IS NOT.** CSO
serves the report at two paths:

    /wp-content/uploads/2019/03/TRINIDAD-AND-TOBAGO-2011-Demographic-Report.pdf   281,190 B
    /wp-content/uploads/2020/01/2011-Demographic-Report.pdf                     7,013,046 B

The first is the one the site links and the one a media search finds first. It **is** a
`%PDF-1.4`, the server's `Content-Length` matches the bytes delivered exactly, and it ends
mid-stream with no `%%EOF`; PyMuPDF opens it, sets `is_repaired=True` and reports
**`page_count = 0`** without raising. So a complete download is not an intact file, and
`fetch()` below asserts the trailer and the page count rather than the byte count. See
`sources.md` §11t.

**`-` IS THE NIL MARKER AND IT IS NOT RARE.** Small religions in small municipalities print
a dash, not a zero — Moravian in Point Fortin, Orisha in several boroughs. A parser that
skips non-numeric tokens silently shifts every subsequent figure in the row left by one, and
the row would still have nine plausible figures in it. Dashes are read as 0 explicitly.

**THE PARSE IS DRIVEN BY EXPECTATION, NOT BY RECOGNITION.** The table is a fixed sequence:
17 unit blocks, each a unit label then nine age figures, then 17 category blocks of the same
shape. `read()` walks the expected sequence token by token and stops on the first deviation,
which is the only way to notice that CSO has re-typeset a page. It has to work on tokens
rather than lines because **page 168 packs three figures onto one line**
(`'113,771 123,008 105,161'`) while every other page puts one per line, and because the
national row carries its own first figure on the label line
(`'TRINIDAD AND TOBAGO 1,322,546'`).

**TWO NESTED UNIVERSES MAKE THE RECONCILIATION TWO-LEVEL.** The table prints `TRINIDAD AND
TOBAGO`, then `TRINIDAD`, then the 14 Trinidad municipalities, then `Tobago`. So the 14 sum
to TRINIDAD, and TRINIDAD + Tobago sums to TRINIDAD AND TOBAGO, on all 17 categories — and
the 15 drawn units are the 14 plus Tobago, with the two summary rows used only as checks.

**MALE + FEMALE == BOTH SEXES IS THE CHECK ON THE READ.** The table repeats in full three
times: BOTH SEXES on pages 168-185, MALE on 186-203, FEMALE on 204-221. Only BOTH SEXES is
drawn and all three are read, because every other identity here reconciles inside one panel
whichever way its tokens were taken. Zimbabwe's panel rule (§9aj) on a much longer table.

**THE UNIVERSE IS NON-INSTITUTIONAL, WHICH IS 0.41% SHORT OF THE CENSUS.** Table 8 excludes
the institutional population — prisons, hospitals, homes, barracks. The 2011 census counted
1,328,019 people in all; this table's universe is 1,322,546, so 5,473 people are outside it.
Nothing is scaled up to close the gap (§14.4); `note_public` says so.

Usage:
    python sources/tt.py --fetch    one 7.0 MB PDF, seconds
    python sources/tt.py            normalise from data/raw/tt/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tt")
OUT = os.path.join(ROOT, "data", "normalized", "tt.csv")

SOURCE_ID = "tt_phc_2011"
YEAR = 2011
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The INTACT copy. The 2019/03 path serves a truncated file — see the docstring.
PDF_URL = "https://cso.gov.tt/wp-content/uploads/2020/01/2011-Demographic-Report.pdf"
PDF_NAME = "tt_demographic_report_2011.pdf"
PDF_PAGES = 442

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

TITLE = "NON-INSTITUTIONAL POPULATION BY SEX, AGE GROUP, RELIGION AND MUNICIPALITY"

# 0-based page indices. Each panel runs 18 pages; the EVEN ones carry column (1), All Ages,
# and the odd ones continue the age bands, which are not read.
PANELS = {
    "both": range(168, 186, 2),
    "male": range(186, 204, 2),
    "female": range(204, 222, 2),
}
DRAWN_PANEL = "both"
SEX_HEADER = {"both": "BOTH SEXES", "male": "MALE", "female": "FEMALE"}

N_AGE_COLS = 9          # All Ages + eight five-year bands, on the even pages
ALL_AGES = 0            # column (1)

# The two nested universes, in the order the table prints them. Not drawn.
NATIONAL = "TRINIDAD AND TOBAGO"
ISLAND = "TRINIDAD"

# The 14 Trinidad municipalities in the order Table 8 prints them, with COD-AB's own
# ADM1_PCODE beside each. The pairing is asserted from the boundary side in
# sources/tt_geo.py; CSO publishes no code of its own in this table.
MUNICIPALITIES = [
    ("City of Port of Spain", "TT10"),
    ("City of San Fernando", "TT20"),
    ("Borough of Arima", "TT30"),
    ("Borough of Chaguanas", "TT40"),
    ("Borough of Point Fortin", "TT80"),
    ("Couva/ Tabaquite/ Talparo", "TT36"),
    ("Diego Martin", "TT31"),
    ("Mayaro/ Rio Claro", "TT51"),
    ("Penal/ Debe", "TT72"),
    ("Princes Town", "TT71"),
    ("San Juan/Laventille", "TT32"),
    ("Sangre Grande", "TT61"),
    ("Siparia", "TT81"),
    ("Tunapuna/ Piarco", "TT33"),
]
TOBAGO = ("Tobago", "TT90")
DRAWN = MUNICIPALITIES + [TOBAGO]

# Every unit block in the order the table prints them.
UNIT_ORDER = [NATIONAL, ISLAND] + [n for n, _ in MUNICIPALITIES] + [TOBAGO[0]]

# In the order CSO prints them, down the rows. Note the CURLY apostrophe in Jehovah's.
CATEGORIES = [
    "Anglican",
    "Baptist-Spiritual Shouter",
    "Baptist-Other",
    "Hinduism",
    "Islam",
    "Jehovah’s Witness",
    "Methodist",
    "Moravian",
    "Orisha",
    "Pentecostal/ Evangelical/ Full Gospel",
    "Presbyterian/ Congregational",
    "Rastafarian",
    "Roman Catholic",
    "Seventh Day Adventist",
    "Other",
    "None",
    "Not Stated",
]
TOTAL_CAT = "Total"      # the unit's own row, printed against the unit label

NATIONAL_TOTAL = 1_322_546          # Table 8's own TRINIDAD AND TOBAGO / All Ages cell
CENSUS_TOTAL = 1_328_019            # the whole 2011 census, institutional included

# **NO IDENTITY IN THIS TABLE IS EXACT, AND THAT IS THE SOURCE RATHER THAN THE PARSE.**
# CSO's 2011 figures are WEIGHTED estimates: its own per-municipality workbooks on
# cso.gov.tt publish fractional people (`10.308105`, `195.796888`), so every integer printed
# in Table 8 has been rounded independently and sums of rounded cells do not tie.
#
# MEASURED, not assumed. Across all 359 identity checks this file runs:
#     -2:  4    -1: 55    0: 249    +1: 47    +2:  4
# symmetric, bounded at two people, and 69% exact. A parse error looks nothing like that —
# it is one-sided and large — so the bound is asserted at 2 and the DISTRIBUTION is printed,
# which is what would actually reveal a re-typeset page.
ROUND_TOL = 2

FIG = re.compile(r"^(?:[\d,]+|-)$")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 5_000_000:
        print("already have", dest)
        return
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, timeout=1800, stream=True, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(dest + ".part", dest)

    # §5a, and then one step further. CSO's OTHER copy of this report is a valid %PDF
    # header, a matching Content-Length and a stream that stops mid-object — so the byte
    # count proves nothing. Assert the TRAILER, and then that a reader finds pages.
    blob_head = open(dest, "rb").read(5)
    with open(dest, "rb") as fh:
        fh.seek(max(0, os.path.getsize(dest) - 4096))
        tail = fh.read()
    if blob_head != b"%PDF-":
        raise SystemExit(f"{dest} is not a PDF -- starts {blob_head!r}")
    if b"%%EOF" not in tail:
        raise SystemExit(f"{dest} has no %%EOF -- it is TRUNCATED at source. This is what "
                         "the 2019/03 copy of this report does; check PDF_URL.")
    import fitz
    doc = fitz.open(dest)
    if doc.page_count != PDF_PAGES:
        raise SystemExit(f"{dest} has {doc.page_count} pages, expected {PDF_PAGES} -- "
                         "a page_count of 0 with no exception means a broken xref")
    print(f"  {os.path.getsize(dest):,} bytes, {doc.page_count} pages")


def _num(tok, where):
    if tok == "-":
        return 0            # the nil marker, not a missing value
    if not FIG.match(tok):
        raise SystemExit(f"{where}: {tok!r} is not a figure")
    return int(tok.replace(",", ""))


def _tokens(doc, pages, panel):
    """Every token of a panel's All-Ages pages, header bands removed."""
    out = []
    for p in pages:
        lines = [l.strip() for l in doc[p].get_text().splitlines() if l.strip()]
        flat = " ".join(" ".join(lines).split())
        if TITLE not in flat:
            raise SystemExit(f"page {p} does not carry Table 8 -- it starts {flat[:110]!r}. "
                             "CSO has re-paginated; find the table and update PANELS.")
        if SEX_HEADER[panel] not in lines[:14]:
            raise SystemExit(f"page {p} is not the {SEX_HEADER[panel]!r} panel")
        if "All Ages" not in lines:
            raise SystemExit(f"page {p} has no 'All Ages' column -- PANELS must list only "
                             "the even pages of each panel")
        try:
            i = lines.index("(9)") + 1
        except ValueError:
            raise SystemExit(f"page {p} has no '(9)' column marker")
        for l in lines[i:]:
            out.extend(l.split())
    return out


def _expect_label(toks, i, want, where):
    """Consume the tokens of `want`, which the text layer may have split anywhere.

    **Compared case-INSENSITIVELY, because one label changes case between panels**: the
    island is `Tobago` in BOTH SEXES and `TOBAGO` in MALE and FEMALE. Nothing else differs,
    and folding is safe here because the walk is expectation-driven — `TRINIDAD` and
    `TRINIDAD AND TOBAGO` are told apart by what is expected next, not by matching.
    """
    buf = ""
    start = i
    w = want.casefold()
    while i < len(toks):
        buf = " ".join((buf + " " + toks[i]).split())
        i += 1
        b = buf.casefold()
        if b == w:
            return i
        if not w.startswith(b):
            raise SystemExit(
                f"{where}: expected {want!r} and read {buf!r} (from token {start}) -- "
                "CSO has changed Table 8's row list and taxonomy/tt2011.py must be "
                "revisited")
    raise SystemExit(f"{where}: ran out of tokens looking for {want!r}")


def _panel(toks, panel):
    """One full pass of Table 8: {unit: {category: All-Ages count}}."""
    out, i = {}, 0
    for unit in UNIT_ORDER:
        i = _expect_label(toks, i, unit, f"{panel} unit")
        vals = [_num(toks[i + k], f"{panel} {unit} total") for k in range(N_AGE_COLS)]
        i += N_AGE_COLS
        cells = {TOTAL_CAT: vals[ALL_AGES]}
        for cat in CATEGORIES:
            i = _expect_label(toks, i, cat, f"{panel} {unit}")
            v = [_num(toks[i + k], f"{panel} {unit}/{cat}") for k in range(N_AGE_COLS)]
            i += N_AGE_COLS
            cells[cat] = v[ALL_AGES]
        out[unit] = cells
    if i != len(toks):
        raise SystemExit(f"{panel}: {len(toks) - i} tokens left over after the last unit "
                         f"-- the table has grown a row. Next: {toks[i:i + 8]}")
    return out


def read():
    import fitz

    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)
    if doc.page_count != PDF_PAGES:
        raise SystemExit(f"{p} has {doc.page_count} pages, expected {PDF_PAGES}")
    return {name: _panel(_tokens(doc, pages, name), name)
            for name, pages in PANELS.items()}


def rows_from(panels):
    cells = panels[DRAWN_PANEL]
    rows = []
    for name, pcode in DRAWN:
        for cat in [TOTAL_CAT] + CATEGORIES:
            note = "level=municipality; universe is the non-institutional population"
            if cat == TOTAL_CAT:
                note += "; unit total, not a religion category"
            rows.append({"geo_id": pcode, "geo_level": "municipality", "geo_name": name,
                         "source_category": cat, "count": cells[name][cat],
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": note})
    return rows


def check(rows, panels):
    from collections import Counter

    ok = True
    cells = panels[DRAWN_PANEL]
    muni = [n for n, _ in MUNICIPALITIES]
    cats = [TOTAL_CAT] + CATEGORIES
    spread = Counter()

    def identity(label, pairs):
        """`pairs` is (what, got, want). Holds within ROUND_TOL; the spread is the check."""
        nonlocal ok
        worst, over = 0, []
        for what, got, want in pairs:
            d = got - want
            spread[d] += 1
            worst = max(worst, abs(d))
            if abs(d) > ROUND_TOL:
                over.append((what, got, want, d))
        good = not over
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} "
              f"({len(pairs)} checks, max |diff| {worst})")
        for what, got, want, d in over[:5]:
            print(f"        {what}: {got:,} vs {want:,}  ({d:+,})")

    units = {r["geo_id"] for r in rows}
    good = len(units) == len(DRAWN)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} municipality {len(units):>3} units "
          f"(expected {len(DRAWN)})")

    good = cells[NATIONAL][TOTAL_CAT] == NATIONAL_TOTAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {cells[NATIONAL][TOTAL_CAT]:,} "
          f"(expected {NATIONAL_TOTAL:,}) -- exact, this one is printed not summed")

    identity("the 17 categories sum to the unit's own total",
             [(u, sum(cells[u][c] for c in CATEGORIES), cells[u][TOTAL_CAT])
              for u in UNIT_ORDER])

    identity("the 14 municipalities sum to TRINIDAD",
             [(c, sum(cells[n][c] for n in muni), cells[ISLAND][c]) for c in cats])

    identity("TRINIDAD + Tobago sums to TRINIDAD AND TOBAGO",
             [(c, cells[ISLAND][c] + cells[TOBAGO[0]][c], cells[NATIONAL][c])
              for c in cats])

    # The other two panels are the check on the READ, not on the census.
    identity("Male + Female == Both Sexes",
             [(f"{u}/{c}", panels["male"][u][c] + panels["female"][u][c],
               panels["both"][u][c]) for u in UNIT_ORDER for c in cats])

    tot = sum(spread.values())
    exact = spread[0]
    print(f"\n  the rounding spread across all {tot} identity checks "
          f"({100.0 * exact / tot:.0f}% exact):")
    print("    " + "  ".join(f"{d:+d}:{spread[d]}" for d in sorted(spread)))
    print("      CSO's 2011 figures are WEIGHTED — its own municipality workbooks publish\n"
          "      fractional people — so each printed integer is rounded independently and\n"
          "      sums of rounded cells do not tie. Symmetric and bounded at two people is\n"
          "      what rounding looks like; a parse error is one-sided and large.")

    nat = cells[NATIONAL]
    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in cats:
        n = nat[cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>10,}  {100.0 * n / NATIONAL_TOTAL:6.2f}%  {cat}{mark}")

    inst = CENSUS_TOTAL - NATIONAL_TOTAL
    print(f"\n  the universe is NON-INSTITUTIONAL: {NATIONAL_TOTAL:,} of the census's "
          f"{CENSUS_TOTAL:,},\n  so {inst:,} people ({100.0 * inst / CENSUS_TOTAL:.2f}%) "
          "are outside this table and are not scaled in.")

    print("\n  the three categories nothing else on this map has:")
    for cat in ("Baptist-Spiritual Shouter", "Orisha", "Rastafarian"):
        shares = sorted(((100.0 * cells[n][cat] / cells[n][TOTAL_CAT], n)
                         for n, _ in DRAWN), reverse=True)
        top = "  ".join(f"{n} {v:.1f}%" for v, n in shares[:3])
        print(f"    {cat:<26} {nat[cat]:>7,}  {100.0 * nat[cat] / NATIONAL_TOTAL:5.2f}%   "
              f"top: {top}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    panels = read()
    rows = rows_from(panels)
    check(rows, panels)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
