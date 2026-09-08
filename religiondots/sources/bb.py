"""Barbados — BSS, 2010 Population and Housing Census, Table 02.06.

Reads (or fetches) data/raw/bb/ and writes data/normalized/bb.csv.

**Twenty-two named categories on 11 parishes, for 226,193 people** — ~20,600 per unit, and
one of the deepest denominational lists in the Caribbean: Barbados counts Nazarenes,
Wesleyans, the Salvation Army, Moravians and the Brethren as separate answers, which almost
nothing else here does.

**IT IS DRAWN AT 2010 AND THE 2021 CENSUS EXISTS. THAT IS A DECISION, NOT AN OVERSIGHT.**
BSS ran a 2021 census and published the same Table 02.06 — *Total Population by Parish, Sex
and Religion* — in `Census-2020-Tables.xlsx`. It is not used, because **the 2021 census has
a 48.7% undercount** and the office says so itself:

    2021   Estimated Resident Population   269,090
           Tabulated Population            136,415
           Estimated Undercount            130,993   =  48.7%

    2010   Estimated Resident Population   277,821
           Tabulable Population            226,193
           Estimated Undercount             49,115   =  18%

And the 2021 report is explicit about what that costs the geography (§6.0, *Tabulated
Results*):

> *"The tabulated results of the 2021 Census can be regarded as a large sample of the
> resident population in Barbados … **In most cases, disaggregation by area is not included
> – as most results at that level would be understated**, considering the significant size
> of the undercount."*

The workbook nonetheless ships the parish cut. **The publisher's own warning is taken over
the publisher's own spreadsheet**, and the eleven-year-older census, whose undercount is
under half as large, is drawn instead. The category lists are the same 23 either way — 2021
renames `Muslim` to `Islam` and `Jewish` to `Judaism` and reorders — so nothing is lost in
depth by choosing 2010.

**THE `NO RELIGION` COLUMN HAS NO HEADER, AND IT IS 46,562 PEOPLE — 20.6% OF THE COUNTRY.**
In the 2010 sheet, column 23 sits between `Other Non-Christian` and `Not Stated` and its
header cell is **blank**. A header-driven read names it `Unnamed: 23` or drops it, and
Barbados silently loses its second largest answer. Two things identify it, and this file
asserts both:

  * **arithmetic** — the categories only sum to each unit's own `Total` when it is included,
    on all twelve rows; and
  * **the 2021 workbook**, whose Table 02.06 has the same categories in the same relative
    order with that position labelled **`No Religious Affiliation`**.

So the 2021 file, rejected as a source, is used as the *evidence* for reading the 2010 one.

**COVERAGE IS NOT UNIFORM ACROSS PARISHES**, and that is the real cost of the 18%. Against
Table C of the 2010 report (*Estimated Resident Population by Parish*), the tabulable
population runs from **74.6% of St. James to 96.1% of St. John**. Nothing here scales any
parish up (§14.4) — so an under-covered parish draws proportionally fewer dots than its true
population warrants — and `check()` prints the whole ladder.

Usage:
    python sources/bb.py --fetch    two workbooks, ~1.0 MB, seconds
    python sources/bb.py            normalise from data/raw/bb/
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bb")
OUT = os.path.join(ROOT, "data", "normalized", "bb.csv")

SOURCE_ID = "bb_phc_2010"
YEAR = 2010
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# BSS's own host. Neither workbook is linked from the census page — both are in the
# WordPress media library and nowhere else (§11v's rule, from the Bahamas).
XLSX_URL = "https://stats.gov.bb/wp-content/uploads/2021/03/Census-Tables-2010.xlsx"
XLSX_NAME = "bb_census_tables_2010.xlsx"

# The 2021 workbook. NOT a source — read only to confirm what the 2010 sheet's blank
# header column is. `Census 2020 Tables` is BSS's own name for the 2021 census.
XLSX_2021_URL = "https://stats.gov.bb/wp-content/uploads/2024/05/Census-2020-Tables.xlsx"
XLSX_2021_NAME = "bb_census_tables_2021.xlsx"

SHEET = "02.06"

# Row layout of sheet 02.06: three header rows, then one block per unit — a label row with
# no figures, then `Total`, `Male`, `Female`. The national block's label is `Total` too.
HDR_CATEGORY_ROW = 2
FIRST_BLOCK_ROW = 3
NATIONAL_LABEL = "Total"
SEX_ROWS = ("Total", "Male", "Female")

# In sheet order. Position 21 (0-based, counting from `Adventist`) is the blank header.
CATEGORIES = [
    "Adventist",
    "Anglican",
    "Baptist",
    "Brethren",
    "Church of God",
    "Jehovah Witness",
    "Methodist",
    "Moravian",
    "Mormon",
    "Nazarene",
    "Other Pentecostal",
    "Roman Catholic",
    "Salvation Army",
    "Wesleyan",
    "Other Christian",
    "Baha'i",
    "Hindu",
    "Jewish",
    "Muslim",
    "Rastafarian",
    "Other Non-Christian",
    "No Religious Affiliation",      # THE BLANK HEADER. See the module docstring.
    "Not Stated",
]
BLANK_HEADER_CATEGORY = "No Religious Affiliation"
TOTAL_CAT = "Total"

# The eleven parishes, in the order the sheet prints them, with the pcode COD-AB uses.
# Asserted from the boundary side in sources/bb_geo.py (§9ak's arrangement) -- BSS
# publishes no code of its own.
PARISHES = [
    ("St. Michael",   "BB08"),
    ("Christ Church", "BB01"),
    ("St. George",    "BB03"),
    ("St. Philip",    "BB10"),
    ("St. John",      "BB05"),
    ("St. James",     "BB04"),
    ("St. Thomas",    "BB11"),
    ("St. Joseph",    "BB06"),
    ("St. Andrew",    "BB02"),
    ("St. Peter",     "BB09"),
    ("St. Lucy",      "BB07"),
]

TABULABLE_POPULATION = 226_193      # Table A, 2010 report
ESTIMATED_RESIDENT = 277_821
INSTITUTIONAL = 2_513
UNDERCOUNT = 49_115

# 2010 report, Table C — *Estimated Resident Population by Parish and Sex*, which includes
# the institutional population and the undercount. Transcribed from the report because it
# is a PDF and the counts are an xlsx; **used ONLY for the coverage diagnostic below and
# for no drawn value**, so it cannot affect the map.
ESTIMATED_BY_PARISH = {
    "St. Michael": 88_529, "Christ Church": 54_336, "St. George": 19_767,
    "St. Philip": 30_662, "St. John": 8_963, "St. James": 28_498,
    "St. Thomas": 14_249, "St. Joseph": 6_620, "St. Andrew": 5_139,
    "St. Peter": 11_300, "St. Lucy": 9_758,
}

# The 2021 census, for the rejection note and for the header check.
Y2021 = dict(estimated=269_090, tabulated=136_415, undercount=130_993, pct=48.7)


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, name, floor in ((XLSX_URL, XLSX_NAME, 300_000),
                             (XLSX_2021_URL, XLSX_2021_NAME, 400_000)):
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
        # §5a: a 200 is not a download. BSS serves these as `application/vnd.ms-excel`
        # whatever they are, so check the container rather than the content-type.
        with open(dest, "rb") as fh:
            magic = fh.read(2)
        if magic != b"PK":
            raise SystemExit(f"{dest} is not an xlsx -- starts {magic!r}")
        print(f"  {os.path.getsize(dest):,} bytes")


def _cell(v):
    """One figure. Blanks are real zeros here -- the sheet leaves no cell empty inside a
    data row, so an empty one would be a layout change rather than a missing value."""
    import pandas as pd

    if pd.isna(v):
        raise SystemExit("empty cell inside a data row -- sheet 02.06 has been re-laid out")
    return int(round(float(v)))


def confirm_blank_header():
    """Prove the 2010 sheet's unlabelled column is `No Religious Affiliation`.

    The 2021 workbook publishes the same table with the same categories in the same
    relative order and that position LABELLED. Read it, line the two lists up, and assert
    the label. This is the whole reason the 2021 file is downloaded.
    """
    import pandas as pd

    path = os.path.join(RAW, XLSX_2021_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    df = pd.read_excel(path, sheet_name=SHEET, header=None)
    hdr = [str(v).strip() for v in df.iloc[HDR_CATEGORY_ROW].tolist()
           if str(v).strip() not in ("nan", "")]
    if not hdr or hdr[0] != TOTAL_CAT:
        raise SystemExit(f"2021 sheet {SHEET} header starts {hdr[:4]}, expected "
                         f"{TOTAL_CAT!r} first")
    cats_2021 = hdr[1:]

    # 2021 reorders and differs on four labels. Three are spelling — a curly apostrophe in
    # `Baha’i` and `Jehovah’s`, and the possessive `Jehovah's Witness` against 2010's
    # `Jehovah Witness` — and two are real renames. Held as an explicit table rather than
    # solved by fuzzy matching: the point of this check is that the two lists are the SAME
    # list, so anything that would let two different lists pass defeats it (§12).
    renames = {
        "Islam": "Muslim",
        "Judaism": "Jewish",
        "Jehovah's Witness": "Jehovah Witness",
    }

    def norm(s):
        return " ".join(str(s).replace("’", "'").split())

    got = {renames.get(norm(c), norm(c)) for c in cats_2021}
    want = {norm(c) for c in CATEGORIES}
    if got != want:
        raise SystemExit(
            "the 2021 category list no longer matches the 2010 one, so it cannot "
            f"identify the blank header.\n  only in 2021: {sorted(got - want)}\n"
            f"  only in 2010: {sorted(want - got)}")
    if BLANK_HEADER_CATEGORY not in {norm(c) for c in cats_2021}:
        raise SystemExit(f"the 2021 sheet has no {BLANK_HEADER_CATEGORY!r} column, so the "
                         "2010 blank header is unidentified -- do not guess it")
    print(f"  OK  the 2021 workbook names the same {len(cats_2021)} categories and "
          f"labels\n      the 2010 sheet's blank column {BLANK_HEADER_CATEGORY!r}")
    return cats_2021


def read():
    """{unit label: {sex: {category: count}}}, national under NATIONAL_LABEL."""
    import pandas as pd

    path = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    df = pd.read_excel(path, sheet_name=SHEET, header=None)

    # ---- the header, and the blank cell that has to be there ----
    hdr = df.iloc[HDR_CATEGORY_ROW].tolist()
    first = next(i for i, v in enumerate(hdr) if str(v).strip() == TOTAL_CAT)
    labels = hdr[first + 1:first + 1 + len(CATEGORIES)]
    blanks = [i for i, v in enumerate(labels) if str(v).strip() in ("nan", "")]
    want_blank = CATEGORIES.index(BLANK_HEADER_CATEGORY)
    if blanks != [want_blank]:
        raise SystemExit(
            f"expected exactly one blank header, at position {want_blank} "
            f"({BLANK_HEADER_CATEGORY!r}); found blanks at {blanks}. BSS has relabelled "
            "sheet 02.06 and CATEGORIES must be rechecked against it.")
    for i, (got, want) in enumerate(zip(labels, CATEGORIES)):
        if i == want_blank:
            continue
        if " ".join(str(got).split()) != want:
            raise SystemExit(f"header {i} is {got!r}, expected {want!r}")

    total_col = first
    cat_cols = list(range(first + 1, first + 1 + len(CATEGORIES)))

    # ---- walk the blocks ----
    out, unit = {}, None
    for r in range(FIRST_BLOCK_ROW, len(df)):
        label = str(df.iat[r, 0]).strip()
        if label in ("nan", ""):
            continue
        if label.lower().startswith("source"):
            break
        if label not in SEX_ROWS or unit is None:
            # a unit label row: no figures on it, except the national row whose label IS
            # `Total` and which is followed by its own `Total` row.
            if label in SEX_ROWS and unit is None:
                unit = NATIONAL_LABEL
                out[unit] = {}
            else:
                unit = label
                out.setdefault(unit, {})
                continue
        if pd.isna(df.iat[r, total_col]):
            continue
        cells = {TOTAL_CAT: _cell(df.iat[r, total_col])}
        for cat, c in zip(CATEGORIES, cat_cols):
            cells[cat] = _cell(df.iat[r, c])
        out[unit][label] = cells

    return out


def rows_from(blocks):
    rows = []
    for name, pcode in PARISHES:
        cells = blocks[name]["Total"]
        for cat in [TOTAL_CAT] + CATEGORIES:
            note = "level=parish; universe is the tabulable population, 81.4% of the " \
                   "estimated resident population"
            if cat == TOTAL_CAT:
                note += "; parish total, not a religion category"
            elif cat == BLANK_HEADER_CATEGORY:
                note += "; the sheet leaves this column's header blank (sources/bb.md §2)"
            rows.append({"geo_id": pcode, "geo_level": "parish", "geo_name": name,
                         "source_category": cat, "count": cells[cat], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return rows


def check(rows, blocks):
    ok = True

    def result(label, bad, n, extra=""):
        nonlocal ok
        good = not bad
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} ({n} checks){extra}")
        for b in bad[:6]:
            print(f"        {b}")

    n_units = len({r["geo_id"] for r in rows})
    result(f"parish {n_units} units (expected {len(PARISHES)})",
           [] if n_units == len(PARISHES) else ["wrong unit count"], n_units)

    names = [n for n, _ in PARISHES]
    missing = [n for n in names if n not in blocks]
    result("every parish block is present", missing, len(PARISHES))
    if missing:
        raise SystemExit("reconciliation FAILED")

    nat = blocks[NATIONAL_LABEL]["Total"]
    result(f"the national Total is {TABULABLE_POPULATION:,}",
           [] if nat[TOTAL_CAT] == TABULABLE_POPULATION
           else [f"got {nat[TOTAL_CAT]:,}"], 1)

    # ---- the categories sum to each unit's own Total. THIS is what identifies the
    #      blank-header column: drop it and every one of these fails by 20.6%.
    bad = []
    for unit in [NATIONAL_LABEL] + names:
        for sex in SEX_ROWS:
            cells = blocks[unit][sex]
            got = sum(cells[c] for c in CATEGORIES)
            if got != cells[TOTAL_CAT]:
                bad.append(f"{unit}/{sex}: {got:,} vs {cells[TOTAL_CAT]:,}")
    result("the categories sum to each unit's own Total", bad,
           (len(names) + 1) * len(SEX_ROWS),
           "  <- fails by 20.6% without the blank-header column")

    # ---- Male + Female == Total ----
    bad = []
    for unit in [NATIONAL_LABEL] + names:
        for cat in [TOTAL_CAT] + CATEGORIES:
            b = blocks[unit]
            if b["Male"][cat] + b["Female"][cat] != b["Total"][cat]:
                bad.append(f"{unit}/{cat}")
    result("Male + Female == Total", bad,
           (len(names) + 1) * (len(CATEGORIES) + 1))

    # ---- the 11 parishes sum to the national row ----
    bad = []
    for cat in [TOTAL_CAT] + CATEGORIES:
        got = sum(blocks[n]["Total"][cat] for n in names)
        if got != nat[cat]:
            bad.append(f"{cat}: {got:,} vs {nat[cat]:,}")
    result("the 11 parishes sum to the national row", bad, len(CATEGORIES) + 1)

    # ---- coverage, which is the country's real caveat ----
    print(f"\n  the universe, and what is outside it (2010 report, Table A):")
    print(f"      {ESTIMATED_RESIDENT:>8,}  estimated resident population")
    print(f"      {TABULABLE_POPULATION:>8,}  TABULABLE population — this map")
    print(f"      {INSTITUTIONAL:>8,}  institutional population, not included above")
    print(f"      {UNDERCOUNT:>8,}  estimated undercount "
          f"({100.0 * UNDERCOUNT / ESTIMATED_RESIDENT:.0f}%)")

    print("\n  coverage is NOT uniform across parishes, and nothing scales it (§14.4):")
    cov = []
    for n in names:
        t = blocks[n]["Total"][TOTAL_CAT]
        e = ESTIMATED_BY_PARISH[n]
        cov.append((100.0 * t / e, n, t, e))
    for pct, n, t, e in sorted(cov):
        print(f"      {n:<16} {t:>7,} of {e:>7,}   {pct:5.1f}%")
    lo, hi = min(cov)[0], max(cov)[0]
    print(f"      spread {lo:.1f}% to {hi:.1f}% — an under-covered parish draws "
          "proportionally\n      fewer dots than its true population warrants, and this "
          "map does not correct that.")

    print(f"\n  why 2010 and not 2021 (sources/bb.md §1):")
    print(f"      2021 estimated {Y2021['estimated']:,}, tabulated "
          f"{Y2021['tabulated']:,}, undercount {Y2021['undercount']:,} "
          f"({Y2021['pct']}%)")
    print(f"      2010 estimated {ESTIMATED_RESIDENT:,}, tabulable "
          f"{TABULABLE_POPULATION:,}, undercount {UNDERCOUNT:,} "
          f"({100.0 * UNDERCOUNT / ESTIMATED_RESIDENT:.0f}%)")
    print("      and BSS's 2021 report says of its own area tables: \"most results at "
          "that\n      level would be understated, considering the significant size of "
          "the undercount\".")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES:
        v = nat[cat]
        print(f"    {v:>8,}  {100.0 * v / TABULABLE_POPULATION:6.2f}%  {cat}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    confirm_blank_header()
    blocks = read()
    rows = rows_from(blocks)
    check(rows, blocks)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
