"""Micronesia — FSM Statistics, 2023 Population and Housing Census, Table B6.

Reads (or fetches) data/raw/fm/ and writes data/normalized/fm.csv.

**THE OFFICE MOVED AND THE QUEUE'S HOST IS GONE.** `fsmstatistics.fm` still resolves and still
returns HTTP 200, and what it serves is a LiteSpeed directory index holding `cgi-bin`,
`dasdas.png` and a stray `htaccess`. The office is now at **`stats.gov.fm`**, WordPress with WP
File Download ([[reference_wpfd_sweep]]), the sixth Pacific office here to run that plugin after
Fiji, PNG, the Solomon Islands, Tonga and Kiribati. Its library was swept with `id=0`, 71 files
over nine pages of ten, twice: 2026-09-08 and again the same day by the session that finished it.

**RELIGION IS A CENSUS QUESTION HERE AND IT IS ASKED OF EVERYONE.** `Table B6` has eleven
categories and its Total row equals Table B1's population exactly, so the universe is the whole
enumerated population and there is no age cutoff to state.

**THE 2023 CENSUS FOUND 75,817 PEOPLE.** `queue.md` priced Micronesia at 107,008, which is the
2010 count; a decade of emigration under the Compact took nearly a third of the country. Draw
the 2023 figure and say the year.

**THE TIER IS MIXED AND THAT IS THE WHOLE DESIGN OF THIS MODULE.** FSM publishes a per-state
basic-tables workbook beside the national one. **Yap's and Pohnpei's carry `Table B6. Religion
by Municipality`; Chuuk's workbook does not exist and Kosrae's stops at Table B5.** So:

    Yap       10,739  ->  20 municipalities   (10 on Yap proper, 10 outer islands)
    Pohnpei   26,102  ->  11 municipalities
    Chuuk     33,885  ->   1 unit, the state
    Kosrae     5,092  ->   1 unit, the state

**33 drawn units, 48.6% of the country's people at municipality and 51.4% at state.** This is
spec §12's Ghana answer stated exactly: *draw the fine unit where it reconciles and the coarse
one where it does not, so the drawn tier is two `geo_level`s and every drawn row is still
`measured`.* Nothing is spread, nothing is modelled, and no row changes tier. `sources/fm.md` §4
prices the alternative (four states everywhere) and says what would reverse this.

**KOSRAE'S MISSING TABLE IS ASSERTED, NOT ASSUMED.** `check_kosrae_has_no_b6()` reads Kosrae's
2023 workbook and fails the build if a `Table B6` ever appears in it. Chuuk's absence cannot be
checked from a file that does not exist, so it is checked from the library sweep instead and
recorded in `sources/fm.md` §4. If either turns up, this module should grow that state's
municipalities the way Yap and Pohnpei already have them.

**`*` MEANS SUPPRESSED OR ZERO AND THE TABLE SAYS SO.** The starred cells are the disclosure
floor and they are everywhere at municipality level (58 of Yap proper's 132, 73 of Pohnpei's),
but they are *small* cells: `impute()` fills only a cell that is the single unknown in its line,
never apportioning, and what is left after that is **228 people, 0.30% of the country**. No
category is destroyed by the finer tier; the worst are Yap's SDA at 65% placed and Pohnpei's
`No religion/Refused` at 72%, both of a base under 300.

**THE WORKBOOK'S MARGINS DO NOT CLOSE AND THAT IS THE PUBLISHER, NOT THE PARSER.** The four
printed state totals sum to 75,818 against a printed national 75,817, and Table B1's age rows
give a third set of state totals, each within one of the printed ones. Pohnpei's eleven
municipalities sum to 26,104 against its own printed 26,102. `MARGIN_SLACK` is the tolerance and
every run prints the discrepancy. Do not tighten it to an equality assert; it will refuse a
correct read.

Usage:
    python sources/fm.py --fetch    four xlsx from stats.gov.fm, ~1.7 MB
    python sources/fm.py            normalise from data/raw/fm/
"""

import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

RAW = os.path.join(ROOT, "data", "raw", "fm")
OUT = os.path.join(ROOT, "data", "normalized", "fm.csv")

SOURCE_ID = "fm_phc_2023"
YEAR = 2023
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# WP File Download's public route. The category id and the file id both came out of the
# `id=0` library sweep rather than off a page; `linkdownload` in that JSON is this string.
FILES = {
    "national": ("https://stats.gov.fm/download/18/population-statistics/2210/"
                 "fsm-basic-tables-2023-population-housing-census.xlsx",
                 "fsm_basic_tables_2023.xlsx"),
    "yap": ("https://stats.gov.fm/download/56/yap/2212/"
            "yap-basic-tables-2023-population-housing-census.xlsx",
            "yap_basic_tables_2023.xlsx"),
    "pohnpei": ("https://stats.gov.fm/download/54/pohnpei/2216/"
                "pni-basic-tables-2023-population-housing-census.xlsx",
                "pohnpei_basic_tables_2023.xlsx"),
    # Downloaded only so that its ABSENT religion table can be asserted rather than assumed.
    "kosrae": ("https://stats.gov.fm/download/55/kosrae/2217/"
               "ksa-basic-tables-2023-population-housing-census.xlsx",
               "kosrae_basic_tables_2023.xlsx"),
}

SHEET = "Basic_Tables"
TITLE = "Table B6. Religion by State, FSM: 2023"
YAP_TITLE = "Table B6. Religion by Municipality, Yap State: 2023"
PNI_TITLE = "Table B6. Religion by Municipality, Pohnpei State: 2023"

# The national header row exactly as it must read. Asserted before a number is taken, so a
# re-ordered workbook fails instead of silently transposing two states.
HEADER = ["Religion", "Total", "Yap", "Chuuk", "Pohnpei", "Kosrae"]
STATES = HEADER[2:]

# Which states are drawn at which tier, and why. See the module docstring and fm.md §4.
FINE_STATES = ["Yap", "Pohnpei"]
COARSE_STATES = ["Chuuk", "Kosrae"]

# Print order. Asserted too, in every one of the three tables.
CATEGORIES = [
    "Roman Catholic",
    "Congregation/Protestant",
    "Assembly of God",
    "Pentecostal",
    "Apostolic",
    "Baptist",
    "SDA",
    "Mormon",
    "Jehovah's Witness",
    "Other religion",
    "No religion/Refused",
]

NATIONAL = 75_817

# Yap's Table B6 is printed as TWO blocks with a group subtotal each, and the second is titled
# `(continued)`. The names are asserted in the workbook's own order; COD-AB's p-codes then run
# FM101..FM120 in exactly this sequence, which is the free independent check on the pairing
# (sources/fm_geo.py). MAIN_OUTER in COD reproduces this same 10/10 split.
YAP_MAIN = ["Rumung", "Maap", "Gagil", "Tomil", "Fanif",
            "Weloy", "Dalipebinaw", "Rull", "Kanifay", "Gilman"]
YAP_OUTER = ["Ulithi", "Fais", "Ngulu", "Woleai", "Eauripik",
             "Ifalik", "Faraulep", "Elato", "Lamotrek", "Satawal"]

# Pohnpei's names are WRAPPED ACROSS TWO HEADER ROWS -- `Madole-` sits above `nihmw`, and so
# do `Mwoak-`/`illoa`, `Sapwu-`/`ahfik`, `Kapinga-`/`marangi`. Read one row only and four of
# the eleven municipalities come back as a suffix, which reads as a changed category list.
POHNPEI_MUNIS = ["Madolenihmw", "U", "Nett", "Sokehs", "Kitti", "Kolonia",
                 "Mwoakilloa", "Pingelap", "Sapwuahfik", "Nukuoro", "Kapingamarangi"]

# How far this publisher's own margins are allowed to disagree with each other, per table.
# Every one of them is a perturbation of one or two people applied to a margin; see the
# module docstring. Anything larger is a parse error rather than a rounding one.
MARGIN_SLACK = 4

# FSM has never forwarded a religion tabulation to UNSD, so `tools/oracle.py` has no row and
# there is no independent witness on these numbers. Recorded rather than skipped quietly.
ORACLE = None


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
    for key, (url, name) in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, headers=ua, timeout=900)
        r.raise_for_status()
        if r.content[:2] != b"PK":
            raise SystemExit(f"stats.gov.fm returned something that is not an xlsx for "
                             f"{key} ({len(r.content):,} bytes) — the WPFD route may have "
                             "changed; re-sweep the library with task=files.getFiles&id=0")
        tmp = dest + ".part"                              # [[reference_wb_truncates]]
        with open(tmp, "wb") as fh:
            fh.write(r.content)
        os.replace(tmp, dest)
        print(f"  {os.path.getsize(dest):,} bytes")


def _cell(x):
    """A table cell -> int, or None for the `*` the workbook uses for suppressed-or-zero."""
    s = "" if x is None else str(x).strip()
    if s in ("*", "-", "", "nan", "None"):
        return None
    return int(round(float(s.replace(",", ""))))


def _sheet(name):
    import pandas as pd

    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing — run `python sources/fm.py --fetch`")
    return pd.read_excel(path, sheet_name=SHEET, header=None)


def _find(df, title):
    for i in range(df.shape[0]):
        if str(df.iat[i, 0]).strip() == title:
            return i
    raise SystemExit(f"{title!r} is not in sheet {SHEET!r} — the workbook was re-cut")


def _read_block(df, head, ncol, what):
    """The BOTH GENDER block under a Table B6 heading -> (labels, totals[], cells{cat: []}).

    `ncol` counts DATA columns, starting at spreadsheet column 1. The row sequence is
    asserted the whole way down, so a workbook that grows a category or reorders the sex
    blocks fails here rather than reading the next table by accident.
    """
    hdr = None
    for r in range(head + 1, head + 4):
        if str(df.iat[r, 0]).strip() == "Religion":
            hdr = r
            break
    if hdr is None:
        raise SystemExit(f"{what}: no `Religion` header row under the title")
    if str(df.iat[hdr + 1, 0]).strip().upper() != "BOTH GENDER":
        raise SystemExit(f"{what}: the row under the header is not BOTH GENDER; "
                         "the sex blocks may have been reordered")
    trow = hdr + 2
    if str(df.iat[trow, 0]).strip() != "Total":
        raise SystemExit(f"{what}: the BOTH GENDER block does not open with a Total row")

    totals = [_cell(df.iat[trow, c]) for c in range(1, 1 + ncol)]
    cells, labels = {}, []
    for k, cat in enumerate(CATEGORIES):
        r = trow + 1 + k
        labels.append(str(df.iat[r, 0]).strip())
        cells[cat] = [_cell(df.iat[r, c]) for c in range(1, 1 + ncol)]
    if labels != CATEGORIES:
        raise SystemExit(f"{what}: the categories are\n  {labels!r}\nexpected\n"
                         f"  {CATEGORIES!r}")
    nxt = str(df.iat[trow + 1 + len(CATEGORIES), 0]).strip().upper()
    if nxt != "MALE":
        raise SystemExit(f"{what}: the row after the eleventh category is {nxt!r}, not MALE "
                         "— Table B6 has grown a category")
    return hdr, totals, cells


def _impute(totals, cells, units, anchor, what):
    """Fill only a cell that is the SINGLE unknown in its line. Never apportion.

    Two directions, iterated until nothing moves:

      * along a category row, from `anchor` (the table's own subtotal column);
      * down a unit column, from that unit's own Total.

    Whatever is still unknown afterwards becomes 0 and shows up in the residual, which is
    reported per unit rather than absorbed. FSM's published tables do not close to the person
    even with every cell known, so a hard assert here would refuse a correct read.
    """
    filled = 0
    for _ in range(len(CATEGORIES) + len(units)):
        moved = 0
        for cat in CATEGORIES:
            unk = [c for c in units if cells[cat][c] is None]
            if len(unk) == 1 and cells[cat][anchor] is not None:
                c = unk[0]
                cells[cat][c] = max(cells[cat][anchor]
                                    - sum(cells[cat][d] for d in units if d != c), 0)
                moved += 1
        for c in units:
            unk = [cat for cat in CATEGORIES if cells[cat][c] is None]
            if len(unk) == 1 and totals[c] is not None:
                known = sum(cells[cat][c] for cat in CATEGORIES
                            if cells[cat][c] is not None)
                cells[unk[0]][c] = max(totals[c] - known, 0)
                moved += 1
        filled += moved
        if not moved:
            break
    left = sum(1 for cat in CATEGORIES for c in units if cells[cat][c] is None)
    for cat in CATEGORIES:
        for c in units:
            if cells[cat][c] is None:
                cells[cat][c] = 0
    print(f"    {what}: {filled} cells recovered from a margin, {left} of "
          f"{len(CATEGORIES) * len(units)} unit cells left unknown and set to 0")
    return cells


def parse_national():
    """Table B6's BOTH GENDER block -> (national[11], {state: [11]}, {state: total})."""
    df = _sheet(FILES["national"][1])
    head = _find(df, TITLE)
    got = [str(df.iat[head + 1, c]).strip() for c in range(6)]
    if got != HEADER:
        raise SystemExit(f"Table B6's header is\n  {got!r}\nexpected\n  {HEADER!r}")

    _, totals_row, cells_row = _read_block(df, head, 5, "national Table B6")
    national_total = totals_row[0]
    totals = {st: totals_row[1 + i] for i, st in enumerate(STATES)}
    national = [cells_row[cat][0] for cat in CATEGORIES]
    cells = {st: [cells_row[cat][1 + i] for cat in CATEGORIES]
             for i, st in enumerate(STATES)}

    if any(v is None for v in national):
        raise SystemExit("the national column has a suppressed cell; nothing can anchor it")
    if national_total != NATIONAL:
        raise SystemExit(f"Table B6's national total is {national_total}, expected {NATIONAL}")
    if sum(national) != NATIONAL:
        raise SystemExit(f"the national categories sum to {sum(national)}, "
                         f"not {NATIONAL} — this table has always closed exactly")
    # THE PUBLISHED STATE TOTALS DO NOT SUM TO THE PUBLISHED NATIONAL TOTAL, and the workbook
    # is like this throughout: Table B1's own age rows come to Yap 10,738 / Chuuk 33,884 /
    # Pohnpei 26,103 / Kosrae 5,092, which is 75,817 exactly, while its printed state totals
    # read 10,739 / 33,885 / 26,102 / 5,092 and come to 75,818. Every state is within one of
    # its own age column. It is a rounding or confidentiality perturbation applied to the
    # margins, not a parse error, and an equality assert here would refuse a correct read.
    slack = sum(totals.values()) - NATIONAL
    if abs(slack) > MARGIN_SLACK:
        raise SystemExit(f"the four state totals sum to {sum(totals.values())}, "
                         f"{slack:+d} against the national {NATIONAL} — more than the "
                         f"{MARGIN_SLACK} this workbook's margins are known to move by")
    if slack:
        print(f"  state totals sum to {sum(totals.values()):,}, {slack:+d} on the national "
              f"{NATIONAL:,}; the workbook's margins are perturbed by up to 1 per state")
    print(f"  national Table B6: {len(CATEGORIES)} categories, {len(STATES)} states, "
          f"national column closes on {NATIONAL:,} EXACTLY")

    # The two coarse states are drawn from this table, so their starred cells are recovered
    # here, against the national column and their own totals.
    idx = {st: i for i, st in enumerate(STATES)}
    flat_totals = [national_total] + [totals[st] for st in STATES]
    flat = {cat: [national[CATEGORIES.index(cat)]]
            + [cells[st][CATEGORIES.index(cat)] for st in STATES] for cat in CATEGORIES}
    flat = _impute(flat_totals, flat, list(range(1, 1 + len(STATES))), 0,
                   "national, four state columns")
    cells = {st: [flat[cat][1 + idx[st]] for cat in CATEGORIES] for st in STATES}
    return national, cells, totals


def parse_yap():
    """Yap's Table B6, both printed blocks -> ({municipality: [11]}, {municipality: total}).

    THE TABLE IS PRINTED IN TWO BLOCKS WITH A GROUP SUBTOTAL EACH and the second is titled
    `(continued)`. Block one is `YAP PROPER` (a TOTAL column for the state, a subtotal column
    for the group, then ten municipalities); block two is `OUTER ISLANDS` (a blank column
    where the state total was, a subtotal, then ten islands). Sum the columns blind and Yap
    comes out at twice its size.
    """
    df = _sheet(FILES["yap"][1])
    out_cells, out_totals = {}, {}
    state_total = state_cats = None

    for title, names, tag in ((YAP_TITLE, YAP_MAIN, "YAP PROPER"),
                              (YAP_TITLE + " (continued)", YAP_OUTER, "OUTER ISLANDS")):
        head = _find(df, title)
        banner = str(df.iat[head + 1, 2]).strip().upper()
        if banner != tag:
            raise SystemExit(f"Yap block {title!r}: the group banner over column 2 is "
                             f"{banner!r}, expected {tag!r} — the two blocks may have swapped")
        hdr, totals, cells = _read_block(df, head, 12, f"Yap {tag}")
        got = [str(df.iat[hdr, c]).strip() for c in range(3, 13)]
        if got != names:
            raise SystemExit(f"Yap {tag}: the municipality names are\n  {got!r}\n"
                             f"expected\n  {names!r}")
        if str(df.iat[hdr, 2]).strip() != "Total":
            raise SystemExit(f"Yap {tag}: column 2 is not the group subtotal")
        units = list(range(2, 12))          # data index 2 = spreadsheet column 3
        cells = _impute(totals, cells, units, 1, f"Yap {tag}, 10 units")
        if tag == "YAP PROPER":
            state_total = totals[0]
            state_cats = [cells[cat][0] for cat in CATEGORIES]
        for j, nm in enumerate(names):
            out_totals[nm] = totals[2 + j]
            out_cells[nm] = [cells[cat][2 + j] for cat in CATEGORIES]
        print(f"    Yap {tag}: subtotal {totals[1]:,}, ten units summing to "
              f"{sum(totals[2 + j] for j in range(10)):,}")

    if state_total is None:
        raise SystemExit("Yap's first block did not carry the state TOTAL column")
    got = sum(out_totals.values())
    if abs(got - state_total) > MARGIN_SLACK:
        raise SystemExit(f"Yap's twenty municipalities sum to {got:,} against the workbook's "
                         f"own state total {state_total:,}")
    print(f"  Yap: {len(out_totals)} municipalities summing to {got:,} against a printed "
          f"state total of {state_total:,}")
    return out_cells, out_totals, state_total, state_cats


def parse_pohnpei():
    """Pohnpei's Table B6 -> ({municipality: [11]}, {municipality: total}).

    ONE block, eleven municipalities, anchored on the state Total column. The names are
    WRAPPED OVER TWO HEADER ROWS and four of them come back as a bare suffix if only the
    lower row is read; both rows are joined and the result asserted against POHNPEI_MUNIS.
    """
    df = _sheet(FILES["pohnpei"][1])
    head = _find(df, PNI_TITLE)
    hdr, totals, cells = _read_block(df, head, 12, "Pohnpei Table B6")

    got = []
    for c in range(2, 13):
        a = str(df.iat[hdr - 1, c]).strip()
        b = str(df.iat[hdr, c]).strip()
        a = "" if a in ("nan", "None") else a
        b = "" if b in ("nan", "None") else b
        got.append((a + b).replace("-", ""))
    if got != POHNPEI_MUNIS:
        raise SystemExit(f"Pohnpei's municipality names are\n  {got!r}\nexpected\n"
                         f"  {POHNPEI_MUNIS!r}\n(they are wrapped over two header rows; "
                         "read the row above the `Religion` row as well)")
    if str(df.iat[hdr, 1]).strip() != "Total":
        raise SystemExit("Pohnpei: column 1 is not the state Total")

    units = list(range(1, 12))              # data index 1 = spreadsheet column 2
    cells = _impute(totals, cells, units, 0, "Pohnpei, 11 units")
    state_total = totals[0]
    state_cats = [cells[cat][0] for cat in CATEGORIES]
    out_totals = {nm: totals[1 + j] for j, nm in enumerate(POHNPEI_MUNIS)}
    out_cells = {nm: [cells[cat][1 + j] for cat in CATEGORIES]
                 for j, nm in enumerate(POHNPEI_MUNIS)}
    got_sum = sum(out_totals.values())
    if abs(got_sum - state_total) > MARGIN_SLACK:
        raise SystemExit(f"Pohnpei's eleven municipalities sum to {got_sum:,} against the "
                         f"workbook's own state total {state_total:,}")
    print(f"  Pohnpei: {len(out_totals)} municipalities summing to {got_sum:,} against a "
          f"printed state total of {state_total:,}")
    return out_cells, out_totals, state_total, state_cats


def check_kosrae_has_no_b6():
    """A TRIPWIRE, NOT AN ASSUMPTION. Kosrae's 2023 workbook stops at Table B5.

    Yap's and Pohnpei's per-state workbooks each carry `Table B6. Religion by Municipality`
    and Kosrae's does not: its List of Tables runs B1 age, B2 relationship, B3 birthplace,
    B4 citizenship, B5 marital status, and then the household tables. Chuuk has no 2023
    workbook at all (the `id=0` library sweep is the evidence; see sources/fm.md §4).

    If FSM ever adds religion to Kosrae's file this raises, and the right response is to give
    Kosrae its four municipalities the way Yap and Pohnpei already have theirs.
    """
    df = _sheet(FILES["kosrae"][1])
    tables = [str(df.iat[i, 0]).strip() for i in range(df.shape[0])
              if str(df.iat[i, 0]).strip().startswith("Table ")]
    b6 = [t for t in tables if t.startswith("Table B6")]
    if b6:
        raise SystemExit(
            "KOSRAE NOW PUBLISHES A TABLE B6 AND THIS MODULE IS OUT OF DATE:\n  "
            + "\n  ".join(b6)
            + "\nDraw Kosrae's four municipalities (COD FM401 Lelu, FM402 Malem, FM403 "
              "Utwe, FM404 Tafunsak) the way Yap and Pohnpei are drawn, and update "
              "sources/fm.md §4 and the `grain` line in countries.py.")
    print(f"  Kosrae's 2023 workbook has {len(tables)} tables and none of them is B6 "
          f"(it stops at {tables[-1].split('.')[0]}); Kosrae is drawn whole")


def normalise():
    national, state_cells, state_totals = parse_national()
    check_kosrae_has_no_b6()
    yap_cells, yap_totals, yap_state, yap_state_cats = parse_yap()
    pni_cells, pni_totals, pni_state, pni_state_cats = parse_pohnpei()
    print("  oracle check SKIPPED — UNSD table 28 has no Micronesia row at any year, "
          "so these figures have no outside witness")

    # --- THE ONE CHECK THAT CROSSES TWO PUBLICATIONS. The per-state workbooks and the
    # national one are separate releases; each state's own TOTAL column must reproduce the
    # national table's column for that state, category by category. A whole column read into
    # the wrong place survives every within-table margin and dies here.
    print("\n  per-state workbook against the national table, category by category:")
    for st, cats in (("Yap", yap_state_cats), ("Pohnpei", pni_state_cats)):
        i = STATES.index(st)
        worst = 0
        for k, cat in enumerate(CATEGORIES):
            d = (cats[k] or 0) - (state_cells[st][k] or 0)
            worst = max(worst, abs(d))
            if abs(d) > MARGIN_SLACK:
                raise SystemExit(
                    f"{st}/{cat}: the state workbook says {cats[k]} and the national table "
                    f"says {state_cells[st][k]} — {d:+d}, more than the {MARGIN_SLACK} this "
                    "publisher's margins move by. One of the two columns was read wrong.")
        st_total = yap_state if st == "Yap" else pni_state
        print(f"    {st:<9} 11 categories agree within {worst}; state total "
              f"{st_total:,} against the national table's {state_totals[st]:,}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows, residual = [], {}

    def emit(geo_id, level, name, counts, total):
        placed = 0
        for k, cat in enumerate(CATEGORIES):
            if counts[k] <= 0:
                continue
            placed += counts[k]
            rows.append({
                "geo_id": geo_id, "geo_level": level, "geo_name": name,
                "source_category": cat, "count": counts[k],
                "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID, "note": "",
            })
        residual[name] = (total or 0) - placed

    for st in COARSE_STATES:
        emit(st.lower(), "state", st, state_cells[st], state_totals[st])
    for nm in YAP_MAIN + YAP_OUTER:
        emit("yap-" + nm.lower(), "municipality", nm, yap_cells[nm], yap_totals[nm])
    for nm in POHNPEI_MUNIS:
        emit("pohnpei-" + nm.lower(), "municipality", nm, pni_cells[nm], pni_totals[nm])

    ids = {r["geo_id"] for r in rows}
    want = len(COARSE_STATES) + len(YAP_MAIN) + len(YAP_OUTER) + len(POHNPEI_MUNIS)
    if len(ids) != want:
        raise SystemExit(f"{len(ids)} distinct geo_ids, expected {want}")

    tmp = OUT + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)

    total = sum(r["count"] for r in rows)
    by_level = {}
    for r in rows:
        by_level.setdefault(r["geo_level"], set()).add(r["geo_id"])
    print(f"\nwrote {OUT}")
    print(f"  {len(rows):,} rows, {len(ids)} units, {total:,} people")
    for lvl in ("municipality", "state"):
        n = len(by_level.get(lvl, ()))
        p = sum(r["count"] for r in rows if r["geo_level"] == lvl)
        print(f"    {lvl:<13} {n:>3} units {p:>7,} people "
              f"({100.0 * p / total:.1f}%, {p / n:,.0f} each)")
    worst = sorted(residual.items(), key=lambda kv: -kv[1])[:5]
    print(f"  drawn {total:,} of {NATIONAL:,} ({100.0 * total / NATIONAL:.3f}%); "
          f"{NATIONAL - total} people are in cells this census suppressed and neither margin "
          "recovers,\n    the worst units being "
          + ", ".join(f"{n} {v}" for n, v in worst if v > 0))


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        normalise()
