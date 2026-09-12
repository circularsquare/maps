"""Belize — SIB, 2022 Population and Housing Census, General Characteristics Table 9.

Reads (or fetches) data/raw/bz/ and writes data/normalized/bz.csv.

**Twelve categories on 6 districts for 397,483 people** — ~66,000 per unit, which is finer
than Guyana's 10 regions (75,000) and an order finer than Zimbabwe's provinces (1.5M). Spec
§3.9b: take the finest geography a country publishes and say what it therefore cannot show.
SIB publishes religion at district and nowhere else, though it publishes *population* down to
city/town/village — see the note in §6 of `sources/bz.md` about why that does not help.

**IT IS WORTH DRAWING FOR ONE CATEGORY AND ONE NUMBER.** `Mennonite` is **15,440 people,
3.9% of Belize**, and no other census on this map names Mennonites at all — the tree has held
`christianity.anabaptist.mennonite` since the United States arrived and nothing outside ASARB
has ever filled it. Belize's are the Kleine Gemeinde and Old Colony settlements that arrived
from Mexico and Canada in 1958, and the geography is the point: they are **9.9% of Orange
Walk and 8.9% of Corozal** against 0.5% of Belize District and 0.5% of Stann Creek.

And `None` is **123,373 people, 31.0% — the highest irreligious share of any country this map
draws in the Americas**, half again Jamaica's 21.4%. It is also very unevenly spread: **46.6%
in Stann Creek** against 21.7% in Toledo. That is a large jump from 2010 and it is reported by
`check()` rather than passed through silently; §4 of `sources/bz.md` is what is and is not
known about it.

**THE FIGURES ARE FRACTIONAL AND THAT IS THE SOURCE, NOT A BUG.** Every cell in this workbook
is a real number — the national total is `397483.45623886667`, not 397,483. SIB publishes
*adjusted* census counts throughout (the `Admin_Area` sheet gives 2010 the same way,
`322423.8195952825`), the adjustment being for census undercount measured by its own
post-enumeration survey. They are carried as floats to the normalised file and rounded only
where dots are made. **Nothing here reconciles to the integer**, so every identity below is
asserted to a tolerance rather than to zero, and the tolerance is tight (1e-6 relative).

**THE TABLE IS AN EXACT PARTITION.** The twelve categories sum to each district's own Total
on all six rows and to the national row, and the six districts sum to the national row on all
twelve columns. `Don't Know/Not Stated` is 4,135 people (1.04%) and is the only §3.5 gap.

**MALE + FEMALE == TOTAL IS THE CHECK ON THE READ.** The sheet lays every group out as three
columns (Total, Male, Female) and only Total is drawn — but all three are read, because the
sex columns are the only check that would catch a district's block landing one column group
to the left or right. Every other identity here reconciles inside one group whichever
columns were taken. Zimbabwe's panel rule (§9aj) on a workbook instead of a PDF.

**DISTRICT ORDER IS A TRAP AND IT IS WHY geo_id IS COD'S OWN CODE.** SIB prints the six
districts **north to south** — Corozal, Orange Walk, Belize, Cayo, Stann Creek, Toledo —
which is Belize's national district order. COD-AB codes them **alphabetically**, so
`BZ01` is Belize District and not Corozal. Numbering the table by position, the way
`sources/mw.py` does, would produce a geo_id that disagrees with the boundary file on five of
six units while every total still reconciled. So the pcode is looked up by NAME here and
`sources/bz_geo.py` asserts the same mapping from the other side.

Usage:
    python sources/bz.py --fetch    one 69 KB xlsx, seconds
    python sources/bz.py            normalise from data/raw/bz/
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bz")
OUT = os.path.join(ROOT, "data", "normalized", "bz.csv")

SOURCE_ID = "bz_phc_2022"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# A plain link on sib.org.bz/census/2022-census/, no wall, no key.
XLSX_URL = "https://sib.org.bz/wp-content/uploads/Census2022_GeneralCharacteristics.xlsx"
XLSX_NAME = "Census2022_GeneralCharacteristics.xlsx"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

SHEET = "Religion_by_District"
TITLE = "Table 9: Population by Religion, District and Sex: 2022"

# In the order SIB prints them, north to south, with COD-AB's alphabetical pcode beside
# each. The pairing is asserted from the boundary side in sources/bz_geo.py.
DISTRICTS = [
    ("Corozal", "BZ03"),
    ("Orange Walk", "BZ04"),
    ("Belize", "BZ01"),
    ("Cayo", "BZ02"),
    ("Stann Creek", "BZ05"),
    ("Toledo", "BZ06"),
]

# In the order SIB prints them, down the rows. `Total` first is the column's own universe.
TOTAL_CAT = "Total"
CATEGORIES = [
    "Roman Catholic",
    "Pentecostal",
    "Seventh Day Adventist",
    "Anglican",
    "Mennonite",
    "Baptist",
    "Methodist",
    "Nazarene",
    "Jehovah's Witness",
    "Other",
    "None",
    "Don't Know/Not Stated",
]

SEXES = ["Total", "Male", "Female"]
DRAWN_SEX = "Total"

NATIONAL = 397_483.45623886667      # the sheet's own Total/Total cell
TOL = 1e-6                          # relative; the cells are floats, see the docstring


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, XLSX_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 20_000:
        print("already have", dest)
        return
    print("GET", XLSX_URL)
    # sib.org.bz answers 406 Not Acceptable to a bare `Mozilla/5.0` and 200 to a full
    # browser token. Not a bot wall — a Content-Negotiation rule that only inspects UA.
    r = requests.get(XLSX_URL, timeout=300, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(dest + ".part", dest)
    # §5a: HTTP 200 is not a download. Assert size AND type — an xlsx is a zip.
    with open(dest, "rb") as fh:
        magic = fh.read(2)
    if magic != b"PK":
        raise SystemExit(f"{dest} is not an xlsx -- starts {magic!r}, "
                         f"{os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def _norm(s):
    return " ".join(str(s or "").split())


def read():
    """Return {sex: {district_or_Total: {category: float}}}.

    The sheet is a single rectangular block: row 3 names the seven groups (Total plus six
    districts) on every third column, row 4 names Total/Male/Female under each, and rows
    5..16 are the categories with row 4 the universe. Both header rows are asserted in
    order before any figure is taken, so a re-ordered or re-labelled column list stops the
    run rather than quietly relabelling the map.
    """
    import openpyxl

    p = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    wb = openpyxl.load_workbook(p, data_only=True)
    if SHEET not in wb.sheetnames:
        raise SystemExit(f"{p} has no {SHEET!r} sheet -- it has {wb.sheetnames}. SIB has "
                         "restructured the workbook.")
    ws = wb[SHEET]
    grid = [[c for c in row] for row in ws.iter_rows(values_only=True)]

    title = next((_norm(c) for row in grid[:4] for c in row
                  if c and _norm(c).startswith("Table 9")), "")
    if title != TITLE:
        raise SystemExit(f"{SHEET} title is {title!r}, expected {TITLE!r} -- SIB has "
                         "renumbered or rescoped the table")

    # Locate the header rows by content rather than by index.
    hdr_g = next(i for i, row in enumerate(grid)
                 if any(_norm(c) == "Religion" for c in row))
    hdr_s = hdr_g + 1
    groups = ["Total"] + [d for d, _ in DISTRICTS]

    # Column of the first data column: the one after the 'Religion' label.
    c0 = next(j for j, c in enumerate(grid[hdr_g]) if _norm(c) == "Religion") + 1

    # Assert the group headers, which are written once per three-column block.
    for k, want in enumerate(groups):
        got = _norm(grid[hdr_g][c0 + 3 * k])
        if got != want:
            raise SystemExit(f"{SHEET}: column group {k} is headed {got!r}, expected "
                             f"{want!r} -- the district order has changed")
    # ...and the Total/Male/Female triple under each.
    for k in range(len(groups)):
        for s, want in enumerate(SEXES):
            got = _norm(grid[hdr_s][c0 + 3 * k + s])
            if got != want:
                raise SystemExit(f"{SHEET}: {groups[k]!r} sub-column {s} is headed "
                                 f"{got!r}, expected {want!r}")

    want_rows = [TOTAL_CAT] + CATEGORIES
    out = {s: {g: {} for g in groups} for s in SEXES}
    r = hdr_s + 1
    for want in want_rows:
        while r < len(grid) and not _norm(grid[r][c0 - 1]):
            r += 1
        got = _norm(grid[r][c0 - 1]) if r < len(grid) else "<eof>"
        if got != want:
            raise SystemExit(f"{SHEET}: expected the {want!r} row and read {got!r} -- SIB "
                             "has changed Table 9's category list and taxonomy/bz2022.py "
                             "must be revisited")
        for k, g in enumerate(groups):
            for s, sex in enumerate(SEXES):
                v = grid[r][c0 + 3 * k + s]
                if not isinstance(v, (int, float)):
                    raise SystemExit(f"{SHEET}: {g}/{sex}/{want} is {v!r}, not a number")
                out[sex][g][want] = float(v)
        r += 1
    return out


def rows_from(panels):
    rows = []
    cells = panels[DRAWN_SEX]
    for name, pcode in DISTRICTS:
        for cat in [TOTAL_CAT] + CATEGORIES:
            note = "level=district"
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            rows.append({"geo_id": pcode, "geo_level": "district", "geo_name": name,
                         "source_category": cat, "count": cells[name][cat],
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": note})
    return rows


def _close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(b))


def check(rows, panels):
    ok = True
    cells = panels[DRAWN_SEX]
    names = [d for d, _ in DISTRICTS]

    units = {r["geo_id"] for r in rows}
    good = len(units) == len(DISTRICTS)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} district   {len(units):>4} units "
          f"(expected {len(DISTRICTS)})")

    good = _close(cells["Total"][TOTAL_CAT], NATIONAL)
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {cells['Total'][TOTAL_CAT]:,.4f} "
          f"(expected {NATIONAL:,.4f})")

    bad = [g for g in ["Total"] + names
           if not _close(sum(cells[g][c] for c in CATEGORIES), cells[g][TOTAL_CAT])]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 12 categories sum to Total on all "
          f"{len(names) + 1} columns ({len(bad)} failures) {bad[:4]}")

    bad = []
    for cat in [TOTAL_CAT] + CATEGORIES:
        s = sum(cells[n][cat] for n in names)
        if not _close(s, cells["Total"][cat]):
            bad.append((cat, s, cells["Total"][cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 6 districts sum to the Total column on all "
          f"{len(CATEGORIES) + 1} rows")
    for c, s, w in bad[:5]:
        print(f"        {c}: {s:,.4f} vs {w:,.4f}")

    # The sex columns are the check on the READ, not on the census.
    bad = []
    for g in ["Total"] + names:
        for cat in [TOTAL_CAT] + CATEGORIES:
            m, f, t = (panels["Male"][g][cat], panels["Female"][g][cat],
                       panels["Total"][g][cat])
            if not _close(m + f, t):
                bad.append((g, cat, m + f, t))
    ok &= not bad
    n_cells = (len(names) + 1) * (len(CATEGORIES) + 1)
    print(f"  {'OK ' if not bad else 'BAD'} Male + Female == Total on all {n_cells} cells "
          f"({len(bad)} failures)")
    for g, c, s, w in bad[:5]:
        print(f"        {g}/{c}: {s:,.4f} vs {w:,.4f}")

    nat = cells["Total"]
    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in [TOTAL_CAT] + CATEGORIES:
        n = nat[cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>12,.1f}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    # The two cells worth a second look before anyone believes the map (§4 of bz.md).
    print("\n  the two large cells, by district:")
    for cat in ("None", "Mennonite"):
        shares = sorted(((100.0 * cells[n][cat] / cells[n][TOTAL_CAT], n) for n in names),
                        reverse=True)
        s = "  ".join(f"{n} {v:.1f}%" for v, n in shares)
        print(f"    {cat:<10} {s}")

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
