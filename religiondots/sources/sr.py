"""Suriname — ABS, Census 7 (2004), census profile at ressort level.

Reads (or fetches) data/raw/sr/ and writes data/normalized/sr.csv.

**THE FINEST GEOGRAPHY IN THE AMERICAS ON THIS MAP, AND THE SHALLOWEST QUESTION IN THE
CARIBBEAN.** 492,829 people on **62 ressorten** — ~7,900 people each, finer than every other
country here except Saint Vincent's enumeration districts and Ireland's small areas — with
**five religion categories** and a `Don't know/No answer` cell of 15.7%.

**THE VINTAGE IS 2004 AND IT IS NOT A CHOICE.** §11t framed this as a fork between fine
geography (2004) and deep categories (2012). It is not a fork; it was checked and there is
only one route:

  * **Census 8 (2012) publishes religion NATIONALLY only.** Volume 1 carries a full
    denominational list — Rooms Katholiek, Luthers, Volle Evangelie, E.B.G., Javanisme,
    Islam Soenniet, Hindoe Sanatan and the rest — crossed with ethnicity and with
    nationality, and **cut by no geography at all**.
  * **The Districtsresultaten presentations cover 3 of 10 districts.** Volume III
    (Marowijne, Brokopondo, Sipaliwini) has `Bevolking naar Godsdienst (denominatie)`;
    Volumes I and II have real text tables on other subjects and **no religion pages**. That
    was verified against the text layer rather than assumed — all three volumes carry text,
    so the absence is the source's and not a scanning artefact.
  * **Census 9 was in the field to July 2025 and has published nothing.** ABS's media
    library holds 3,662 items; everything Census 9 is promotional or legislative. When its
    results land they supersede this file entirely.

So 2004 is the only whole-country sub-national religion table Suriname has. Drawn, with the
vintage stated in `note_public`. Precedent: China 2000 (§14.6) and Ethiopia 2007 are drawn
at greater age.

**THE TABLE IS AN EXACT PARTITION.** The six religion rows sum to each ressort's own
population total, and the 62 ressorten sum to the national column, with a gap of zero on
every cell — integers, no rounding, no suppression.

**THE FILE HAS NO DISTRICT COLUMN, AND THE DISTRICT FILE SUPPLIES IT.** The ressort workbook
is 62 unlabelled columns in print order. Its sibling `district-profiel-census.xls` publishes
**Aantal Ressorten per district** — 12, 7, 5, 3, 6, 6, 6, 5, 6, 6 — which sums to 62 and
consumes the columns exactly, in order. That is what makes a district assignment a *read*
rather than a guess, and it matters because **the ressort names are not unique**: there is a
`Welgelegen` in both Paramaribo and Coronie, and a `Centrum` in both Paramaribo and
Brokopondo. Joining on name alone would collide on four units.

Usage:
    python sources/sr.py --fetch    two small .xls, seconds
    python sources/sr.py            normalise from data/raw/sr/
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sr")
OUT = os.path.join(ROOT, "data", "normalized", "sr.csv")

SOURCE_ID = "sr_phc_2004"
YEAR = 2004
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://www.statistics-suriname.org/wp-content/uploads/2019/03"
RESSORT_URL = f"{BASE}/census-profile-on-ressort-level.xls"
DISTRICT_URL = f"{BASE}/district-profiel-census.xls"
RESSORT_NAME = "census-profile-on-ressort-level.xls"
DISTRICT_NAME = "district-profiel-census.xls"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

SHEET = "POPULATION BY RESSORT"
HEADER_ROW = 3          # ressort names
FIRST_COL = 3           # col 2 is `Total`; the ressorten start at 3

# Read off district-profiel-census.xls row 2 (`Aantal Ressorten`) and asserted against it in
# check(), rather than hardcoded blind. Order is the print order of both files.
GROUPS = [("Paramaribo", 12), ("Wanica", 7), ("Nickerie", 5), ("Coronie", 3),
          ("Saramacca", 6), ("Commewijne", 6), ("Marowijne", 6), ("Para", 5),
          ("Brokopondo", 6), ("Sipaliwini", 6)]

TOTAL_CAT = "Total"
CATEGORIES = [
    "Christianity",
    "Hinduism",
    "Islam",
    "Traditional Religion +Others",
    "No religion",
    "Don't know/No answer",
]

NATIONAL = 492_829

# (district, ressort as ABS spells it, COD-AB ADM2_PCODE). The pairing is DERIVED
# independently from names in sources/sr_geo.py and asserted against this list; see that
# file for the four spellings that do not fold onto COD's and why the resolution is forced
# rather than chosen.
RESSORTEN = [
    # Paramaribo
    ("Paramaribo", "Blauwgrond", "SR0702"),
    ("Paramaribo", "Rainville", "SR0709"),
    ("Paramaribo", "Munder", "SR0707"),
    ("Paramaribo", "Centrum", "SR0703"),
    ("Paramaribo", "Beekhuizen", "SR0701"),
    ("Paramaribo", "Weg naar Zee", "SR0711"),
    ("Paramaribo", "Welgelegen (Par'bo)", "SR0712"),
    ("Paramaribo", "Tammenga", "SR0710"),
    ("Paramaribo", "Flora", "SR0704"),
    ("Paramaribo", "Latour", "SR0705"),
    ("Paramaribo", "Pontbuiten", "SR0708"),
    ("Paramaribo", "Livorno", "SR0706"),
    # Wanica
    ("Wanica", "Kwatta", "SR1005"),
    ("Wanica", "Saramacca Polder", "SR1007"),
    ("Wanica", "Koewarasan", "SR1004"),
    ("Wanica", "De Nieuwe Grond", "SR1001"),
    ("Wanica", "Lelydorp", "SR1006"),
    ("Wanica", "Houttuin", "SR1003"),
    ("Wanica", "Domburg", "SR1002"),
    # Nickerie
    ("Nickerie", "Wageningen", "SR0504"),
    ("Nickerie", "Groot Henar", "SR0501"),
    ("Nickerie", "Oostelijke Polders", "SR0503"),
    ("Nickerie", "Nieuw Nickerie", "SR0502"),
    ("Nickerie", "Westelijke Polders", "SR0505"),
    # Coronie
    ("Coronie", "Welgelegen (Coronie)", "SR0303"),
    ("Coronie", "Totness", "SR0302"),
    ("Coronie", "Johanna Maria", "SR0301"),
    # Saramacca
    ("Saramacca", "Calcutta", "SR0801"),
    ("Saramacca", "Tijgerkreek", "SR0805"),
    ("Saramacca", "Groningen", "SR0802"),
    ("Saramacca", "Kampong Baroe", "SR0804"),
    ("Saramacca", "Wayambo", "SR0806"),
    ("Saramacca", "Jarikaba", "SR0803"),
    # Commewijne
    ("Commewijne", "Margaretha", "SR0203"),
    ("Commewijne", "Bakkie", "SR0202"),
    ("Commewijne", "Nieuw Amsterdam", "SR0205"),
    ("Commewijne", "Alkmaar", "SR0201"),
    ("Commewijne", "Tamanredjo", "SR0206"),
    ("Commewijne", "Meerzorg", "SR0204"),
    # Marowijne
    ("Marowijne", "Moengo", "SR0403"),
    ("Marowijne", "Wanhatti", "SR0406"),
    ("Marowijne", "Galibi", "SR0402"),
    ("Marowijne", "Moengo Tapoe", "SR0404"),
    ("Marowijne", "Albina", "SR0401"),
    ("Marowijne", "Patamacca", "SR0405"),
    # Para
    ("Para", "Para Noord", "SR0603"),
    ("Para", "Para Oost", "SR0604"),
    ("Para", "Para Zuid", "SR0605"),
    ("Para", "Bigi Poika", "SR0601"),
    ("Para", "Carolina", "SR0602"),
    # Brokopondo
    ("Brokopondo", "Kwakoegron", "SR0104"),
    ("Brokopondo", "Marchallkreeek", "SR0105"),
    ("Brokopondo", "Klaaskreek", "SR0103"),
    ("Brokopondo", "Brokopondo Centrum", "SR0102"),
    ("Brokopondo", "Brownsweg", "SR0101"),
    ("Brokopondo", "Sarakreek", "SR0106"),
    # Sipaliwini
    ("Sipaliwini", "Tapanahony", "SR0906"),
    ("Sipaliwini", "Boven-Suriname", "SR0903"),
    ("Sipaliwini", "Boven-Saramacca", "SR0902"),
    ("Sipaliwini", "Boven-Coppename", "SR0901"),
    ("Sipaliwini", "Kabalebo", "SR0905"),
    ("Sipaliwini", "Coeroeni", "SR0904"),
]
EXPECTED = 62


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, name in ((RESSORT_URL, RESSORT_NAME), (DISTRICT_URL, DISTRICT_NAME)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, timeout=600, headers={"User-Agent": UA})
        r.raise_for_status()
        with open(dest + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dest + ".part", dest)
        # §5a: a 200 is not a download. These are OLE2 .xls, not xlsx.
        with open(dest, "rb") as fh:
            magic = fh.read(8)
        if magic[:4] != b"\xd0\xcf\x11\xe0":
            raise SystemExit(f"{dest} is not an OLE2 .xls -- starts {magic!r}")
        print(f"  {os.path.getsize(dest):,} bytes")


def _norm(s):
    return " ".join(str(s or "").split())


def _label(sh, row):
    return _norm(" ".join(_norm(sh.cell_value(row, c)) for c in (0, 1)))


def read():
    """Return (cells, groups_from_district_file).

    `cells` is {ressort_name: {category: int}} plus a `Total` key for the national column.
    """
    import xlrd

    p = os.path.join(RAW, RESSORT_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    sh = xlrd.open_workbook(p).sheet_by_name(SHEET)

    names = [_norm(sh.cell_value(HEADER_ROW, c)) for c in range(FIRST_COL, sh.ncols)]
    if len(names) != EXPECTED:
        raise SystemExit(f"{SHEET} has {len(names)} ressort columns, expected {EXPECTED} "
                         "-- ABS has reissued the file")
    want = [nm for _, nm, _ in RESSORTEN]
    if names != want:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(names, want)) if a != b]
        raise SystemExit(f"ressort column order has changed at {bad[:3]} -- RESSORTEN and "
                         "its pcodes must be revisited")

    # The religion block: find `Religion` and take the six labelled rows under it.
    start = next((r for r in range(sh.nrows) if _label(sh, r).lower().endswith("religion")),
                 None)
    if start is None:
        raise SystemExit(f"{SHEET} has no `Religion` block")
    cells = {"Total": {}}
    for nm in names:
        cells[nm] = {}
    row = start + 1
    for cat in [TOTAL_CAT] + CATEGORIES:
        got = _label(sh, row)
        if got != cat:
            raise SystemExit(f"{SHEET} row {row}: expected {cat!r}, read {got!r} -- ABS "
                             "has changed the religion category list and taxonomy/"
                             "sr2004.py must be revisited")
        cells["Total"][cat] = int(round(float(sh.cell_value(row, 2))))
        for k, nm in enumerate(names):
            v = sh.cell_value(row, FIRST_COL + k)
            if not isinstance(v, (int, float)):
                raise SystemExit(f"{SHEET} {nm}/{cat} is {v!r}, not a number")
            cells[nm][cat] = int(round(float(v)))
        row += 1

    # The district file supplies the grouping the ressort file lacks.
    dp = os.path.join(RAW, DISTRICT_NAME)
    if not os.path.exists(dp):
        raise SystemExit(f"missing {dp} -- run with --fetch first")
    ds = xlrd.open_workbook(dp).sheet_by_index(0)
    hdr = [_norm(ds.cell_value(0, c)) for c in range(ds.ncols)]
    groups = []
    for c in range(3, ds.ncols):
        n = ds.cell_value(2, c)
        if isinstance(n, float) and n > 0:
            groups.append((hdr[c], int(n)))
    return cells, groups


def rows_from(cells):
    rows = []
    for dist, nm, pcode in RESSORTEN:
        for cat in [TOTAL_CAT] + CATEGORIES:
            note = f"level=ressort; district={dist}"
            if cat == TOTAL_CAT:
                note += "; unit total, not a religion category"
            rows.append({"geo_id": pcode, "geo_level": "ressort", "geo_name": nm,
                         "source_category": cat, "count": cells[nm][cat],
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": note})
    return rows


def check(rows, cells, groups):
    ok = True
    names = [nm for _, nm, _ in RESSORTEN]

    units = {r["geo_id"] for r in rows}
    good = len(units) == EXPECTED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} ressort   {len(units):>4} units "
          f"(expected {EXPECTED})")

    good = cells["Total"][TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {cells['Total'][TOTAL_CAT]:,} "
          f"(expected {NATIONAL:,})")

    # The district file's grouping is what assigns each column a district. It has to agree
    # with this file's own list, or the district labels -- and the pcodes -- are wrong.
    want = GROUPS
    good = groups == want
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} district-profiel-census.xls gives ressorten per "
          f"district as\n      {[n for _, n in groups]}")
    if not good:
        print(f"      expected {want}")
    tot = sum(n for _, n in groups)
    good = tot == EXPECTED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} they sum to {tot} (expected {EXPECTED}) and "
          "consume the columns in order")

    bad = [n for n in ["Total"] + names
           if sum(cells[n][c] for c in CATEGORIES) != cells[n][TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 6 categories sum to the unit total on all "
          f"{len(names) + 1} columns ({len(bad)} failures) {bad[:4]}")

    bad = []
    for cat in [TOTAL_CAT] + CATEGORIES:
        s = sum(cells[n][cat] for n in names)
        if s != cells["Total"][cat]:
            bad.append((cat, s, cells["Total"][cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 62 ressorten sum to the Total column on "
          f"all {len(CATEGORIES) + 1} rows")
    for c, s, w in bad[:5]:
        print(f"        {c}: {s:,} vs {w:,}")
    print("      an EXACT partition in both directions -- integers, no rounding, "
          "no suppression.")

    nat = cells["Total"]
    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in [TOTAL_CAT] + CATEGORIES:
        n = nat[cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    dk = nat["Don't know/No answer"]
    print(f"\n  the non-answer is {dk:,} people, {100.0 * dk / NATIONAL:.2f}% — larger "
          "than Trinidad's 11.10%\n  and the second largest on this map. Marked, never "
          "redistributed (§3.5).")

    print("\n  the three religions the Indo-Caribbean geography is about:")
    for cat in ("Hinduism", "Islam", "Christianity"):
        sh = sorted(((100.0 * cells[n][cat] / cells[n][TOTAL_CAT], n)
                     for n in names if cells[n][TOTAL_CAT] > 2000), reverse=True)
        top = "  ".join(f"{n} {v:.0f}%" for v, n in sh[:4])
        print(f"    {cat:<14} {nat[cat]:>8,}  {100.0 * nat[cat] / NATIONAL:5.2f}%   "
              f"top: {top}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    cells, groups = read()
    rows = rows_from(cells)
    check(rows, cells, groups)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
