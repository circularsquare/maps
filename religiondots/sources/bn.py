"""Brunei — 2021 Population and Housing Census (BPP 2021), religion by district.

Reads (or fetches) the census tables workbook and Annex A of the census report into
data/normalized/bn.csv. `sources/bn.md` is the write-up; `sources/bn_geo.py` builds the four
districts and the placement grid, and imports `read_c1` from here for its mukim witness.

## THE TABLE

Department of Economic Planning and Statistics (DEPS), *Report of the Population and Housing
Census (BPP) 2021: Demographic, Household and Housing Characteristics*, Annex A, **Table A4,
"Population by Religion, District and Sex, 2021"**, printed p.83, and the same table as sheet
`A4` of the tables workbook `EXCEL TABLE A-C.xls`. Counts in persons, males and females, for the
nation and the four districts, four religions:

    Islam  Christianity  Buddhism  Others

Nothing finer is published: the mukim tables (C1 to C10) carry residential status, age and sex
and no religion, and no table crosses religion with race.

## BOTH FILES SURVIVE ONLY ON WAYBACK

The new DEPS site (`deps.gov.bn`) links the old `deps.mofe.gov.bn` document library, which no
longer answers. The workbook has one digest in every capture from 2023-10-23 to 2025-09-08. Annex
A has two: the 2023-09-30 capture (`2WW3Z24P...`) is the first 1,048,576 bytes of the file with no
`%%EOF`, which PyMuPDF would open anyway, and every capture from 2024-06-24 on is the whole
31-page file pinned below.

## THE CHECKS

The workbook is the source; the PDF page is the same table read independently, and both must
equal the transcription below. Then the table has to close every way the annex allows:

    persons = males + females in every cell; districts sum to the total column; religions sum
        to the total row
    A4's district totals = A1's (population by residential status and district)
    A11 (religion by residential status) and A12 (a)-(d) (the same per district) close on A4
    A10 (a)-(d), age by religion per district, sums to A4 for every religion
    C1's 39 mukims sum to A1's district totals
    UNSD Demographic Yearbook table 28, Brunei Darussalam 2021, equals A4's national column

## THE QUESTIONNAIRE

2021 household form (`Q_BPP2021.pdf`, Wayback 20220626223640, p.10 as printed), item **E10
Ugama / Religion**: 1 Islam, 2 Kristian / Christianity, 3 Buddha / Buddhism, 4 Hindu / Hinduism,
5 Lain-lain / Others (Sila nyatakan / Please specify). There is no box for no religion and none
for not stated. Every published table folds code 4 into `Others`, so the tables' `Others` is
Hindus, everyone who wrote in an answer (another religion, or none), and whatever the office did
with a blank; A4's total is A1's whole population, so nobody is left out of the table.

Usage:
    python sources/bn.py --fetch    the workbook (~0.9 MB) and Annex A (~1.3 MB) from Wayback
    python sources/bn.py            normalise from data/raw/bn/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bn")
OUT = os.path.join(ROOT, "data", "normalized", "bn.csv")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fetch_checks import FetchCheckError, check_body, digest, wayback_raw   # noqa: E402

SOURCE_ID = "bn_bpp2021_annexA_tA4"
YEAR = 2021
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

_LIB = "https://deps.mofe.gov.bn/DEPD%20Documents%20Library/DOS/POP/2021/"
FILES = {
    "xls": dict(url=_LIB + "EXCEL%20TABLE%20A-C.xls", ts="20231023184321",
                path=os.path.join(RAW, "bpp2021_excel_table_A-C.xls"),
                digest="TBSZB7XQDR7RSAHO3CA4B7IKOZNPXJCS", size=856_576),
    "pdf": dict(url=_LIB + "ANNEX%20A.pdf", ts="20240624173053",
                path=os.path.join(RAW, "bpp2021_annex_A.pdf"),
                digest="TJ5JRHQ7FQH5DTOPRD4RWXIZWIWGZNY7", size=1_329_106),
}
PDF_PAGES = 31
PAGE_A4 = 5            # 0-based; printed p.83
A4_CAPTION = "Table A4 : Population by Religion, District and Sex, 2021"

DISTRICTS = ["Brunei Muara", "Belait", "Tutong", "Temburong"]
CATS = ["Islam", "Christianity", "Buddhism", "Others"]
TOTAL_ROW = "Jumlah/Total"
STATUSES = ["Brunei Citizens", "Permanent Residents", "Temporary Residents"]
AGES = ["00-04", "05-09", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
MUKIMS_PER_DISTRICT = {"Brunei Muara": 18, "Belait": 8, "Tutong": 8, "Temburong": 5}

# Table A4, persons, transcribed from printed p.83 and asserted equal to both copies.
A4 = {
    "Brunei Muara": (269_074, 20_076, 19_306, 10_074),
    "Belait":       (46_072, 7_028, 7_235, 5_196),
    "Tutong":       (39_763, 1_151, 1_104, 5_192),
    "Temburong":    (7_126, 1_207, 100, 1_011),
}
A4_DISTRICT_TOTAL = {"Brunei Muara": 318_530, "Belait": 65_531, "Tutong": 47_210,
                     "Temburong": 9_444}
A4_NATIONAL = (362_035, 29_462, 27_745, 21_473)
TOTAL = 440_715

UNSD_NAME = "Brunei Darussalam"
UNSD_CATS = {"Muslim": "Islam", "Christian": "Christianity", "Buddhist": "Buddhism",
             "Other Religions": "Others"}

NUM = re.compile(r"^\d{1,3}(?:,\d{3})*$")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    for kind, f in FILES.items():
        if os.path.exists(f["path"]) and os.path.getsize(f["path"]) == f["size"]:
            print("already have", f["path"])
            continue
        url = wayback_raw(f["ts"], f["url"])
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
            body = r.read()
        try:
            check_body(body, kind, where=url, pin_size=f["size"], pin_digest=f["digest"])
        except FetchCheckError as e:
            raise SystemExit(f"{os.path.basename(f['path'])}: {e}")
        with open(f["path"] + ".part", "wb") as fh:
            fh.write(body)
        os.replace(f["path"] + ".part", f["path"])
        print(f"wrote {f['path']} ({len(body):,} bytes)")


def workbook():
    import xlrd

    p = FILES["xls"]["path"]
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing — run: python sources/bn.py --fetch")
    return xlrd.open_workbook(p)


def cell(v, where):
    """One count cell: an int, or None where the sheet leaves it empty. Anything else stops."""
    if isinstance(v, float):
        if v < 0 or v != int(v):
            raise SystemExit(f"{where}: {v!r} is not a count")
        return int(v)
    if isinstance(v, str) and not v.strip():
        return None
    raise SystemExit(f"{where}: {v!r} is neither a count nor an empty cell")


def _text(v):
    return " ".join(str(v).split())


def read_grid(sh, lo, hi, headers, labels, what):
    """{label: {header: (persons, males, females)}} from rows lo..hi-1 of one sheet.

    Row labels are the English ones in column 0, each exactly once in the block among rows that
    carry a number. The Malay label sits on its own empty row above the English one, and for
    Islam the two are the same word, so a label row with no numbers is a heading and skipped.
    Each header is found by its text above the first label row (the header rows repeat `Islam`
    too, so what must be unique is the column); its Persons column is that column, with Males
    and Females the two after it, and the sub-header row is asserted to say so.
    """
    rows = {}
    for r in range(lo, hi):
        lab = _text(sh.cell_value(r, 0))
        if lab in labels and any(isinstance(v, float) for v in sh.row_values(r)[1:]):
            if lab in rows:
                raise SystemExit(f"{what}: row {lab!r} twice (rows {rows[lab]} and {r})")
            rows[lab] = r
    missing = [lab for lab in labels if lab not in rows]
    if missing:
        raise SystemExit(f"{what}: no row labelled {missing}")
    first = min(rows.values())
    cols = {}
    for h in headers:
        hit = {c for r in range(lo, first) for c in range(1, sh.ncols)
               if _text(sh.cell_value(r, c)) == h}
        if len(hit) != 1:
            raise SystemExit(f"{what}: header {h!r} is in columns {sorted(hit)}, expected one")
        cols[h] = hit.pop()
    sub = [r for r in range(lo, first) if _text(sh.cell_value(r, 1)) == "Persons"]
    if len(sub) != 1:
        raise SystemExit(f"{what}: {len(sub)} `Persons` sub-header rows")
    for h, c in cols.items():
        got = tuple(_text(sh.cell_value(sub[0], c + i)) for i in range(3))
        if got != ("Persons", "Males", "Females"):
            raise SystemExit(f"{what}: under {h!r} the sub-header reads {got}")
    return {lab: {h: tuple(cell(sh.cell_value(r, c + i), f"{what} {lab!r} {h!r}")
                           for i in range(3))
                  for h, c in cols.items()}
            for lab, r in rows.items()}


def district_blocks(sh, what):
    """{district: (first row, row after last)} for a sheet printed as blocks (a)-(d)."""
    starts = {}
    for r in range(sh.nrows):
        line = " ".join(_text(v) for v in sh.row_values(r)).upper()
        for d in DISTRICTS:
            if f"DAERAH {d.upper()} /" in line:
                if d in starts:
                    raise SystemExit(f"{what}: block {d} starts twice")
                starts[d] = r
    if list(starts) != DISTRICTS:
        raise SystemExit(f"{what}: blocks {list(starts)}, expected {DISTRICTS} in that order")
    bounds = sorted(starts.values()) + [sh.nrows]
    return {d: (starts[d], bounds[bounds.index(starts[d]) + 1]) for d in DISTRICTS}


def read_a4(wb):
    sh = wb.sheet_by_name("A4")
    return read_grid(sh, 0, sh.nrows, ["Total"] + DISTRICTS, CATS + [TOTAL_ROW], "A4")


def read_a1(wb):
    sh = wb.sheet_by_name("A1")
    return read_grid(sh, 0, sh.nrows, ["Total"] + DISTRICTS, STATUSES + [TOTAL_ROW], "A1")


def read_a11(wb):
    sh = wb.sheet_by_name("A11")
    return read_grid(sh, 0, sh.nrows, ["Total"] + STATUSES, CATS + [TOTAL_ROW], "A11")


def read_a12(wb):
    sh = wb.sheet_by_name("A12 (a)-(d)")
    return {d: read_grid(sh, lo, hi, ["Total"] + STATUSES, CATS + [TOTAL_ROW], f"A12 {d}")
            for d, (lo, hi) in district_blocks(sh, "A12").items()}


def read_a10(wb):
    sh = wb.sheet_by_name("A10 (a)-(d)")
    return {d: read_grid(sh, lo, hi, ["Total"] + CATS, AGES + [TOTAL_ROW], f"A10 {d}")
            for d, (lo, hi) in district_blocks(sh, "A10").items()}


def read_c1(wb):
    """Table C1, persons by mukim: {district: {mukim: persons}}, with every row's persons asserted
    equal to males plus females and each district's mukims to its own printed row."""
    sh = wb.sheet_by_name("C1-C5")
    caps = [r for r in range(sh.nrows) if _text(sh.cell_value(r, 0)).startswith("Table C")]
    c1 = [r for r in caps if _text(sh.cell_value(r, 0)).startswith("Table C1 ")]
    after = [r for r in caps if not _text(sh.cell_value(r, 0)).startswith("Table C1 ")]
    if not c1 or not after:
        raise SystemExit("C1-C5: cannot bound Table C1 by its captions")
    lo, hi = c1[0], min(after)
    heads = {r for r in range(lo, hi) if _text(sh.cell_value(r, 1)) == "Total"}
    if len(heads) != len(DISTRICTS):
        raise SystemExit(f"C1: {len(heads)} `Total` header rows in column 1, expected one per district")
    out, printed, cur = {}, {}, None
    for r in range(lo, hi):
        raw = str(sh.cell_value(r, 0))
        name = _text(raw)
        if not name or not isinstance(sh.cell_value(r, 1), float):
            continue
        p, m, f = (cell(sh.cell_value(r, c), f"C1 {name!r}") for c in (1, 2, 3))
        if p != m + f:
            raise SystemExit(f"C1 {name!r}: persons {p} != males {m} + females {f}")
        if raw == raw.lstrip() and name in DISTRICTS:
            cur = name
            printed[cur] = p
            out[cur] = {}
        elif raw != raw.lstrip() and cur is not None:
            if name in out[cur]:
                raise SystemExit(f"C1: mukim {name!r} twice in {cur}")
            out[cur][name] = p
        else:
            raise SystemExit(f"C1 row {r}: {raw!r} is neither a district nor an indented mukim")
    if list(out) != DISTRICTS:
        raise SystemExit(f"C1 districts {list(out)}, expected {DISTRICTS}")
    for d in DISTRICTS:
        if sum(out[d].values()) != printed[d]:
            raise SystemExit(f"C1 {d}: mukims sum to {sum(out[d].values())}, row prints {printed[d]}")
    return out


def read_a4_pdf(doc):
    """Table A4 off printed p.83: {English row label: 15 counts}, Total then the four districts,
    each persons, males, females. The text layer prints the Malay label, the English label and
    then the row's numbers; a line can hold two numbers."""
    lines = [_text(x) for x in doc.load_page(PAGE_A4).get_text().splitlines()]
    lines = [x for x in lines if x]
    if A4_CAPTION not in lines:
        raise SystemExit(f"page index {PAGE_A4} has no {A4_CAPTION!r}")
    out = {}
    for i, ln in enumerate(lines):
        if ln not in CATS + [TOTAL_ROW] or ln in out:
            continue
        toks, j = [], i + 1
        while j < len(lines) and len(toks) < 15:
            parts = lines[j].split()
            if not all(NUM.match(p) for p in parts):
                break
            toks += parts
            j += 1
        if len(toks) == 15:
            out[ln] = tuple(int(t.replace(",", "")) for t in toks)
    return out


def check(wb, doc):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Brunei — BPP 2021, Annex A, Table A4\n")
    for kind, f in FILES.items():
        with open(f["path"], "rb") as fh:
            d = digest(fh.read())
        say(d == f["digest"], f"{os.path.basename(f['path'])} digest is the pinned {f['digest']}")
    say(doc.page_count == PDF_PAGES, f"Annex A is {doc.page_count} pages (expected {PDF_PAGES})")

    # 1. the workbook, the PDF page and the transcription are one table
    a4 = read_a4(wb)
    xls_persons = {d: tuple(a4[c][d][0] for c in CATS) for d in DISTRICTS}
    say(xls_persons == A4, "sheet A4's persons by district equal the transcription")
    say(tuple(a4[c]["Total"][0] for c in CATS) == A4_NATIONAL
        and a4[TOTAL_ROW]["Total"][0] == TOTAL,
        f"sheet A4's national column is {A4_NATIONAL}, total {TOTAL:,}")
    say({d: a4[TOTAL_ROW][d][0] for d in DISTRICTS} == A4_DISTRICT_TOTAL,
        "sheet A4's district totals equal the transcription")
    pdf = read_a4_pdf(doc)
    want = {lab: tuple(v for h in ["Total"] + DISTRICTS for v in a4[lab][h])
            for lab in CATS + [TOTAL_ROW]}
    say(pdf == want, f"Table A4 parsed off printed p.83: {len(pdf)} rows x 15 counts, identical "
                     "to the workbook")
    if pdf != want:
        for k in want:
            if pdf.get(k) != want[k]:
                print(f"        {k!r}: page {pdf.get(k)} workbook {want[k]}")

    # 2. the table closes on itself
    bad = [(lab, h) for lab in a4 for h in a4[lab] if a4[lab][h][0] != a4[lab][h][1] + a4[lab][h][2]]
    say(not bad, f"persons = males + females in all {len(a4) * 5} cells of A4 {bad or ''}")
    bad = [(lab, i) for lab in a4 for i in range(3)
           if a4[lab]["Total"][i] != sum(a4[lab][d][i] for d in DISTRICTS)]
    say(not bad, f"the four districts sum to the total column, persons, males and females {bad or ''}")
    bad = [(h, i) for h in ["Total"] + DISTRICTS for i in range(3)
           if a4[TOTAL_ROW][h][i] != sum(a4[c][h][i] for c in CATS)]
    say(not bad, f"the four religions sum to the total row in every column {bad or ''}")

    # 3. A1: the same districts, by residential status
    a1 = read_a1(wb)
    say(all(a1[TOTAL_ROW][h] == a4[TOTAL_ROW][h] for h in ["Total"] + DISTRICTS),
        "A1's district totals (persons, males, females) equal A4's")
    say(all(a1[TOTAL_ROW][h][i] == sum(a1[s][h][i] for s in STATUSES)
            for h in ["Total"] + DISTRICTS for i in range(3)),
        "A1's three residential statuses sum to its totals")

    # 4. A11 and A12: religion by residential status, nationally and per district
    a11 = read_a11(wb)
    say(all(a11[lab]["Total"] == a4[lab]["Total"] for lab in CATS + [TOTAL_ROW]),
        "A11's total column equals A4's national column")
    say(all(a11[lab]["Total"][i] == sum(a11[lab][s][i] for s in STATUSES)
            for lab in CATS + [TOTAL_ROW] for i in range(3)),
        "A11's statuses sum to its total column")
    a12 = read_a12(wb)
    say(all(a12[d][lab]["Total"] == a4[lab][d] for d in DISTRICTS for lab in CATS + [TOTAL_ROW]),
        "A12 (a)-(d)'s total columns equal A4's district columns")
    say(all(a12[d][lab]["Total"][i] == sum(a12[d][lab][s][i] for s in STATUSES)
            for d in DISTRICTS for lab in CATS + [TOTAL_ROW] for i in range(3)),
        "A12's statuses sum to its totals in every district")
    say(all(a11[lab][s][i] == sum(a12[d][lab][s][i] for d in DISTRICTS)
            for lab in CATS + [TOTAL_ROW] for s in STATUSES for i in range(3)),
        "A12's four districts sum to A11 for every religion and status")
    say(all(a12[d][TOTAL_ROW][s] == a1[s][d] for d in DISTRICTS for s in STATUSES),
        "A12's status totals per district equal A1's")

    # 5. A10: age by religion per district
    a10 = read_a10(wb)
    blanks = [(d, age, h) for d in DISTRICTS for age in AGES for h in a10[d][age]
              if None in a10[d][age][h]]
    z = lambda t: tuple(0 if v is None else v for v in t)   # noqa: E731
    say(all(a10[d][TOTAL_ROW][c] == a4[c][d] for d in DISTRICTS for c in CATS)
        and all(a10[d][TOTAL_ROW]["Total"] == a4[TOTAL_ROW][d] for d in DISTRICTS),
        "A10 (a)-(d)'s total rows equal A4 for every religion in every district")
    say(all(a10[d][TOTAL_ROW][h][i] == sum(z(a10[d][age][h])[i] for age in AGES)
            for d in DISTRICTS for h in ["Total"] + CATS for i in range(3)),
        f"A10's 18 age rows sum to its total rows everywhere, reading the {len(blanks)} empty "
        "cells as zero")

    # 6. C1: mukims
    c1 = read_c1(wb)
    say({d: len(m) for d, m in c1.items()} == MUKIMS_PER_DISTRICT,
        f"C1 lists {sum(len(m) for m in c1.values())} mukims, "
        f"{ {d: len(m) for d, m in c1.items()} }")
    say(all(sum(c1[d].values()) == a1[TOTAL_ROW][d][0] for d in DISTRICTS),
        "C1's mukims sum to A1's district totals")

    # 7. UNSD Demographic Yearbook table 28
    import oracle as unsd
    got = unsd.oracle(UNSD_NAME, YEAR)
    if got is None:
        say(False, f"UNSD has no {UNSD_NAME} {YEAR} row (run python tools/oracle.py --fetch)")
    else:
        cats, total, exact = unsd.partition(got.get(unsd.TOTAL, {}))
        mine = {UNSD_CATS.get(k, k): v for k, v in cats.items()}
        say(exact and total == TOTAL and mine == dict(zip(CATS, A4_NATIONAL)),
            f"UNSD table 28, {UNSD_NAME} {YEAR}: {cats}, total {total:,}, equal to A4 to the person")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return a4, a10, a12


def emit():
    rows = []
    for d in DISTRICTS:
        for c, n in zip(CATS, A4[d]):
            if n <= 0:
                continue
            rows.append({
                "geo_id": d, "geo_level": "district", "geo_name": d,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": f"Table A4 persons; district total {A4_DISTRICT_TOTAL[d]}",
            })
    return rows


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    for f in FILES.values():
        if not os.path.exists(f["path"]):
            raise SystemExit(f"{f['path']} missing — run: python sources/bn.py --fetch")
    a4, a10, a12 = check(workbook(), fitz.open(FILES["pdf"]["path"]))
    rows = emit()

    total = sum(r["count"] for r in rows)
    print(f"\n  4 districts, {total:,} people")
    for c, n in zip(CATS, A4_NATIONAL):
        print(f"    {n:>9,}  {100.0 * n / total:6.3f}%  {c}")
    print(f"\n  {'district':<14}{'people':>9}  " + "  ".join(f"{c:>12}" for c in CATS))
    for d in DISTRICTS:
        t = A4_DISTRICT_TOTAL[d]
        print(f"  {d:<14}{t:>9,}  " + "  ".join(f"{100.0 * n / t:11.2f}%" for n in A4[d]))
    print("\n  by residential status (A12), citizens / permanent / temporary:")
    for c in CATS:
        parts = []
        for d in DISTRICTS:
            v = [a12[d][c][s][0] for s in STATUSES]
            parts.append(f"{d} {v[0]:,}/{v[1]:,}/{v[2]:,}")
        print(f"    {c:<13} " + "; ".join(parts))
    print("\n  share under 15 (A10), Others against the district:")
    for d in DISTRICTS:
        young = lambda h: sum((a10[d][a][h][0] or 0) for a in AGES[:3])   # noqa: E731
        print(f"    {d:<14} Others {100.0 * young('Others') / a10[d][TOTAL_ROW]['Others'][0]:5.1f}%"
              f"   everyone {100.0 * young('Total') / a10[d][TOTAL_ROW]['Total'][0]:5.1f}%"
              f"   Others male {100.0 * a4['Others'][d][1] / a4['Others'][d][0]:5.1f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
