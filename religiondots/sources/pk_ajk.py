"""Pakistan, Azad Jammu and Kashmir: 2023 census religion by district, from the AJK government's
own Statistical Year Book 2025.

Imported by sources/pk_2023.py, which appends these ten districts to data/normalized/pk.csv, and
by sources/pk_2023_geo.py, which reads `tehsils()` as the second key on the boundary join.
`sources/pk.md` §10 is the write-up.

WHY THIS SOURCE. PBS's own 2023 Table 9 covers the four provinces and Islamabad only; Azad
Kashmir and Gilgit-Baltistan were enumerated by the same census and are in no PBS table
(sources/pk.md §9.3). The AJK Bureau of Statistics (Planning & Development Department,
Muzaffarabad) prints the census's AJK results in its yearbook, citing "Population & Housing
Census Report 2023, Pakistan Bureau of Statistics":

    https://pndajk.gov.pk/uploadfiles/downloads/Statistical%20Year%20Book%202025.pdf
    (listed at https://pndajk.gov.pk/statyearbook.php; 295 pages, 11.5 MB)

    Table 15.23  Rural & Urban Population by Religion of AJ&K (Census 2023)     p.187 (pdf 223)
    Table 15.24  District wise Population of AJ&K by Religion (Census 2023)     p.187 (pdf 223)
    Table 15.15  Tehsil-wise Area, Population and Nos. of Households (Census 2023) p.184 (pdf 220)

THE SAME EIGHT CATEGORIES AS PBS TABLE 9, in a different column order and with two shortened
headers: the yearbook prints `Hindu` for Table 9's `Hindu Jati` and `Scheduled Caste` for
`Scheduled Castes`. They are written to pk.csv under Table 9's names, because they are the same
answers to the same question on the same census form; HEADER_TO_T9 is the whole translation.
The 2023 yearbook's 2017 tables (Year Book 2023, Table 15.24) carry six categories and are not
used: 2023 is the vintage of the rest of Pakistan.

READ BY GEOMETRY. Words are grouped into rows by height; a row is its label words plus its
numbers sorted left to right, and must carry exactly the table's column count. Asserted:
  * the header words sit in the order HEADER says, on both religion tables;
  * every row of 15.24: Total = the eight religions; the AJ&K row = the ten districts, on every
    column;
  * 15.23: Total = the eight on all 12 rows; Male + Female + Transgender = All Sexes, and Rural +
    Urban = AJ&K, on every column; its AJ&K All Sexes row equals 15.24's AJ&K row;
  * 15.15's 32 tehsils, read in printed order, fall into the ten districts by exact cumulative
    sums of the 2023 population, district by district in 15.24's order, with nothing left over;
    and 15.15's Total equals 15.24's.

Usage:
    python sources/pk_ajk.py --fetch      one GET, 11.5 MB, then print the checks
    python sources/pk_ajk.py              print the checks from data/raw/pk2023/
"""

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pk2023")
PDF_NAME = "ajk_statistical_year_book_2025.pdf"
PDF = os.path.join(RAW, PDF_NAME)
URL = "https://pndajk.gov.pk/uploadfiles/downloads/Statistical%20Year%20Book%202025.pdf"
LEAST_BYTES = 10_000_000
PAGES = 295

PROVINCE = "Azad Jammu and Kashmir"
SOURCE_ID = "pk_census_2023_ajk_yearbook2025_t15_24"

# Table 9's eight, in pk_2023.CATS order
T9 = ["Muslim", "Christian", "Hindu Jati", "Qadiani/Ahmadi", "Scheduled Castes", "Sikh",
      "Parsi", "Others"]
# the yearbook's columns 2..9, left to right, and their Table 9 names
HEADER = ["Muslim", "Hindu", "Christian", "Qadiani/Ahmadi", "Scheduled Caste", "Sikh", "Parsi",
          "Others"]
HEADER_TO_T9 = {"Muslim": "Muslim", "Hindu": "Hindu Jati", "Christian": "Christian",
                "Qadiani/Ahmadi": "Qadiani/Ahmadi", "Scheduled Caste": "Scheduled Castes",
                "Sikh": "Sikh", "Parsi": "Parsi", "Others": "Others"}
# the first word of each header, as the page sets them (Qadiani/ and Scheduled wrap)
HEADER_WORDS = ["Muslim", "Hindu", "Christian", "Qadiani/", "Scheduled", "Sikh", "Parsi",
                "Others"]

# the yearbook's district order, which is also the order 15.15 prints the tehsils in
DISTRICTS = ["Muzaffarabad", "Neelum", "Jhelum Valley", "Bagh", "Haveli", "Poonch", "Sudhnoti",
             "Kotli", "Mirpur", "Bhimber"]
TEHSIL_COUNT = 32

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

_NUM = re.compile(r"^-$|^\d{1,3}(,\d{3})+$|^\d+$")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) > LEAST_BYTES:
        print("already have", PDF)
        return
    r = requests.get(URL, timeout=600, headers=UA)
    r.raise_for_status()
    if r.content[:5] != b"%PDF-" or b"%%EOF" not in r.content[-2048:]:
        raise SystemExit(f"{URL}: not a whole PDF (starts {r.content[:16]!r})")
    if len(r.content) < LEAST_BYTES:
        raise SystemExit(f"{URL}: only {len(r.content):,} bytes")
    with open(PDF, "wb") as fh:
        fh.write(r.content)
    print(f"wrote {PDF} ({len(r.content):,} bytes)")


def _num(t):
    return 0 if t == "-" else int(t.replace(",", ""))


def _rows(page, y0, y1):
    """Words between heights y0 and y1, grouped into rows: [(y, [(x0, x1, text), ...])]."""
    words = [w for w in page.get_text("words") if y0 < (w[1] + w[3]) / 2 < y1]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - yc) < 3:
            rows[-1][1].append((w[0], w[2], w[4]))
        else:
            rows.append([yc, [(w[0], w[2], w[4])]])
    for r in rows:
        r[1].sort()
    return rows


def _find_page(doc, *needles):
    hits = [i for i in range(doc.page_count)
            if all(re.search(n, doc[i].get_text()) for n in needles)]
    if len(hits) != 1:
        raise SystemExit(f"{PDF_NAME}: {len(hits)} pages match {needles}, expected one")
    return doc[hits[0]], hits[0] + 1


def _title_y(page, table):
    """Height of the `Table: 15.xx` title (the page sets it as `Table:15.23` or `Table: 15.24`)."""
    for y, ws in _rows(page, 0, 1e9):
        txt = " ".join(t for _, _, t in ws).replace("Table:", "Table: ")
        if re.search(rf"Table:\s*{re.escape(table)}\b", txt):
            return y
    raise SystemExit(f"{PDF_NAME}: no title for Table {table}")


def _check_header(page, y0, y1, table):
    rows = _rows(page, y0, y1)
    for y, ws in rows:
        txt = [t for _, _, t in ws]
        if "Muslim" in txt and "Others" in txt:
            # `Qadiani/` and `Scheduled` wrap, so they sit a line above; read the band by x
            band = sorted((x0, t) for yy, w2 in rows if abs(yy - y) < 9 for x0, _, t in w2)
            got = [t for _, t in band if t in HEADER_WORDS]
            if got != HEADER_WORDS:
                raise SystemExit(f"Table {table}: header order {got}, expected {HEADER_WORDS}")
            return
    raise SystemExit(f"Table {table}: no header row with Muslim ... Others")


def _data_rows(page, y0, y1, ncol):
    out = []
    for y, ws in _rows(page, y0, y1):
        nums = [t for _, _, t in ws if _NUM.match(t)]
        label = " ".join(t for _, _, t in ws if not _NUM.match(t))
        if not nums or label.startswith("Source") or label == "":
            continue
        if len(nums) != ncol:
            continue      # the column-number row `1 2 ... 10` is caught by the label test below
        out.append((label, [_num(t) for t in nums], y))
    return out


def read(verbose=True):
    """-> (units, tehsil_map). units are pk_2023.py's dicts, cells = [Total] + T9 order."""
    import fitz

    with open(PDF, "rb") as fh:
        raw = fh.read()
    if raw[:5] != b"%PDF-" or b"%%EOF" not in raw[-2048:] or len(raw) < LEAST_BYTES:
        raise SystemExit(f"{PDF}: not the whole yearbook -- run: python sources/pk_ajk.py --fetch")
    doc = fitz.open(PDF)
    if doc.page_count != PAGES:
        raise SystemExit(f"{PDF_NAME}: {doc.page_count} pages, expected {PAGES}")
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= good
        if verbose:
            print(f"  {'OK ' if good else 'BAD'} {msg}")

    # `Table:` is what keeps the list of tables (pdf p.17), which names it too, out
    page, pno = _find_page(doc, r"Table:\s*15\.24", r"District wise Population of AJ&K by Religion "
                                           r"\(Census 2023\)")
    y23, y24 = _title_y(page, "15.23"), _title_y(page, "15.24")
    _check_header(page, y23, y24, "15.23")
    _check_header(page, y24, 1e9, "15.24")

    # ---- 15.24, district by religion
    rows24 = [r for r in _data_rows(page, y24, 1e9, 9) if not r[0].isdigit()]
    by = {lab: cells for lab, cells, _ in rows24}
    good = list(by) == DISTRICTS + ["AJ&K"]
    say(good, f"Table 15.24 (pdf p.{pno}): rows {list(by)}")
    if not good:
        raise SystemExit("Table 15.24 rows are not the ten districts and AJ&K")
    for lab, c in by.items():
        if sum(c[:8]) != c[8]:
            say(False, f"15.24 {lab}: religions {sum(c[:8]):,} != total {c[8]:,}")
    say(all(sum(c[:8]) == c[8] for c in by.values()),
        "15.24: every row's eight religions add to its total")
    colsum = [sum(by[d][i] for d in DISTRICTS) for i in range(9)]
    say(colsum == by["AJ&K"], f"15.24: the ten districts sum to the AJ&K row on all 9 columns "
                              f"({by['AJ&K'][8]:,})")

    # ---- 15.23, area by sex by religion
    rows23 = [r for r in _data_rows(page, y23, y24, 9) if not r[0].isdigit()]
    sections, cur, t23 = ["Rural", "Urban", "AJ&K"], None, {}
    marks = [(y, " ".join(t for _, _, t in ws)) for y, ws in _rows(page, y23, y24)]
    for lab, cells, y in rows23:
        sec = [m for my, m in marks if my < y and m in sections]
        cur = sec[-1] if sec else None
        t23[(cur, lab)] = cells
    sexes = ["All Sexes", "Male", "Female", "Transgender"]
    want = {(s, x) for s in sections for x in sexes}
    say(set(t23) == want, f"Table 15.23: 12 rows, Rural/Urban/AJ&K by four sexes")
    if set(t23) != want:
        raise SystemExit(f"15.23 rows {sorted(t23)}")
    say(all(sum(c[:8]) == c[8] for c in t23.values()), "15.23: every row's religions add")
    say(all(t23[(s, "All Sexes")][i] == sum(t23[(s, x)][i] for x in sexes[1:])
            for s in sections for i in range(9)), "15.23: male + female + transgender = all")
    say(all(t23[("AJ&K", x)][i] == t23[("Rural", x)][i] + t23[("Urban", x)][i]
            for x in sexes for i in range(9)), "15.23: rural + urban = AJ&K, all columns")
    say(t23[("AJ&K", "All Sexes")] == by["AJ&K"],
        "15.23's AJ&K row equals 15.24's, cell by cell (two tables, one census)")

    # ---- 15.15, tehsils, as the second key for the boundary join
    tpage, tpno = _find_page(doc, r"Table:\s*15\.15", r"Tehsil-wise Area, Population and Nos\. of Households of "
                                  r"AJ&K \(Census 2023\)")
    y15, y16 = _title_y(tpage, "15.15"), _title_y(tpage, "15.16")
    teh = []
    for y, ws in _rows(tpage, y15, y16):
        label = " ".join(t for _, _, t in ws if not re.match(r"^-?[\d,.]+$", t))
        ints = [t for _, _, t in ws if re.match(r"^\d{1,3}(,\d{3})+$", t)]
        if len(ints) == 2 and label and not label.startswith(("Source", "Table")):
            teh.append((label, _num(ints[0])))
    total = [p for lab, p in teh if lab == "Total"]
    teh = [(lab, p) for lab, p in teh if lab != "Total"]
    say(len(teh) == TEHSIL_COUNT and total == [by["AJ&K"][8]],
        f"Table 15.15 (pdf p.{tpno}): {len(teh)} tehsils, Total {total} = 15.24's AJ&K")
    tehsil_map, i = {}, 0
    for d in DISTRICTS:
        acc, names = 0, []
        while i < len(teh) and acc < by[d][8]:
            acc += teh[i][1]
            names.append(teh[i][0])
            i += 1
        if acc != by[d][8]:
            say(False, f"15.15: tehsils for {d} sum to {acc:,}, district is {by[d][8]:,}")
        tehsil_map[d] = names
    say(i == len(teh) and all(sum(p for lab, p in teh if lab in tehsil_map[d]) == by[d][8]
                              for d in DISTRICTS),
        "15.15: the tehsils, in printed order, fall into the ten districts by exact cumulative "
        "sums of population, nothing left over")
    if not ok:
        raise SystemExit("AJK yearbook reconciliation FAILED")

    t9_to_col = {HEADER_TO_T9[h]: i for i, h in enumerate(HEADER)}
    units = []
    for d in DISTRICTS + ["AJ&K"]:
        c = by[d]
        cells = [c[8]] + [c[t9_to_col[cat]] for cat in T9]
        is_prov = d == "AJ&K"
        units.append({"level": "province" if is_prov else "district", "province": PROVINCE,
                      "district": None if is_prov else f"{d.upper()} DISTRICT",
                      "name": PROVINCE.upper() if is_prov else f"{d.upper()} DISTRICT",
                      "cells": cells, "page": pno, "file": PDF_NAME, "source_id": SOURCE_ID})
    return units, {f"{d.upper()} DISTRICT": v for d, v in tehsil_map.items()}


def tehsils():
    return read(verbose=False)[1]


def main():
    if "--fetch" in sys.argv:
        fetch()
    print("Azad Jammu and Kashmir -- 2023 census religion, AJK Statistical Year Book 2025\n")
    units, tmap = read()
    ds = [u for u in units if u["level"] == "district"]
    tot = sum(u["cells"][0] for u in ds)
    print(f"\n  {len(ds)} districts, {tot:,} people. Shares:")
    for i, cat in enumerate(T9, start=1):
        n = sum(u["cells"][i] for u in ds)
        print(f"    {n:>10,}  {100.0 * n / tot:7.4f}%  {cat}")
    for d, t in tmap.items():
        print(f"    {d:24s} {t}")


if __name__ == "__main__":
    main()
