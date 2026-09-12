"""Zimbabwe — ZIMSTAT, 2022 Population and Housing Census Report, Table 2.14.

Reads (or fetches) data/raw/zw/ and writes data/normalized/zw.csv.

**Eleven categories on 10 provinces for 15,178,957 people.** That is ~1.5M per unit and it
is **the coarsest counting geography anywhere on this map** — coarser than Kenya's 47
counties (1.0M) and than Guyana's 10 regions (75k), which were the previous extremes at
either end. Spec §3.9b is what makes it drawable: there is no minimum unit count, the rule
is to take the finest geography a country publishes and say what it therefore cannot show.
ZIMSTAT publishes religion at province and nowhere else.

**IT IS WORTH DRAWING FOR ONE CATEGORY.** `Apostolic Sect` is **6,112,503 people, 40.3% of
Zimbabwe, and the largest single religious answer in the country** — the Vapostori, the
indigenous prophetic churches founded by Johane Marange and Johane Masowe in the 1930s. No
other source on this map counts them, and they are nearly twice the size of everything
`christianity.africaninstituted` held before Zimbabwe arrived (Kenya 3.29M, Benin 676k).

**THE TABLE IS AN EXACT PARTITION AND THERE IS NO GAP AT ALL.** The eleven categories sum
to the province total on all ten rows and to 15,178,957 nationally, which is the whole
census count. No `not stated`, no residual, no §3.5 gap — the third source here of which
that is true, after Malawi and Guyana.

**THREE TABLES, AND THE OTHER TWO ARE THE CHECK.** Table 2.14(a) is Male, 2.14(b) Female,
2.14(c) both sexes; (a) and (b) share page 144 and (c) is on 145. Only (c) is drawn, and the
other two are read anyway because **`Male + Female == Total` on all 120 cells is free and is
the only check that would catch a column landing in the wrong table** — every other identity
here reconciles inside one table whichever way its columns were read. Malawi's panel rule,
on a source that lays the panels out differently.

**THE PARSE IS A PLAIN LINE READ**, unusually. The text layer gives the province name and
then its twelve figures one per line, in column order, so there is no geometry to do —
which is the reverse of Benin's booklets and of Malawi's rotated page. It is still checked
rather than trusted: the table title, the eleven header labels in order, and the ten
province names in order are all asserted before any figure is taken.

Usage:
    python sources/zw.py --fetch    one 11.8 MB PDF, seconds
    python sources/zw.py            normalise from data/raw/zw/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "zw")
OUT = os.path.join(ROOT, "data", "normalized", "zw.csv")

SOURCE_ID = "zw_phc_2022"
YEAR = 2022
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# ZIMSTAT is WordPress and its wp/v2/media library holds almost nothing — 19 documents, and
# not this one. The census reports are plain links on /population-census/ instead, under a
# `Census/` upload directory. Two of the three links on that page carry a DOUBLE SLASH
# (`uploads//Census/`) and this one does not; both forms serve.
PDF_URL = ("https://www.zimstat.co.zw/wp-content/uploads/Census/"
           "2022_PHC_Report_27012023_Final.pdf")
PDF_NAME = "zw_phc2022_report.pdf"

# 0-based page indices. (a) Male and (b) Female share a page; (c) is on the next.
PAGE_MF = 144
PAGE_TOTAL = 145

TITLES = {
    "male": re.compile(r"Table\s*2\.14\(a\)\s*:\s*Distribution of Male Population by "
                       r"Province and Religion", re.I),
    "female": re.compile(r"Table\s*2\.14\(b\)\s*:\s*Distribution of Female Population by "
                         r"Province and Religion", re.I),
    "total": re.compile(r"Table\s*2\.14\(c\)\s*:\s*Distribution of Population by "
                        r"Province and Religion", re.I),
}
DRAWN_PANEL = "total"

# In the order ZIMSTAT prints them, left to right. `Total` last is the row's own universe.
CATEGORIES = [
    "African Tradition",
    "Roman Catholic",
    "Protestant",
    "Apostolic Sect",
    "Pentecost",
    "Other Christian",
    "Islam",
    "Judaism",
    "Hinduism",
    "None",
    "Other",
]
TOTAL_CAT = "Total"

# In the order ZIMSTAT prints them, which is also its own province-code order — asserted
# against COD's `adm1_pcode` in sources/zw_geo.py rather than assumed here.
PROVINCES = [
    "Bulawayo",
    "Manicaland",
    "Mashonaland Central",
    "Mashonaland East",
    "Mashonaland West",
    "Matabeleland North",
    "Matabeleland South",
    "Midlands",
    "Masvingo",
    "Harare",
]
EXPECTED_PROVINCES = 10
NATIONAL = 15_178_957          # Table 2.14(c)'s own Total row

NUM = re.compile(r"^[\d,]+$")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 5_000_000:
        print("already have", dest)
        return
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, timeout=900, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # §5a: HTTP 200 is not a download. Assert size AND type.
    with open(dest, "rb") as fh:
        magic = fh.read(5)
    if magic != b"%PDF-":
        raise SystemExit(f"{dest} is not a PDF -- starts {magic!r}, "
                         f"{os.path.getsize(dest):,} bytes")
    print(f"  {os.path.getsize(dest):,} bytes")


def _cell(tok, where):
    if not NUM.match(tok):
        raise SystemExit(f"{where}: {tok!r} is not a figure -- Table 2.14 has been "
                         "re-typeset and the row would silently shift")
    return int(tok.replace(",", ""))


def _panel(lines, start, label):
    """One 2.14 table, read from the flat line list starting at its title.

    The text layer emits the province name and then its twelve figures, one per line, in
    column order. The header labels come first, also one per line, and they are asserted
    in order before any figure is taken — so a re-ordered column list stops the run rather
    than relabelling the map.
    """
    i = start
    # The eleven category labels plus `Total`, each on its own line or split over two.
    # Match them in order, allowing the wrapping ZIMSTAT does on the longer ones.
    want = CATEGORIES + [TOTAL_CAT]
    seen, buf = [], ""
    while i < len(lines) and len(seen) < len(want):
        txt = lines[i].strip()
        i += 1
        if not txt or txt == "Province":
            continue
        buf = (buf + " " + txt).strip()
        target = want[len(seen)]
        if buf.lower() == target.lower():
            seen.append(target)
            buf = ""
        elif not target.lower().startswith(buf.lower()):
            raise SystemExit(
                f"{label}: after {seen}, read {buf!r} where {target!r} was expected -- "
                "ZIMSTAT has changed Table 2.14's column list and taxonomy/zw2022.py "
                "must be revisited")
    if len(seen) != len(want):
        raise SystemExit(f"{label}: only found {len(seen)} of {len(want)} column headers")

    rows = {}
    for province in PROVINCES + ["Total"]:
        while i < len(lines) and not lines[i].strip():
            i += 1
        got = lines[i].strip() if i < len(lines) else "<eof>"
        if got.lower() != province.lower():
            raise SystemExit(f"{label}: expected the {province!r} row and read {got!r} -- "
                             "the province order has changed")
        i += 1
        vals = []
        while len(vals) < len(want) and i < len(lines):
            txt = lines[i].strip()
            i += 1
            if not txt:
                continue
            vals.append(_cell(txt, f"{label} {province!r}"))
        rows[province] = dict(zip(want, vals))
    return rows, i


def read():
    import fitz

    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)

    panels = {}
    for page_no, labels in ((PAGE_MF, ["male", "female"]), (PAGE_TOTAL, ["total"])):
        if page_no >= doc.page_count:
            raise SystemExit(f"{p} has {doc.page_count} pages, expected page {page_no}")
        lines = doc[page_no].get_text().splitlines()
        flat = " ".join(" ".join(lines).split())
        for label in labels:
            if not TITLES[label].search(flat):
                raise SystemExit(f"page {page_no} does not carry Table 2.14({label[0]}) -- "
                                 f"it starts {flat[:120]!r}. ZIMSTAT has re-paginated; "
                                 "find the table and update PAGE_MF / PAGE_TOTAL.")
        cursor = 0
        for label in labels:
            hit = next((j for j in range(cursor, len(lines))
                        if TITLES[label].search(" ".join(lines[j].split()))), None)
            if hit is None:
                raise SystemExit(f"page {page_no}: no line matching Table 2.14({label[0]})")
            panels[label], cursor = _panel(lines, hit + 1, label)

    rows = []
    for i, province in enumerate(PROVINCES, start=1):
        cells = panels[DRAWN_PANEL][province]
        for cat in CATEGORIES + [TOTAL_CAT]:
            note = "level=province"
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            rows.append({"geo_id": f"ZW{i:02d}", "geo_level": "province",
                         "geo_name": province, "source_category": cat,
                         "count": cells[cat], "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID, "note": note})
    return rows, panels


def check(rows, panels):
    ok = True

    units = {r["geo_id"] for r in rows}
    good = len(units) == EXPECTED_PROVINCES
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} province   {len(units):>4} units "
          f"(expected {EXPECTED_PROVINCES})")

    nat = panels[DRAWN_PANEL]["Total"]
    good = nat[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} national universe {nat[TOTAL_CAT]:,} "
          f"(expected {NATIONAL:,})")

    # ZIMSTAT neither suppresses nor rounds this table, so every identity is an equality.
    bad = [p for p in PROVINCES + ["Total"]
           if sum(panels[DRAWN_PANEL][p][c] for c in CATEGORIES)
           != panels[DRAWN_PANEL][p][TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 11 categories sum to Total on all "
          f"{len(PROVINCES) + 1} rows ({len(bad)} failures) {bad[:4]}")
    print("      an EXACT partition -- ZIMSTAT publishes no `not stated` cell and no "
          "residual,\n      so 100% of the census is drawn and there is no §3.5 gap.")

    bad = []
    for cat in CATEGORIES + [TOTAL_CAT]:
        s = sum(panels[DRAWN_PANEL][p][cat] for p in PROVINCES)
        if s != panels[DRAWN_PANEL]["Total"][cat]:
            bad.append((cat, s, panels[DRAWN_PANEL]["Total"][cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 10 provinces sum to the Total row on all "
          f"{len(CATEGORIES) + 1} columns")
    for c, s, w in bad[:5]:
        print(f"        {c}: {s:,} vs {w:,}")

    # The other two tables are the check on the READ, not on the census: every identity
    # above holds inside one table whichever way its columns were taken.
    bad = []
    for p in PROVINCES + ["Total"]:
        for cat in CATEGORIES + [TOTAL_CAT]:
            m, f, t = (panels["male"][p][cat], panels["female"][p][cat],
                       panels["total"][p][cat])
            if m + f != t:
                bad.append((p, cat, m + f, t))
    ok &= not bad
    n_cells = (len(PROVINCES) + 1) * (len(CATEGORIES) + 1)
    print(f"  {'OK ' if not bad else 'BAD'} Male + Female == Total on all {n_cells} cells "
          f"({len(bad)} failures)")
    for p, c, s, w in bad[:5]:
        print(f"        {p}/{c}: {s:,} vs {w:,}")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES + [TOTAL_CAT]:
        n = nat[cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>11,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, panels = read()
    check(rows, panels)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
