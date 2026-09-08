"""Kiribati — Kiribati NSO, 2015 Population Census, Report Volume 1, Table 6.

Reads (or fetches) data/raw/ki/ and writes data/normalized/ki.csv.

**RELIGION BY ISLAND, FOURTEEN CATEGORIES, AND THE UN HAS THE SAME NUMBERS.** `Table 6:
Population by island, sex and religion: 2015` runs over four pages of the 2015 report and gives
every one of Kiribati's **24 inhabited islands** its own row, 4,589 people each. The islands sum
to the printed national row **exactly, on all fifteen columns**, and the national row then
matches **UNSD Demographic Yearbook table 28 on all fourteen categories, to the person**, from
the return Kiribati forwarded rather than from this PDF.

**WHY NOT 2020, AND WHY NOT 2005.** The 2020 census is newer and publishes religion
**nationally only** (report Table G-3); its Census Atlas has a `Religious affiliation by island`
map but the map is a raster and carries no numbers. The per-island workbooks on `nso.gov.ki`
are finer still — religion by **village** — but they are the **2005** census, twenty years old,
and their column set differs island to island (Beru has an `AOG` column that Abaiang does not).
2015 is the newest year published with a geography, and its list is the longest of the three.

**THE ONE THING 2015 CANNOT SHOW.** The Kiribati Protestant Church became the **Kiribati
Uniting Church** in 2014, and by 2020 the census counted KUC at 21% and a continuing KPC at 8%
as separate answers. In 2015 they are still one cell, so this map draws the union whole and
cannot show the split. The 2015 report says so itself: its historical annex is headed *"KUC used
to be known as KPC"*.

**KIRIBATI IS THE MOST SPREAD-OUT COUNTRY ON EARTH AND IT STRADDLES THE ANTIMERIDIAN.** The
Gilberts sit at 173 E and Kiritimati at 157 W, 4,000 km apart, so the country's own bounding box
spans 351 degrees of longitude and every naive width check fires on it. See `ki_geo.py` for the
check that works. [[reference_antimeridian]]

Usage:
    python sources/ki.py --fetch    one ~3 MB PDF from nso.gov.ki
    python sources/ki.py            normalise from data/raw/ki/
"""

import csv
import os
import re
import sys
import unicodedata

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

RAW = os.path.join(ROOT, "data", "raw", "ki")
OUT = os.path.join(ROOT, "data", "normalized", "ki.csv")

SOURCE_ID = "ki_phc_2015"
YEAR = 2015
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# NSO runs WP File Download, the fifth Pacific office here to do so after Fiji, PNG, the
# Solomon Islands and Tonga ([[reference_wpfd_sweep]]). Its 482-file library was swept with
# `id=0`; this is the only file needed.
PDF_URL = ("https://nso.gov.ki/download/91/2015-census/1054/"
           "population-census-2015-report-volume-1-final.pdf")
PDF_NAME = "census_report_2015.pdf"

# Table 6 spans these pages, 0-indexed. Asserted by title rather than trusted.
PAGES = (56, 57, 58, 59)
TITLE = re.compile(r"Table 6.*Population by island, sex and religion", re.I)

# The 14 category columns in print order, reassembled from the table's three stacked header
# rows. `HEADER_TOKENS` is the bottom row as it extracts, and is asserted before any number is
# read, so a re-typeset table fails instead of silently shifting a column.
CATEGORIES = [
    "Roman Catholic",
    "KPC",
    "Seventh Day Adventist",
    "Church Of God",
    "Latter Day Saints",
    "Assembly of God",
    "Bahai",
    "Jehova's Witness (Te Koaua)",
    "Islam",
    "Four Square",
    "Te Ran",
    "All Nation",
    "No religion",
    "Other",
]
HEADER_TOKENS = ("Total Total Catholic KPC Adventist Of God Saints of God Bahai "
                 "(Te Koaua) Islam Square Ran Nation religion Other")

RESIDUALS = []          # `Other` is a named tail; Kiribati has no refusal or not-stated cell

NATIONAL = 110_136
EXPECTED_ISLANDS = 24

# UNSD table 28's 2015 spelling for the same fourteen cells. Two are worth noting: the
# Yearbook renders `KPC` as **`Kempsville Presbyterian Church`**, which is a false expansion
# of the initials — it is the Kiribati Protestant Church — and it splits the Witnesses' own
# Gilbertese name `Te koaua` off from the English. Pairing on an explicit table rather than on
# the string keeps a disagreement about a NUMBER from hiding behind one about a NAME.
ORACLE_ALIAS = {
    "Roman Catholic": "Catholic",
    "KPC": "Kempsville Presbyterian Church",
    "Seventh Day Adventist": "Seventh Day Adventist",
    "Church Of God": "Church of God",
    "Latter Day Saints": "Latter Day Saints (Mormon)",
    "Assembly of God": "Assembly of God",
    "Bahai": "Baha'i",
    "Jehova's Witness (Te Koaua)": "Te koaua",
    "Islam": "Muslim",
    "Four Square": "Church of the Foursquare Gospel",
    "Te Ran": "Te Ran",
    "All Nation": "All Nations Baptist",
    "No religion": "No Religion",
    "Other": "Other",
}


def fold(s):
    """Casefold, strip accents and punctuation — for COMPARING names, never for storing."""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print("already have", dest)
        return
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, headers=ua, timeout=900)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise SystemExit(f"nso.gov.ki returned something that is not a PDF "
                         f"({len(r.content):,} bytes)")
    if b"%%EOF" not in r.content[-4096:]:
        raise SystemExit("no %%EOF trailer — truncated at source "
                         "[[reference_pdf_truncated_at_source]]")
    tmp = dest + ".part"                                  # [[reference_wb_truncates]]
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, dest)
    print(f"  {os.path.getsize(dest):,} bytes")


def _lines(page):
    """The page's words grouped into visual rows, left to right. [[reference_pdf_table_geometry]]

    Table 6 is laid out cleanly enough that this is all the geometry needed: every data row is
    a label followed by exactly fifteen figures, and the only trap is the three-line stacked
    header, whose fragments look like island names. `parse()` cuts those off by y.
    """
    rows = {}
    for w in page.get_text("words"):
        y = round(w[1], 1)
        key = next((k for k in rows if abs(k - y) < 3.0), None)
        if key is None:
            rows[y] = []
            key = y
        rows[key].append((w[0], w[4]))
    return [(y, [t for _, t in sorted(v)]) for y, v in sorted(rows.items())]


NUM = re.compile(r"^-$|^[\d,]+$")


def _val(tok):
    return 0 if tok == "-" else int(tok.replace(",", ""))


def parse():
    """Table 6 -> (national, [(island, [total, *14]), ...]), with the structure asserted."""
    import fitz

    path = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing — run `python sources/ki.py --fetch`")
    doc = fitz.open(path)
    if doc.page_count == 0:
        raise SystemExit("page_count 0 — the PDF is damaged "
                         "[[reference_pdf_truncated_at_source]]")
    if not TITLE.search(doc[PAGES[0]].get_text()):
        raise SystemExit(f"page {PAGES[0] + 1} is not the head of Table 6 — the report was "
                         "re-paginated; find the title and update PAGES")

    national, islands, pending = None, [], None
    for pi in PAGES:
        page = doc[pi]
        rows = _lines(page)
        # The header's bottom row anchors the table: everything above it on this page is
        # title or stacked header, and its fragments would otherwise read as island names.
        head_y = next((y for y, toks in rows
                       if " ".join(toks).startswith("Total Total Catholic KPC")), None)
        if head_y is None:
            raise SystemExit(f"page {pi + 1} has no Table 6 header row — check PAGES")
        got = " ".join(next(toks for y, toks in rows if y == head_y))
        if got != HEADER_TOKENS:
            raise SystemExit(f"page {pi + 1} header is\n  {got!r}\nexpected\n  "
                             f"{HEADER_TOKENS!r}\nthe table was re-typeset; check CATEGORIES")
        for y, toks in rows:
            if y <= head_y:
                continue
            if (toks[0] in ("Total", "Male", "Female") and len(toks) >= 16
                    and all(NUM.match(t) for t in toks[1:16])):
                if toks[0] != "Total":
                    continue                                  # sex rows are not read
                vals = [_val(t) for t in toks[1:16]]
                if national is None and pending is None:
                    national = vals
                elif pending is not None:
                    islands.append((pending, vals))
                    pending = None
                else:
                    raise SystemExit(f"a Total row on page {pi + 1} has no island above it")
            elif len(toks) <= 3 and not any(NUM.match(t) for t in toks):
                name = " ".join(toks)
                if re.fullmatch(r"[A-Za-z][A-Za-z' ()-]*", name):
                    pending = name
    if pending is not None:
        raise SystemExit(f"island {pending!r} has no Total row")
    if national is None:
        raise SystemExit("Table 6's national row was not found")

    if len(islands) != EXPECTED_ISLANDS:
        raise SystemExit(f"{len(islands)} islands, expected {EXPECTED_ISLANDS}: "
                         f"{[i for i, _ in islands]}")
    if national[0] != NATIONAL:
        raise SystemExit(f"national total {national[0]}, expected {NATIONAL}")

    total = [0] * 15
    for _, vals in islands:
        for k in range(15):
            total[k] += vals[k]
    if total != national:
        bad = [(CATEGORIES[k - 1] if k else "Total", total[k], national[k])
               for k in range(15) if total[k] != national[k]]
        raise SystemExit(f"islands do not sum to the national row: {bad}")
    print(f"  {len(islands)} islands sum to the national row EXACTLY on all 15 columns")

    for name, vals in islands:
        if sum(vals[1:]) != vals[0]:
            raise SystemExit(f"{name}: categories sum to {sum(vals[1:])}, total says {vals[0]}")
    if sum(national[1:]) != national[0]:
        raise SystemExit("the national categories do not sum to the national total")
    print(f"  partition: EXACT at all {len(islands)} islands and nationally")
    return national, islands


def check(national):
    """UNSD table 28, which is Kiribati's own return and not a copy of this PDF."""
    try:
        import oracle
        got = oracle.oracle("Kiribati", YEAR)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  oracle check SKIPPED ({exc}) — run `python tools/oracle.py --fetch`")
        return
    if not got:
        print("  oracle check SKIPPED — no Kiribati 2015 row")
        return
    counts = got.get(oracle.TOTAL, {})
    bad = []
    for k, cat in enumerate(CATEGORIES, start=1):
        want = counts.get(ORACLE_ALIAS[cat])
        if want is None:
            bad.append(f"{cat}: not in the DYB under {ORACLE_ALIAS[cat]!r}")
        elif int(want) != national[k]:
            bad.append(f"{cat}: DYB {int(want)} vs report {national[k]}")
    if bad:
        raise SystemExit("UNSD table 28 disagrees with the report:\n   " + "\n   ".join(bad))
    print(f"  UNSD table 28: all {len(CATEGORIES)} categories agree with the report, "
          "to the person")


def normalise():
    national, islands = parse()
    check(national)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for name, vals in islands:
        geo_id = fold(name).replace(" ", "")
        for k, cat in enumerate(CATEGORIES, start=1):
            if vals[k] == 0:
                continue
            rows.append({
                "geo_id": geo_id,
                "geo_level": "island",
                "geo_name": name,
                "source_category": cat,
                "count": vals[k],
                "basis": BASIS,
                "year": YEAR,
                "source_id": SOURCE_ID,
                "note": "",
            })
    ids = {r["geo_id"] for r in rows}
    if len(ids) != EXPECTED_ISLANDS:
        raise SystemExit(f"{len(ids)} distinct geo_ids for {EXPECTED_ISLANDS} islands")

    tmp = OUT + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)

    total = sum(r["count"] for r in rows)
    print(f"wrote {OUT}")
    print(f"  {len(rows):,} rows, {len(ids)} islands, {total:,} people")
    if total != NATIONAL:
        raise SystemExit(f"the CSV holds {total:,}, the census says {NATIONAL:,}")
    print("  drawn 110,136 (100.00%): Kiribati has no refusal cell and no not-stated cell")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        normalise()
