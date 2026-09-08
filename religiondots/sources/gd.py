"""Grenada — CSO, 2021 Housing and Population Census, Table 23.

Reads (or fetches) data/raw/gd/ and writes data/normalized/gd.csv.

**Twenty-six categories on 8 census units, for 108,279 people** — and the category list is
one of the best in the Caribbean: Grenada counts `SPIRITUAL BAPTIST`, `MENNONITE`,
`LUTHERAN`, `MORAVIAN`, `PRESBYTERIAN` and `INDEPENDENT BAPTISTE` as separate answers, and
it splits `ATHEIST` from `NO RELIGIOUS AFFILIATION`, which almost nothing else here does.

**THE COUNTING TIER IS 8 UNITS AND THE MAP DRAWS 7.** The census reports the **Town of
St. George** — the capital, 2,681 people — apart from the **Rest of St. George**, and
*nobody publishes a boundary for it*: not COD-AB, whose ADM1 is the six parishes plus
Carriacou and Petite Martinique, and not OpenStreetMap, which has the six parishes at
`admin_level=6` and only a `place=town` **node** for St. George's. So the two halves are
folded back into one St. George here, and `sources/gd.md` §3 records what that costs — the
town is 15.8% `NOT STATED` against 7.1% nationally, which is the sharpest thing in the
country and is exactly what the fold hides. **`gd.csv` carries BOTH tiers**: the census's
own 8 units at `geo_level=census_unit`, and the 7 drawn ones at `geo_level=parish`. Nothing
is lost from the file, only from the map, and if a town boundary ever appears the split is
already there.

**Carriacou and Petite Martinique are one unit because the census makes them one.** COD-AB
gives them separate polygons (GD01, GD08); `sources/gd_geo.py` dissolves them, because
splitting one published figure between two islands would be inventing a magnitude (§14.4).

**THE PDF's TEXT LAYER SUBSTITUTES `Ǫ` (U+01EA, O WITH OGONEK) FOR `Q`.** `MARTINIǪUE`,
`MARTINǪUE`. It is a font-encoding artefact and it appears in the column headings, so any
comparison against a hand-typed name fails on a character nobody can see. `_fold()` maps it
back; nothing else in the document is affected.

**THE UNIVERSE IS THE NON-INSTITUTIONAL POPULATION IN PRIVATE DWELLINGS, 108,279:**

    108,279   non-institutional, in private dwellings   <- Table 23, and this map
        690   institutional population
         52   homeless population
    109,021   total population, Census 2021

**99.3% of the country is inside the drawn universe**, which is the cleanest ladder of any
Caribbean source here — Barbados draws 81.4% of its own estimate and Cayman 96.3%. Nothing
is scaled (§14.4).

Usage:
    python sources/gd.py --fetch    one 618 KB PDF, seconds
    python sources/gd.py            normalise from data/raw/gd/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gd")
OUT = os.path.join(ROOT, "data", "normalized", "gd.csv")

SOURCE_ID = "gd_phc_2021_prelim"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# CSO's own host. The CARICOM mirror (sources.md §11v) serves this same report under a
# longer filename and the two files are BYTE-IDENTICAL, so the publisher's is read.
PDF_URL = ("https://stats.gov.gd/wp-content/uploads/2025/04/"
           "2021-National-Housing-Population-Census-Results-Latest-PRELIMINARY.pdf")
PDF_NAME = "gd_census_2021_preliminary.pdf"

# Two needles apiece: the caption alone also matches the table of contents, and a
# contents page carries no figures, so a caption-only search reads an empty page and
# succeeds (§12, Chile -- a read that succeeds is not a read that returned data).
T23_NEEDLES = ("POPULATION BY RELIGION AND PARISH", "ROMAN")
T17_NEEDLES = ("NON-INSTITUTIONAL POPULATION BY SEX AND PARISH", "REST OF")
T1_NEEDLES = ("POPULATION AND HOUSING CENSUS RESULT BY CATEGORY", "Homeless")

VALUE_X0 = 150.0
T1_STUB_X = 200.0                  # Table 1's stub is wider; see _stub_line()
NUM = re.compile(r"^(?:-|–|[\d,]+)$")

# The one token on the Table 23 page that sits on the caption line and appears once.
# Above it is the page furniture, which lands in whichever column it is over.
CAPTION_TOKEN = "23."

# In table order, top to bottom, spelled as CSO spells them -- `MORMOM` and `BAPTISTE`
# included. taxonomy/gd2021.py maps the strings; it does not correct them.
CATEGORIES = [
    "ANGLICAN",
    "BUDDHIST",
    "BAHAI",
    "BRETHREN",
    "CHURCH OF GOD",
    "EVANGELICAL",
    "HINDU",
    "INDEPENDENT BAPTISTE",
    "JEHOVAH WITNESSES",
    "METHODIST",
    "MENNONITE",
    "MORAVIAN",
    "MORMOM",                       # CSO's spelling of Mormon. Not corrected here.
    "MUSLIM",
    "PENTECOSTAL",
    "PRESBYTERIAN",
    "RASTAFARIAN",
    "ROMAN CATHOLIC",
    "SALVATION ARMY",
    "SEVENTH DAY ADVENTIST",
    "SPIRITUAL BAPTIST",
    "LUTHERAN",
    "ATHEIST",
    "NO RELIGIOUS AFFILIATION",
    "OTHER (SPECIFY)",
    "NOT STATED",
]
TOTAL_CAT = "TOTAL"

# The census's own 8 units, left to right as Table 23 prints them, each with the parish
# it is folded into and that parish's COD-AB pcode. Asserted from the boundary side in
# sources/gd_geo.py -- CSO publishes no code of its own.
CENSUS_UNITS = [
    ("REST OF ST.GEORGE",                "St. George",                       "GD04"),
    ("TOWN OF ST.GEORGE",                "St. George",                       "GD04"),
    ("ST.JOHN",                          "St. John",                         "GD05"),
    ("ST.MARK",                          "St. Mark",                         "GD06"),
    ("ST.PATRICK",                       "St. Patrick",                      "GD07"),
    ("ST.ANDREW",                        "St. Andrew",                       "GD02"),
    ("ST.DAVID",                         "St. David",                        "GD03"),
    ("CARRIACOU AND PETITE MARTINIQUE",  "Carriacou and Petite Martinique",  "GD01"),
]
# The heading Table 23 prints over each column, which is NOT the label Table 17 uses for
# the same unit: Table 23 says `ST.GEORGE` where Table 17 says `REST OF ST.GEORGE`, and
# it abbreviates the dependency. Held separately so the header check compares like with
# like rather than being loosened until it passes.
COLUMN_HEADINGS = [
    "ST.GEORGE", "TOWN OF ST.GEORGE", "ST. JOHN", "ST. MARK", "ST.PATRICK",
    "ST.ANDREW", "ST.DAVID", "CARRIACOU& PETITE MARTINIQUE",
]
PARISHES = [
    ("St. George",                      "GD04"),
    ("St. John",                        "GD05"),
    ("St. Mark",                        "GD06"),
    ("St. Patrick",                     "GD07"),
    ("St. Andrew",                      "GD02"),
    ("St. David",                       "GD03"),
    ("Carriacou and Petite Martinique", "GD01"),
]
FOLDED = ("REST OF ST.GEORGE", "TOWN OF ST.GEORGE")

DRAWN_UNIVERSE = 108_279           # Table 1, non-institutional in private dwellings
INSTITUTIONAL = 690
HOMELESS = 52
TOTAL_POPULATION = 109_021

# Table 1's own rows, asserted rather than transcribed.
T1_LADDER = {
    "Non-Institutional Population in Private Dwelling": DRAWN_UNIVERSE,
    "Institutional Population": INSTITUTIONAL,
    "Homeless Population": HOMELESS,
    "Total Population": TOTAL_POPULATION,
}

# U+01EA / U+01EB, which this PDF's text layer emits where the page shows a Q.
OGONEK = {"Ǫ": "Q", "ǫ": "q"}


def _fold(s):
    for bad, good in OGONEK.items():
        s = s.replace(bad, good)
    return "".join(str(s).split())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 400_000:
        print("already have", dest)
        return
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, timeout=900, stream=True, headers={"User-Agent": UA})
    r.raise_for_status()
    with open(dest + ".part", "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    os.replace(dest + ".part", dest)
    size = os.path.getsize(dest)
    # §5a: a 200 is not a download, and a PDF can arrive truncated with a
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


def _open():
    import fitz

    path = os.path.join(RAW, PDF_NAME)
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

    A row is a list of (x0, text) left to right. Anything not `ncols` wide -- the page
    number, a caption's year -- is rejected, and check() proves nothing real was
    rejected by summing the columns.
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


def _stub_line(page, mid, x_ceiling, tol=8.0):
    """The stub cell sitting on one row of figures, in reading order.

    `x_ceiling` is per table: Table 23's figures start at x=181 and Table 1's at 229,
    so one number would either cut a stub in half or swallow a figure. Table 1's stub
    reaches x=157 — the `in` of *Population in Private Dwelling* — which is inside
    Table 23's value column and outside its own.
    """
    ws = [(round((y0 + y1) / 2, 1), x0, w)
          for x0, y0, x1, y1, w, *_ in page.get_text("words")
          if x0 < x_ceiling and abs((y0 + y1) / 2 - mid) <= tol]
    return " ".join(w for _, _, w in sorted(ws))


def _caption_y(page, token):
    ys = [(y0 + y1) / 2 for x0, y0, x1, y1, w, *_ in page.get_text("words")
          if w == token]
    if len(ys) != 1:
        raise SystemExit(f"the caption token {token!r} appears {len(ys)} times on the "
                         "page; the header band cannot be located")
    return ys[0]


def _header(page, col_x, y_floor, y_ceiling):
    """Column headings, each word filed under the column its x0 is nearest to.

    Grenada's headings wrap over up to five lines and break mid-word — `ST.PATRIC` /
    `K`, `CARRIACO` / `U&` / `PETITE` / `MARTINIǪU` / `E` — so they are reassembled per
    column in reading order rather than per line, and compared with the spaces squashed
    out and the ogonek folded back to a Q.

    The group heading `PARISH` is printed in the stub column and so never reaches here.
    """
    cols = {x: [] for x in col_x}
    for x0, y0, x1, y1, w, *_ in page.get_text("words"):
        mid = (y0 + y1) / 2
        if not (y_floor < mid < y_ceiling) or x0 < VALUE_X0 - 20:
            continue
        cols[min(col_x, key=lambda c: abs(c - x0))].append((mid, x0, w))
    return [" ".join(w for _, _, w in sorted(cols[x])) for x in col_x]


def read():
    """Table 23 -> {category: {census unit: count}}, plus the `TOTAL` column."""
    doc = _open()
    page = _page(doc, T23_NEEDLES)
    rows, rejected = _grid(page, len(CENSUS_UNITS) + 1)

    want = len(CATEGORIES) + 1              # every category, then the `TOTAL` row
    if len(rows) != want:
        raise SystemExit(f"Table 23 gave {len(rows)} full rows of "
                         f"{len(CENSUS_UNITS) + 1}, expected {want}. Rejected: "
                         f"{[[w for _, w in ts] for _, ts in rejected]}")

    # ---- the stub column must be the expected labels, in order ----
    got = "".join(_fold(w) for w in _stub(page, rows[0][0] - 12))
    exp = "".join(_fold(c) for c in CATEGORIES + [TOTAL_CAT])
    if got != exp:
        raise SystemExit(f"Table 23's stub column is not the expected list.\n"
                         f"  got  {got}\n  want {exp}")

    # ---- and the column headings must be the expected units, in order ----
    header = _header(page, [x for x, _ in rows[0][1]],
                     _caption_y(page, CAPTION_TOKEN), rows[0][0] - 12)
    exp_hdr = COLUMN_HEADINGS + [TOTAL_CAT]
    if [_fold(h) for h in header] != [_fold(h) for h in exp_hdr]:
        raise SystemExit(f"Table 23's column headings are not the expected units.\n"
                         f"  got  {header}\n  want {exp_hdr}")

    out = {}
    for (mid, ts), name in zip(rows, CATEGORIES + [TOTAL_CAT]):
        cells = [_cell(w) for _, w in ts]
        out[name] = {u: v for (u, _, _), v in zip(CENSUS_UNITS, cells[:-1])}
        out[name][TOTAL_CAT] = cells[-1]
    doc.close()
    return out


def read_table17():
    """Table 17 — the 2021 population of each census unit, in table order.

    An INDEPENDENT table: Table 23's column totals must reproduce it exactly, and since
    all eight figures differ this pins which column is which unit rather than merely
    showing that the arithmetic closes. Each row is MALE, FEMALE, TOTAL for 2021 and
    then the same three for 2011, so the third figure is the one wanted.
    """
    doc = _open()
    page = _page(doc, T17_NEEDLES)
    rows, _ = _grid(page, 6)
    doc.close()
    want = len(CENSUS_UNITS) + 1
    if len(rows) != want:
        raise SystemExit(f"Table 17 gave {len(rows)} rows of 6, expected {want}")
    return [_cell(ts[2][1]) for _, ts in rows]


def read_table1():
    """Table 1 — the universe ladder, looked up by stub label."""
    doc = _open()
    page = _page(doc, T1_NEEDLES)
    rows, _ = _grid(page, 3)
    found = {}
    for mid, ts in rows:
        label = _stub_line(page, mid, T1_STUB_X)
        if label:
            found.setdefault(label, []).append(_cell(ts[2][1]))
    doc.close()

    out = {}
    for label in T1_LADDER:
        got = found.get(label)
        if not got or len(got) != 1:
            raise SystemExit(f"Table 1 has {label!r} {len(got or [])} times, expected "
                             f"once. Rows read: {sorted(found)}")
        out[label] = got[0]
    return out


def rows_from(table):
    rows = []
    base = ("universe is the non-institutional population in private dwellings, "
            "108,279 of a total 109,021")

    # ---- the census's own tier, kept whole ----
    for unit, parish, pcode in CENSUS_UNITS:
        for cat in CATEGORIES + [TOTAL_CAT]:
            note = f"level=census_unit; {base}; folded into {parish} for drawing"
            if cat == TOTAL_CAT:
                note += "; unit total, not a religion category"
            rows.append({"geo_id": unit, "geo_level": "census_unit", "geo_name": unit,
                         "source_category": cat, "count": table[cat][unit],
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": note})

    # ---- and the drawn tier: St. George's two halves added back together ----
    for parish, pcode in PARISHES:
        members = [u for u, p, _ in CENSUS_UNITS if p == parish]
        for cat in CATEGORIES + [TOTAL_CAT]:
            note = f"level=parish; {base}"
            if cat == TOTAL_CAT:
                note += "; parish total, not a religion category"
            if len(members) > 1:
                note += ("; the census's Town of St. George is folded in here — no "
                         "boundary set publishes the town (sources/gd.md §3)")
            rows.append({"geo_id": pcode, "geo_level": "parish", "geo_name": parish,
                         "source_category": cat,
                         "count": sum(table[cat][u] for u in members),
                         "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                         "note": note})
    return rows


def check(rows, table, t17, t1):
    ok = True

    def result(label, bad, n, extra=""):
        nonlocal ok
        good = not bad
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} ({n} checks){extra}")
        for b in bad[:8]:
            print(f"        {b}")

    drawn = {r["geo_id"] for r in rows if r["geo_level"] == "parish"}
    result(f"parish {len(drawn)} drawn units (expected {len(PARISHES)})",
           [] if len(drawn) == len(PARISHES) else ["wrong unit count"], len(drawn))

    result(f"the national TOTAL is {DRAWN_UNIVERSE:,}",
           [] if table[TOTAL_CAT][TOTAL_CAT] == DRAWN_UNIVERSE
           else [f"got {table[TOTAL_CAT][TOTAL_CAT]:,}"], 1)

    # ---- the 8 units sum to the TOTAL column, on every row. EXACT here.
    bad = []
    for cat in CATEGORIES + [TOTAL_CAT]:
        got = sum(table[cat][u] for u, _, _ in CENSUS_UNITS)
        if got != table[cat][TOTAL_CAT]:
            bad.append(f"{cat}: {got:,} vs {table[cat][TOTAL_CAT]:,}")
    result("the 8 units sum to the row's own TOTAL, exactly", bad, len(CATEGORIES) + 1)

    # ---- the categories sum to each unit's own TOTAL. EXACT here too.
    bad = []
    for unit in [u for u, _, _ in CENSUS_UNITS] + [TOTAL_CAT]:
        got = sum(table[c][unit] for c in CATEGORIES)
        if got != table[TOTAL_CAT][unit]:
            bad.append(f"{unit}: {got:,} vs {table[TOTAL_CAT][unit]:,}")
    result("the categories sum to each unit's own TOTAL, exactly", bad,
           len(CENSUS_UNITS) + 1,
           "\n        <- Grenada's table reconciles to the person, which Saint Lucia's "
           "(§9aw)\n           and Cayman's (§9at) do not")

    # ---- an INDEPENDENT table, elsewhere in the report, in the same order ----
    got = [table[TOTAL_CAT][u] for u, _, _ in CENSUS_UNITS] + [table[TOTAL_CAT][TOTAL_CAT]]
    bad = [f"position {i}: Table 23 {a:,} vs Table 17 {b:,}"
           for i, (a, b) in enumerate(zip(got, t17)) if a != b]
    result("Table 17 reproduces Table 23's unit totals, in order", bad, len(got),
           "  <- all eight figures differ, so this pins the columns")

    bad = [f"{k}: read {t1[k]:,}, this file says {v:,}"
           for k, v in T1_LADDER.items() if t1[k] != v]
    result("Table 1's universe ladder is what this file says it is", bad, len(T1_LADDER))

    # ---- the fold, and what it costs ----
    rest, town = FOLDED
    print(f"\n  THE FOLD — the census's tier is 8 units and the map draws 7:")
    for u in FOLDED:
        t = table[TOTAL_CAT][u]
        ns = table["NOT STATED"][u]
        print(f"      {u:<22} {t:>7,} people   NOT STATED {ns:>6,}  "
              f"{100.0 * ns / t:5.1f}%")
    nat_ns = table["NOT STATED"][TOTAL_CAT]
    print(f"      {'Grenada':<22} {DRAWN_UNIVERSE:>7,} people   NOT STATED "
          f"{nat_ns:>6,}  {100.0 * nat_ns / DRAWN_UNIVERSE:5.1f}%")
    print(f"      folded to {table[TOTAL_CAT][rest] + table[TOTAL_CAT][town]:,} in "
          "St. George. No boundary set anywhere publishes the town\n      (sources/gd.md "
          "§3); gd.csv keeps both tiers so the split survives the map.")

    print(f"\n  the universe, and what is outside it (Table 1):")
    print(f"      {DRAWN_UNIVERSE:>8,}  non-institutional, in private dwellings — "
          "THIS MAP")
    print(f"      {INSTITUTIONAL:>8,}  institutional population")
    print(f"      {HOMELESS:>8,}  homeless population")
    print(f"      {TOTAL_POPULATION:>8,}  total population, Census 2021   "
          f"({100.0 * DRAWN_UNIVERSE / TOTAL_POPULATION:.1f}% drawn)")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES:
        v = table[cat][TOTAL_CAT]
        print(f"    {v:>8,}  {100.0 * v / DRAWN_UNIVERSE:6.2f}%  {cat}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    table = read()
    t17 = read_table17()
    t1 = read_table1()
    rows = rows_from(table)
    check(rows, table, t17, t1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
