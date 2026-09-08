"""Malawi — NSO, 2018 Population and Housing Census, Table E5, religion by district.

Reads (or fetches) data/raw/mw/ and writes data/normalized/mw.csv.

**Ten denominations on 32 districts for 17,563,749 people** — about 549,000 per unit, which
is finer per head than Kenya's 47 counties (1.0M) and than Ghana at the region tier, and it
is the whole of what NSO publishes. The categories separate the **Church of Central Africa
Presbyterian** and the **Anglicans** from the Catholics and from each other, which no other
African source on this map does: Ghana's four Christian cells are Catholic / Protestant /
Pentecostal / Other, and Kenya's five are mission / evangelical / AIC / Orthodox / other.
Malawi is the first to name a single Presbyterian body.

**THE TABLE IS PRINTED SIDEWAYS AND THAT IS THE WHOLE PARSE.** Table E5 is rotated 90° on
the page: every text line has `dir == (0, -1)`, each printed *column* is one area, and the
eleven figures run down it in header order. Read the page the normal way — group words into
horizontal lines — and each "line" you get back is one DENOMINATION across many districts,
in an order that changes from page to page as the areas do. That reading parses without
error and transposes the country. This module groups by the line's **x** instead, so a
column is an area, and asserts that every data column holds exactly twelve entries: a name
and eleven numbers.

**THREE PANELS, NOT ONE.** Pages 134-138 hold Both sexes / Males / Females one after the
other, 36 area columns each, 108 in total. The panels are identified by their first column's
name (`Malawi`, `Males`, `Females`) rather than by position, and only the first is drawn.
The other two are read anyway because `Males + Females == Both sexes` on every cell is a
free check on the transposition, and it is the check that would catch a column landing in
the wrong panel.

**THE 32 UNITS ARE DISJOINT AND THE FOUR CITIES ARE NOT INSIDE THEIR DISTRICTS.** Mzuzu,
Lilongwe, Zomba and Blantyre Cities are printed as peers of Mzimba, Lilongwe, Zomba and
Blantyre, not as parts of them, and the region rows prove it: the seven Northern units sum
to the Northern row exactly. §12's shape-3 trap (a tier hiding inside the drawn one) is
therefore absent here, and `check()` asserts the sum rather than assuming it.

**`-` IS AN IN-BAND ZERO**, in one cell — Likoma's `Traditional` — and it is Sri Lanka's
trap again (§9j). A numeric regex drops it, the column then holds eleven entries instead of
twelve, and every *remaining* figure shifts up one row: Likoma would be recorded as having
156 traditionalists and 32 people of other denominations and no No-Religion cell at all,
with no total anywhere disagreeing. The parser maps `-` to 0 and raises on any other
non-numeric token.

**THE NATIONAL TABLE IS ONE CATEGORY FINER THAN THE DISTRICT ONE**, which is §3.9's trade
made across two tables of one report rather than inside one. Table 3.4 (page 19) splits
Table E5's `Other Denomination` (992,304) into **Buddhism 5,506 + Hinduism 3,211 + Other
non-Christian 983,587**, summing exactly. Those 8,717 Buddhists and Hindus have no
geography, so they cannot be drawn apart; the split is read anyway and asserted against the
district table, because it is the only independent check on that column and because
`sources/mw.md` §3 needs the number.

Usage:
    python sources/mw.py --fetch    one 8.1 MB PDF, seconds
    python sources/mw.py            normalise from data/raw/mw/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mw")
OUT = os.path.join(ROOT, "data", "normalized", "mw.csv")

SOURCE_ID = "mw_phc_2018"
YEAR = 2018
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# NSO's own site is a Nuxt app whose listing API (cms.nsomalawi.mw/api/census) is 401.
# The DOWNLOAD route is open and needs no key; the trailing filename is decorative and the
# real one comes back in Content-Disposition. sources/mw.md 1 has the whole finding.
PDF_URL = ("https://cms.nsomalawi.mw/api/download/270/"
           "2018-Malawi-Population-and-Housing-Census-Main-Report.pdf")
PDF_NAME = "mw_phc2018_main_report.pdf"

# 0-based page indices of Table E5's five pages, and of the national Table 3.4. Verified by
# title match at read time rather than trusted.
PAGES = [144, 145, 146, 147, 148]
NATIONAL_PAGE = 29
TITLE_RE = re.compile(r"Table\s*E5\.\s*Population of Malawi by Denomination,\s*Region,"
                      r"\s*and District", re.I)
NATIONAL_TITLE_RE = re.compile(r"Table\s*3\.4:\s*Population distribution by religious "
                               r"denomination", re.I)

# Down each printed column, in the order the header prints them. `Total` first because it is
# the area's own universe and not a religion.
CATEGORIES = [
    "Total",
    "Catholic",
    "CCAP",
    "Seventh Day Adventist/Baptist/Apostolic",
    "Anglican",
    "Pentecostal",
    "Other Christian Denominations",
    "Islam",
    "Traditional",
    "Other Denomination",
    "No Religion",
]
TOTAL_CAT = "Total"

# The three panels, named by the first column of each. NSO writes the country's name in the
# first panel and the sex in the other two.
PANELS = ["Malawi", "Males", "Females"]
DRAWN_PANEL = "Malawi"

REGIONS = ["Northern", "Central", "Southern"]

# Region -> its districts, in the order NSO prints them, which is also ADM2 p-code order.
# Asserted against the printed names rather than used to find them.
DISTRICTS = {
    "Northern": ["Chitipa", "Karonga", "Nkhata Bay", "Rumphi", "Mzimba", "Likoma",
                 "Mzuzu City"],
    "Central": ["Kasungu", "Nkhotakota", "Ntchisi", "Dowa", "Salima", "Lilongwe",
                "Mchinji", "Dedza", "Ntcheu", "Lilongwe City"],
    "Southern": ["Mangochi", "Machinga", "Zomba", "Chiradzulu", "Blantyre", "Mwanza",
                 "Thyolo", "Mulanje", "Phalombe", "Chikwawa", "Nsanje", "Balaka", "Neno",
                 "Zomba City", "Blantyre City"],
}
EXPECTED_DISTRICTS = 32
AREAS_PER_PANEL = 1 + len(REGIONS) + EXPECTED_DISTRICTS      # 36

NATIONAL = 17_563_749          # Table E5's own Malawi row, and Table 3.4's Total

# Table 3.4 splits Table E5's `Other Denomination` three ways, nationally only.
OTHER_SPLIT = {"Buddhism": 5_506, "Hinduism": 3_211,
               "Other non-Christian Denomination": 983_587}

NUM = re.compile(r"^[\d,]+$")
DASH = {"-", "‐", "‑", "‒", "–", "—", "−"}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 5_000_000:
        print("already have", dest)
        return
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, timeout=900, verify=False, stream=True,
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
    """One printed figure -> int. `-` means zero and must not be dropped."""
    if tok in DASH:
        return 0
    if not NUM.match(tok):
        raise SystemExit(f"{where}: {tok!r} is neither a number nor a dash -- Table E5 "
                         "has been re-typeset and the parse would silently shift")
    return int(tok.replace(",", ""))


def _columns(page, pageno):
    """Every rotated text column on the page, as (x, [entries]) sorted left to right.

    The table is printed sideways, so one printed COLUMN is one area and its entries run
    down the page. Lines are grouped by x; entries within a column are ordered by
    descending y, which is the order the header prints the denominations in.
    """
    cols = {}
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            # The page footer is horizontal; the table is not. Anything not rotated is
            # furniture and never data.
            if tuple(round(v) for v in line["dir"]) != (0, -1):
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            cols.setdefault(round(line["bbox"][0], 1), []).append(
                (line["bbox"][3], text))

    out = []
    for x in sorted(cols):
        entries = [t for _, t in sorted(cols[x], key=lambda e: -e[0])]
        if len(entries) != len(CATEGORIES) + 1:
            continue          # the title, the stub header and its wrapped label fragments
        out.append((x, entries))
    if not out:
        raise SystemExit(f"page {pageno}: no data column with "
                         f"{len(CATEGORIES) + 1} entries -- the table has been re-laid out")
    return out


def read():
    import fitz

    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)

    columns = []
    for pageno in PAGES:
        if pageno >= doc.page_count:
            raise SystemExit(f"{p} has {doc.page_count} pages, expected page {pageno}")
        page = doc[pageno]
        text = " ".join(page.get_text().split())
        if not TITLE_RE.search(text):
            raise SystemExit(f"page {pageno} is not Table E5 -- it starts {text[:120]!r}. "
                             "NSO has re-paginated; find the table and update PAGES.")
        for _, entries in _columns(page, pageno):
            name = entries[0]
            vals = [_cell(t, f"page {pageno}, column {name!r}") for t in entries[1:]]
            columns.append((name, vals))

    want = len(PANELS) * AREAS_PER_PANEL
    if len(columns) != want:
        raise SystemExit(f"{len(columns)} area columns across {len(PAGES)} pages, expected "
                         f"{want} ({len(PANELS)} panels x {AREAS_PER_PANEL} areas)")

    panels = {}
    for i, label in enumerate(PANELS):
        block = columns[i * AREAS_PER_PANEL:(i + 1) * AREAS_PER_PANEL]
        if block[0][0] != label:
            raise SystemExit(f"panel {i} starts with {block[0][0]!r}, expected {label!r} "
                             "-- the panels are not where they were")
        panels[label] = block

    # The area sequence inside a panel, asserted rather than assumed. A transposed or
    # inserted row would leave every total intact (§12 shape 2) and nothing else sees it.
    areas = []
    for region in REGIONS:
        areas.append(region)
        areas.extend(DISTRICTS[region])
    for label, block in panels.items():
        # The first column of a panel is the panel's own name -- `Malawi` for the both-sexes
        # panel, `Males` and `Females` for the other two -- and the 35 after it are the same
        # areas in the same order.
        got = [n for n, _ in block[1:]]
        if got != areas:
            first = next((a, b) for a, b in zip(got, areas) if a != b)
            raise SystemExit(f"panel {label!r} area order differs from Table E5's: "
                             f"first mismatch {first[0]!r} vs {first[1]!r}")

    out = []
    codes = {}
    for region_i, region in enumerate(REGIONS, start=1):
        for district_i, name in enumerate(DISTRICTS[region], start=1):
            codes[name] = f"MW{region_i}{district_i:02d}"

    for i, (name, vals) in enumerate(panels[DRAWN_PANEL]):
        if i == 0:
            level, code = "country", "MW"
        elif name in REGIONS:
            level, code = "region", f"MW{REGIONS.index(name) + 1}"
        else:
            level, code = "district", codes[name]
        for cat, n in zip(CATEGORIES, vals):
            note = f"level={level}"
            if cat == TOTAL_CAT:
                note += "; universe total, not a religion category"
            out.append({"geo_id": code, "geo_level": level,
                        "geo_name": DRAWN_PANEL if i == 0 else name,
                        "source_category": cat, "count": n, "basis": BASIS,
                        "year": YEAR, "source_id": SOURCE_ID, "note": note})

    return out, panels, _read_national(doc)


def _read_national(doc):
    """Table 3.4's three-way split of `Other Denomination`, national only."""
    page = doc[NATIONAL_PAGE]
    text = page.get_text()
    if not NATIONAL_TITLE_RE.search(" ".join(text.split())):
        raise SystemExit(f"page {NATIONAL_PAGE} is not Table 3.4 -- NSO has re-paginated")
    lines = [ln.strip() for ln in text.splitlines()]
    found = {}
    for label in OTHER_SPLIT:
        try:
            i = lines.index(label)
        except ValueError:
            raise SystemExit(f"Table 3.4 has no {label!r} row")
        nxt = next((ln for ln in lines[i + 1:] if ln), "")
        found[label] = _cell(nxt.replace(" ", ""), f"Table 3.4 {label!r}")
    return found


def check(rows, panels, national):
    ok = True

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
    for lv, want in (("district", EXPECTED_DISTRICTS), ("region", len(REGIONS)),
                     ("country", 1)):
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<9} {got:>4} units (expected {want})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(TOTAL_CAT) == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national universe {nat.get(TOTAL_CAT):,} "
          f"(expected {NATIONAL:,})")
    print("      the whole census count -- NSO publishes no `not stated` cell for religion, "
          "so\n      there is no gap between the table's universe and the country.")

    # NSO neither suppresses nor rounds this table, so every identity is an equality.
    by_unit = {}
    for r in rows:
        by_unit.setdefault((r["geo_level"], r["geo_id"]), {})[r["source_category"]] = \
            r["count"]
    bad = [k for k, d in by_unit.items()
           if sum(v for c, v in d.items() if c != TOTAL_CAT) != d[TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 10 denominations sum to Total on all "
          f"{len(by_unit)} rows ({len(bad)} failures)")
    for k in bad[:5]:
        print(f"        {k}")

    # §12 shape 3, both tiers. The four cities are peers of their districts, not parts of
    # them, and this is what proves it: if Mzuzu City were inside Mzimba the Northern row
    # would come up 221,272 short on Total and short on every column.
    bad = []
    for region_i, region in enumerate(REGIONS, start=1):
        want = by_unit[("region", f"MW{region_i}")]
        for cat in CATEGORIES:
            s = sum(by_unit[("district", k)][cat] for lv, k in by_unit
                    if lv == "district" and k.startswith(f"MW{region_i}"))
            if s != want[cat]:
                bad.append((region, cat, s, want[cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the districts sum to their region on all "
          f"{len(CATEGORIES)} columns -- so the four cities are DISJOINT from their "
          "districts")
    for region, cat, s, w in bad[:5]:
        print(f"        {region}/{cat}: {s:,} vs {w:,}")

    bad = []
    for cat in CATEGORIES:
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "region" and r["source_category"] == cat)
        if s != nat[cat]:
            bad.append((cat, s, nat[cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 3 regions sum to the Malawi row on all "
          f"{len(CATEGORIES)} columns")
    for c, s, n in bad:
        print(f"        {c}: {s:,} vs {n:,}")

    # The panels are the check on the transposition: a column read into the wrong panel, or
    # a page's columns ordered wrongly, breaks this and nothing else would.
    bad = []
    for i in range(AREAS_PER_PANEL):
        both = panels[DRAWN_PANEL][i][1]
        male = panels["Males"][i][1]
        female = panels["Females"][i][1]
        for j, cat in enumerate(CATEGORIES):
            if male[j] + female[j] != both[j]:
                bad.append((panels[DRAWN_PANEL][i][0], cat,
                            male[j] + female[j], both[j]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} Males + Females == Both sexes on all "
          f"{AREAS_PER_PANEL * len(CATEGORIES)} cells ({len(bad)} failures)")
    for a, c, s, w in bad[:5]:
        print(f"        {a}/{c}: {s:,} vs {w:,}")

    # Table 3.4 against Table E5 -- the only independent reading of `Other Denomination`.
    good = national == OTHER_SPLIT
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} Table 3.4's split reads as expected: {national}")
    s = sum(national.values())
    good = s == nat["Other Denomination"]
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} it sums to Table E5's `Other Denomination` "
          f"{s:,} vs {nat['Other Denomination']:,}")
    print(f"      so {national['Buddhism'] + national['Hinduism']:,} Buddhists and Hindus "
          "are inside a column that\n      has no geography for them -- sources/mw.md §3.")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES:
        n = nat[cat]
        mark = "  <- universe" if cat == TOTAL_CAT else ""
        print(f"    {n:>11,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, panels, national = read()
    check(rows, panels, national)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
