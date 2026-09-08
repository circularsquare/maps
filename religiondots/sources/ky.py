"""Cayman Islands — ESO, 2021 Census of Population and Housing, Tables 4.9A and 4.10A-F.

Reads (or fetches) data/raw/ky/ and writes data/normalized/ky.csv.

**Seventeen named categories on 6 districts, for 68,811 people** — ~11,500 per unit, the
second-finest counting geography in the Caribbean after Saint Vincent's enumeration
districts. Table 4.9A is the national tabulation and Tables 4.10A to 4.10F are the six
districts: George Town, West Bay, Bodden Town, North Side, East End, and the Sister Islands
(Cayman Brac and Little Cayman together).

**THE UNIVERSE IS THE `CENSUS SURVEY TABULAR POPULATION COUNT`, WHICH IS NOT THE CENSUS.**
ESO publishes four different population figures for 2021 and the tables use the smallest:

    71,432   total population counted
      -327   the institutional population (prisons, dorms, retirement homes)
    71,105   non-institutional population — what ESO calls "the total population"
    -2,294   the census NON-RESPONSE ESTIMATE, derived from household refusals and verified
             no-contacts, weighted by the district tabular population distribution
    68,811   the tabular count — every table in the report, including religion

So **3.67% of the Cayman Islands is outside this map**, and it is outside for two different
reasons, only one of which is a refusal. Nothing here is scaled up to close it (§14.4);
`note_public` says so.

**THE ROW LIST IS NOT FIXED AND THE COLUMN COUNT IS NOT EITHER.** Two separate irregularities,
neither announced:

  * **North Side has no `Muslim` row at all** — not a zero, not a dash, the row is absent.
    Every other district prints eighteen rows and North Side prints seventeen.
  * **The Sister Islands table has ELEVEN figures per row where the others have twelve.** The
    columns are Total/Male/Female/DK-NS repeated for All, Caymanian and Non-Caymanian, and
    4.10F simply omits the last one.

So the parse cannot assume a fixed row list (the Bahamas' problem, §9ar) *or* a fixed column
count. It reads a row as `label tokens, then a run of figures`, takes the first figure of the
run, and asserts that the run length is constant WITHIN a table — which is what would catch a
genuinely mis-columned page while tolerating both of the above.

**`-` IS THE NIL MARKER**, as in Trinidad. Read as 0 explicitly rather than skipped, because
a skipped token shifts every later figure in the row left by one.

**THE DISTRICTS SUM TO THE NATIONAL TABLE TO WITHIN ONE PERSON.** 68,810 against a printed
68,811, and the discrepancy is measured and reported per category rather than tolerated
blindly — see `check()`. ESO gives no note about it.

Usage:
    python sources/ky.py --fetch    one 5.8 MB PDF, seconds
    python sources/ky.py            normalise from data/raw/ky/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ky")
OUT = os.path.join(ROOT, "data", "normalized", "ky.csv")

SOURCE_ID = "ky_phc_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# ESO's own copy. `statistics.caricom.org/wp-content/uploads/2025/11/The-Cayman-Islands-2021-
# Census-of-Population-and-Housing-Report-July-2022.pdf` is byte-identical and is where the
# report was found (§11v); the publisher's copy is the citable one.
PDF_URL = ("https://www.eso.ky/storage/page_docums/uploadFilePdf/685/"
           "The%20Cayman%20Islands%27%202021%20Census%20of%20Population%20and%20"
           "Housing%20Report%20-%20July%202022.pdf")
PDF_MIRROR = ("https://statistics.caricom.org/wp-content/uploads/2025/11/"
              "The-Cayman-Islands-2021-Census-of-Population-and-Housing-Report-"
              "July-2022.pdf")
PDF_NAME = "ky_census_report_2021.pdf"
PDF_PAGES = 376

# 1-based PDF pages. 4.9A is the national tabulation, used only as a check.
#
# The id is **COD-AB's `ADM1_PCODE`**, carried here by hand and asserted from the boundary
# side in `sources/ky_geo.py` — ESO publishes no code of its own, and COD's ADM1 is exactly
# these six districts under exactly these names, Sister Islands included. Same arrangement
# as Trinidad (§9ak), and for the same reason: nothing else would catch a transposition,
# because every total in this file reconciles whichever polygon a district is paired with.
NATIONAL_PAGE = 125
DISTRICTS = [
    ("KY03", "George Town",    127),
    ("KY06", "West Bay",       128),
    ("KY01", "Bodden Town",    129),
    ("KY04", "North Side",     130),
    ("KY02", "East End",       131),
    ("KY05", "Sister Islands", 132),
]

# In the order ESO prints them. `Muslim` is absent from North Side's table entirely, which is
# why membership rather than sequence is what gets validated.
CATEGORIES = [
    "Anglican",
    "Methodist",
    "Hindu",
    "Muslim",
    "Judaism",
    "Rastafarian",
    "Non-denominational",
    "None",
    "Other",
    "Baptist",
    "Church of God",
    "Jehovah Witness",
    "Pentecostal",
    "Presbyterian/United",
    "Roman Catholic",
    "Seventh-day Adventist",
    "Wesleyan Holiness",
    "DK/NS",
]
TOTAL_CAT = "Total"

TABULAR_POPULATION = 68_811       # Table 4.9A's own Total row
NON_INSTITUTIONAL = 71_105
CENSUS_TOTAL = 71_432
INSTITUTIONAL = 327
NONRESPONSE_ESTIMATE = 2_294

# **ESO'S TABLES DO NOT INTERNALLY RECONCILE, AND THE TWO REASONS ARE DIFFERENT.** Measured
# across the whole report rather than assumed:
#
#   1. A district's category rows fall 1 short of its own printed Total in George Town,
#      Bodden Town and the Sister Islands, and 1 OVER in East End. Two-sided, one person,
#      on a 68,811-person table — the signature of independently rounded figures, and
#      nothing a parse error looks like.
#   2. North Side is 5 short, and 3 of those 5 are structural: it is the one district whose
#      table OMITS a row rather than printing a dash, and the omitted row is `Muslim`. The
#      national table is short by exactly 3 on `Muslim` and on nothing else, which pins the
#      omitted cell at 3 people. It is NOT added back — see check() and taxonomy/ky2021.py.
#
# The bounds below are set from those measurements, and check() prints the whole spread and
# names the omission, because a tolerance that hides its own contents is not a check.
DISTRICT_TOL = 5          # a district's rows against its own Total
NATIONAL_TOL = 3          # the six districts against Table 4.9A, per category

FIG = re.compile(r"^(?:[\d,]+|-)$")
VALID_RUNS = (11, 12)             # 4.10F drops the Non-Caymanian DK/NS column


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 4_000_000:
        print("already have", dest)
        return

    last = None
    for url in (PDF_URL, PDF_MIRROR):
        print("GET", url)
        try:
            r = requests.get(url, timeout=1800, stream=True,
                             headers={"User-Agent": UA})
            r.raise_for_status()
        except Exception as exc:
            print(f"  failed: {exc}")
            last = exc
            continue
        with open(dest + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        os.replace(dest + ".part", dest)
        break
    else:
        raise SystemExit(f"could not fetch the report from either host: {last}")

    # §5a, and tt.py's sharpening: a complete download is not an intact file.
    with open(dest, "rb") as fh:
        head = fh.read(5)
        fh.seek(max(0, os.path.getsize(dest) - 4096))
        tail = fh.read()
    if head != b"%PDF-":
        raise SystemExit(f"{dest} is not a PDF -- starts {head!r}")
    if b"%%EOF" not in tail:
        raise SystemExit(f"{dest} has no %%EOF -- it is TRUNCATED at source")
    import fitz
    doc = fitz.open(dest)
    if doc.page_count != PDF_PAGES:
        raise SystemExit(f"{dest} has {doc.page_count} pages, expected {PDF_PAGES} -- "
                         "ESO has reissued the report; re-check DISTRICTS' page numbers")
    print(f"  {os.path.getsize(dest):,} bytes, {doc.page_count} pages")


def _num(tok, where):
    if tok == "-":
        return 0                  # the nil marker, not a missing value
    if not FIG.match(tok):
        raise SystemExit(f"{where}: {tok!r} is not a figure")
    return int(tok.replace(",", ""))


def _table(doc, page, where):
    """{category: first figure} for one 4.9A/4.10x page.

    A row is `label tokens, then a run of figures`; the first figure of the run is the
    All/Total column. The run LENGTH is asserted constant within the table, which is the
    check that survives 4.10F having one column fewer than the rest.
    """
    lines = [l.strip() for l in doc[page - 1].get_text().splitlines() if l.strip()]
    flat = " ".join(" ".join(lines).split())
    if "Religio" not in flat:
        raise SystemExit(f"{where} p{page} is not a religion table -- it starts "
                         f"{flat[:110]!r}")

    toks = []
    for l in lines:
        toks.extend(l.split())

    rows, i, label, runs = [], 0, [], set()
    while i < len(toks):
        if FIG.match(toks[i]):
            j = i
            while j < len(toks) and FIG.match(toks[j]):
                j += 1
            n = j - i
            if label and n in VALID_RUNS:
                rows.append((" ".join(label), _num(toks[i], where)))
                runs.add(n)
                label = []
                i = j
                continue
            # A short run inside the header band (page numbers, the year) — not a data row.
            for k in range(i, j):
                label.append(toks[k])
            i = j
            continue
        label.append(toks[i])
        i += 1

    if len(runs) > 1:
        raise SystemExit(f"{where} p{page}: rows have {sorted(runs)} figures -- the "
                         "columns are not constant within the table, which is a "
                         "mis-parse rather than an ESO irregularity")

    # The first data row's label carries the whole column header band plus the district
    # name; everything after it is a clean category label.
    out = {}
    for k, (lab, v) in enumerate(rows):
        name = TOTAL_CAT if k == 0 else " ".join(lab.split())
        if k == 0 and not lab.rstrip().endswith(TOTAL_CAT):
            raise SystemExit(f"{where} p{page}: the first data row is {lab!r}, which does "
                             f"not end in {TOTAL_CAT!r}")
        if k and name not in CATEGORIES:
            raise SystemExit(f"{where} p{page}: unknown category {name!r}. ESO has changed "
                             "the religion list; add it to CATEGORIES and taxonomy/"
                             "ky2021.py.")
        if name in out:
            raise SystemExit(f"{where} p{page}: {name!r} appears twice")
        out[name] = v
    return out


def read():
    import fitz

    path = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(path)
    if doc.page_count != PDF_PAGES:
        raise SystemExit(f"{path} has {doc.page_count} pages, expected {PDF_PAGES}")

    national = _table(doc, NATIONAL_PAGE, "Table 4.9A")
    districts = {}
    for code, name, page in DISTRICTS:
        cells = _table(doc, page, f"Table 4.10 {name}")
        districts[code] = {"name": name, "cells": cells}
    doc.close()
    return national, districts


def rows_from(districts):
    rows = []
    for code, name, _ in DISTRICTS:
        d = districts[code]
        for cat, v in d["cells"].items():
            note = "level=district; universe is ESO's census survey tabular population count"
            if cat == TOTAL_CAT:
                note += "; district total, not a religion category"
            rows.append({"geo_id": code, "geo_level": "district", "geo_name": d["name"],
                         "source_category": cat, "count": v, "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return rows


def check(rows, national, districts):
    ok = True

    def result(label, bad, n, extra=""):
        nonlocal ok
        good = not bad
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} ({n} checks){extra}")
        for b in bad[:6]:
            print(f"        {b}")

    n_units = len({r["geo_id"] for r in rows})
    result(f"district {n_units} units (expected {len(DISTRICTS)})",
           [] if n_units == len(DISTRICTS) else ["wrong unit count"], n_units)

    result(f"the national table's own total is {TABULAR_POPULATION:,}",
           [] if national[TOTAL_CAT] == TABULAR_POPULATION
           else [f"got {national[TOTAL_CAT]:,}"], 1)

    # ---- the categories sum to each district's own Total row ----
    bad, spread = [], []
    for code, name, _ in DISTRICTS:
        cells = districts[code]["cells"]
        got = sum(v for c, v in cells.items() if c != TOTAL_CAT)
        d = got - cells[TOTAL_CAT]
        spread.append((name, got, cells[TOTAL_CAT], d))
        if abs(d) > DISTRICT_TOL:
            bad.append(f"{name}: {got:,} vs {cells[TOTAL_CAT]:,} ({d:+,})")
    result(f"the categories sum to each district's own Total row "
           f"(tolerance {DISTRICT_TOL})", bad, len(DISTRICTS))
    for name, got, want, d in spread:
        print(f"        {name:<16} rows {got:>7,}   printed Total {want:>7,}   {d:+,}")

    # ---- the same, nationally: this one IS exact ----
    got = sum(v for c, v in national.items() if c != TOTAL_CAT)
    result("the categories sum to the national Total row EXACTLY",
           [] if got == national[TOTAL_CAT] else
           [f"{got:,} vs {national[TOTAL_CAT]:,}"], 1,
           "  <- so the discrepancies above are the DISTRICT tables', not the list's")

    # ---- the six districts sum to the national table, category by category ----
    per_cat, diffs = {}, []
    for cat in [TOTAL_CAT] + CATEGORIES:
        s = sum(districts[c]["cells"].get(cat, 0) for c, _, _ in DISTRICTS)
        per_cat[cat] = s
        d = s - national.get(cat, 0)
        if d:
            diffs.append((cat, s, national.get(cat, 0), d))
    over = [f"{c}: districts {s:,} vs national {n:,} ({d:+,})"
            for c, s, n, d in diffs if abs(d) > NATIONAL_TOL]
    result(f"the 6 districts sum to the national table (tolerance {NATIONAL_TOL})",
           over, len(CATEGORIES) + 1)

    print(f"\n  where the district sums and the national table disagree at all "
          f"({len(diffs)} of {len(CATEGORIES) + 1} categories):")
    for c, s, n, d in diffs:
        print(f"      {c:<24} districts {s:>7,}   national {n:>7,}   {d:+,}")
    if not diffs:
        print("      none — exact on every category")

    # ---- THE OMITTED ROWS, AND WHAT THE NATIONAL TABLE SAYS THEY HOLD ----
    print("\n  categories printed per district (ESO omits a row rather than printing a "
          "dash):")
    omitted = {}
    for code, name, _ in DISTRICTS:
        cells = districts[code]["cells"]
        missing = [c for c in CATEGORIES if c not in cells]
        for c in missing:
            omitted.setdefault(c, []).append(name)
        print(f"      {name:<16} {len(cells) - 1:>2} of {len(CATEGORIES)}"
              f"{'   OMITS: ' + ', '.join(missing) if missing else ''}")

    print("\n  an omitted row is NOT a zero, and the national table prices each one:")
    unexplained = []
    for cat, where in omitted.items():
        short = national.get(cat, 0) - per_cat.get(cat, 0)
        print(f"      {cat!r} omitted by {', '.join(where)} — the national table is short "
              f"{short:,} on it,\n        so the omitted cell(s) hold {short:,} "
              f"{'person' if short == 1 else 'people'}.")
        if len(where) > 1:
            unexplained.append(cat)
    if not omitted:
        print("      none — every district prints every row")
    print("      **Not added back.** §14.4: the value is implied by a residual, not "
          "published,\n      and a district's dash elsewhere in this same table means ESO "
          "prints zeros when it\n      has them. taxonomy/ky2021.py records it; the map "
          "draws North Side with no Muslims.")
    if unexplained:
        raise SystemExit(f"a category is omitted by more than one district ({unexplained}) "
                         "-- the national residual can no longer price the omission, and "
                         "this file's reasoning about it does not hold")

    print(f"\n  the universe, and what is outside it:")
    print(f"      {CENSUS_TOTAL:>8,}  total population counted")
    print(f"      {-INSTITUTIONAL:>8,}  institutional population")
    print(f"      {NON_INSTITUTIONAL:>8,}  non-institutional — ESO's \"total population\"")
    print(f"      {-NONRESPONSE_ESTIMATE:>8,}  census non-response estimate")
    print(f"      {TABULAR_POPULATION:>8,}  tabular count — every table, including religion")
    outside = CENSUS_TOTAL - TABULAR_POPULATION
    print(f"      so {outside:,} people ({100.0 * outside / CENSUS_TOTAL:.2f}%) are outside "
          "this map and nothing scales them in.")

    print(f"\n  {len(rows):,} rows. Categories, national (from the districts):")
    for cat in CATEGORIES:
        v = per_cat[cat]
        n_d = sum(1 for c, _, _ in DISTRICTS if cat in districts[c]["cells"])
        print(f"    {v:>8,}  {100.0 * v / TABULAR_POPULATION:6.2f}%  {cat:<24} on {n_d}/6")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    national, districts = read()
    rows = rows_from(districts)
    check(rows, national, districts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
