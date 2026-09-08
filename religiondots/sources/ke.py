"""Kenya — KNBS, 2019 KPHC Volume IV Table 2.30, religion by county.

Reads (or fetches) data/raw/ke/ and writes data/normalized/ke.csv.

**THE DEEPEST RELIGION QUESTION IN AFRICA, ON THE COARSEST GEOGRAPHY THIS PROJECT DRAWS.**
Thirteen categories — and two of them, `Evangelical Churches` and `African Instituted
Churches`, are counted by no other census anywhere on this map — for 47.2M people across
**47 counties**. That is ~1.0M per unit, coarser even than the Philippines' 929,000 (§9m),
and it is the whole cost of the country.

**IT IS THE OFFICE'S CEILING, NOT A CHOICE MADE HERE, AND THE VOLUME SAYS SO ON ITS OWN
FACE.** Volume IV runs to 498 pages and 40-odd tables, and *every other table in it* is
titled "…by County and Sub-County" — activity status, disability, albinism, crops,
livestock, mobile phones, births. Religion is Table 2.30, "…by Religious Affiliation and
County", it occupies exactly one page, and there is no sub-county twin anywhere in the
volume. Religion is the one variable KNBS stopped at county for. The census itself collects
down to the enumeration area (the Volume IV questionnaire annex lists County / Sub-County /
Division / Location / Sub-Location / E.A.), so the data exists and is not published.

The upgrade path is IPUMS, whose Kenya 2019 sample is 10% and identifies **division**,
below county — see sources.md §10a. It needs the account, which is not yet approved.

**THE UNIVERSE IS THE CONVENTIONAL HOUSEHOLD POPULATION**, and Table 2.30 carries the
footnote saying so: "The question was not asked to those who are in hotels/lodges, Hospital,
Prison/Police Cell, Children's Home, Travellers and Outdoor Sleepers." 47,213,282 against a
census 47,564,296 — 351,014 people, 0.74%, who were never asked. Not scaled up (§14.4).

Usage:
    python sources/ke.py --fetch    one 4.9 MB PDF, seconds
    python sources/ke.py            normalise from data/raw/ke/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ke")
OUT = os.path.join(ROOT, "data", "normalized", "ke.csv")

SOURCE_ID = "ke_kphc_2019_v4"
YEAR = 2019
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

PDF_URL = ("https://www.knbs.or.ke/wp-content/uploads/2023/09/"
           "2019-Kenya-population-and-Housing-Census-Volume-4-Distribution-of-Population-"
           "by-Socio-Economic-Characteristics.pdf")
PDF_NAME = "kphc2019_volume4.pdf"

# 0-based page index of Table 2.30 in that PDF. Verified by title match at read time rather
# than trusted, so a re-issue with different pagination fails loudly instead of parsing the
# ethnicity table on the next page as though it were religion.
PAGE = 434
TITLE_RE = re.compile(r"Table\s*2\.30:\s*Distribution of Population by\s+Religious "
                      r"Affiliation and County", re.I)

# Column order across the page, left to right, exactly as printed. `Total` first because it
# is the row's own universe and not a category.
CATEGORIES = [
    "Total",
    "Catholic",
    "Protestant",
    "Evangelical Churches",
    "African Instituted Churches",
    "Orthodox",
    "Other Christian",
    "Islam",
    "Hindu",
    "Traditionists",
    "Other Religion",
    "No religion /Atheists",
    "Don't Know",
    "Not Stated",
]
TOTAL_CAT = "Total"

NATIONAL = 47_213_282          # Table 2.30's own KENYA row
CENSUS_POPULATION = 47_564_296  # 2019 KPHC total, all persons

EXPECTED_COUNTIES = 47
NATIONAL_ROW = "KENYA"

NUM = re.compile(r"^[\d,]+$")


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
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


def _table_rows(page):
    """(name, [14 ints]) per printed row, in page order.

    Parsed by ORDER, not by x-position. Every row prints its 14 figures left to right in
    the header's order and never omits one — there is no suppression in this table and no
    blank cells — so the ordinal reading is exact, and `check()` proves it twice over: the
    13 categories must sum to each row's own Total, and the 47 counties must sum to the
    KENYA row, on all fourteen columns. A column-boundary reading would be more fragile
    (the figures are right-aligned, so a wide number crosses into its neighbour's band) and
    no better checked.
    """
    lines = {}
    for w in page.get_text("words"):
        lines.setdefault(round(w[1]), []).append(w)

    rows = []
    for y in sorted(lines):
        ws = sorted(lines[y], key=lambda w: w[0])
        toks = [w[4] for w in ws]
        nums = [t for t in toks if NUM.match(t)]
        if len(nums) < 10:
            continue                      # title, header, footnote, page number
        name = " ".join(t for t in toks if not NUM.match(t)).strip()
        if len(nums) != len(CATEGORIES):
            raise SystemExit(f"row {name!r} has {len(nums)} figures, expected "
                             f"{len(CATEGORIES)} -- the table has been re-laid out")
        rows.append((name, [int(n.replace(",", "")) for n in nums]))
    return rows


def read():
    import fitz

    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)
    if PAGE >= doc.page_count:
        raise SystemExit(f"{p} has {doc.page_count} pages, expected page {PAGE}")
    page = doc[PAGE]
    text = " ".join(page.get_text().split())
    if not TITLE_RE.search(text):
        raise SystemExit(f"page {PAGE} is not Table 2.30 -- it starts {text[:120]!r}. "
                         "KNBS has re-paginated; find the table and update PAGE.")

    rows = _table_rows(page)
    nat = [r for r in rows if r[0].upper() == NATIONAL_ROW]
    counties = [r for r in rows if r[0].upper() != NATIONAL_ROW]
    if len(nat) != 1:
        raise SystemExit(f"{len(nat)} rows named {NATIONAL_ROW}, expected 1")
    if len(counties) != EXPECTED_COUNTIES:
        raise SystemExit(f"{len(counties)} county rows, expected {EXPECTED_COUNTIES}")

    out = []
    for level, group in (("country", nat), ("county", counties)):
        for i, (name, vals) in enumerate(group, start=1):
            # KNBS prints the counties in county-code order, 001 Mombasa .. 047 Nairobi.
            # The code is NOT in the table; it is the row's position, which is why
            # sources/ke_geo.py verifies it against the boundary file's own codes by NAME
            # rather than trusting it.
            code = "0" if level == "country" else f"{i:03d}"
            for cat, n in zip(CATEGORIES, vals):
                note = f"level={level}"
                if cat == TOTAL_CAT:
                    note += "; universe total, not a religion category"
                out.append({"geo_id": code, "geo_level": level, "geo_name": name,
                            "source_category": cat, "count": n, "basis": BASIS,
                            "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return out


def check(rows):
    ok = True

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
    for lv, want in (("county", EXPECTED_COUNTIES), ("country", 1)):
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<9} {got:>4} units (expected {want})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(TOTAL_CAT) == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national universe {nat.get(TOTAL_CAT):,} "
          f"(expected {NATIONAL:,})")
    gap = CENSUS_POPULATION - nat.get(TOTAL_CAT, 0)
    print(f"      the census counted {CENSUS_POPULATION:,}; {gap:,} ({100.0 * gap / CENSUS_POPULATION:.2f}%) "
          "were in hotels, hospitals, prisons,\n      children's homes, travelling or "
          "sleeping outdoors and were never asked (the table's own footnote).")

    # KNBS neither suppresses nor rounds this table, so both identities are equalities.
    by_unit = {}
    for r in rows:
        by_unit.setdefault((r["geo_level"], r["geo_id"]), {})[r["source_category"]] = r["count"]
    bad = [k for k, d in by_unit.items()
           if sum(v for c, v in d.items() if c != TOTAL_CAT) != d[TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 13 categories sum to Total on all "
          f"{len(by_unit)} rows ({len(bad)} failures)")
    for k in bad[:5]:
        print(f"        {k}")

    bad = []
    for cat in CATEGORIES:
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "county" and r["source_category"] == cat)
        if s != nat[cat]:
            bad.append((cat, s, nat[cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all {len(CATEGORIES)} columns sum from the 47 "
          "counties to the KENYA row exactly")
    for c, s, n in bad:
        print(f"        {c}: {s:,} vs {n:,}")

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
    rows = read()
    check(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
