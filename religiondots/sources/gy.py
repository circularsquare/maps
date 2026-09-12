"""Guyana — Bureau of Statistics, 2012 Population and Housing Census, religion by region.

Reads (or fetches) data/raw/gy/ and writes data/normalized/gy.csv.

**Table 2.19**, *Distribution of the Population by Religious Affiliation and Administrative
Regions*, one page of the 66-page *Final 2012 Census Compendium 2: Population Composition*.
13 categories over the 10 administrative regions, 746,955 people. No API, no portal, no
login — a single 3.6 MB PDF on the office's own publications page.

**§12's Kenya rule collecting again: the size of the TABLE is the predictor, not the size of
the report.** This is a PDF-only office with no dissemination platform of any kind, and it
still took an afternoon, because the whole religion cross-tabulation is one page.

WHY GUYANA IS WORTH DRAWING. It is the only country on this map where Hinduism and Islam are
both large minorities of a Christian-majority country in the Americas — 24.8% Hindu and 6.8%
Muslim, the indentured-labour inheritance — and the Christian side is split seven ways
(Anglican, Methodist, Pentecostal, Roman Catholic, Jehovah's Witness, Seventh Day Adventist,
Other Christian) rather than left as one cell. It also counts **Rastafarians** (3,496)
directly, which almost nothing else does. 74,700 people per unit is a finer grain than
Kenya's 1,012,000 and close to Lithuania's 40,000.

THE TOTAL COLUMN IS CLIPPED IN THE PDF AND MUST NOT BE READ. The rightmost column of Table
2.19 overflows its cell, so the text layer holds `38,96` where the figure is 38,962, `170,2`
for 170,289 and `746,9` for 746,955. Nothing about this is visible in reading order and the
truncated values parse perfectly happily as numbers — `38,96` is a number. **Every row total
is recomputed from the ten regions**, and the clipped string is then asserted to be a PREFIX
of that sum, which turns the defect into a check: it fails if a row is misparsed AND keeps
passing if the office ever re-renders the PDF with a wider column.

THE INDEPENDENT CHECK IS A SECOND PUBLISHED TABLE. `Table 2.17` on page 42 gives the 2012
national figure for every category, uncut, from a different page and a different layout. The
ten regions sum to it exactly, category by category — a quantity the regional parse does not
determine, which is §12's strongest available check. The column sums close on the region
totals as well, and the grand total is the published census population, so the table is a
perfect partition in both directions.

NON-RESPONSE IS PRORATED INTO THE CATEGORIES BY THE OFFICE, AND THERE IS NO WAY TO UNDO IT.
Table 2.19's own note: *"'363 Religious Affiliation Not Stated' added to '16,331 No-Contact
Persons' and '7,443 Institution Population' and prorated."* So 24,137 people — 3.2% of
Guyana — were distributed across the thirteen categories in proportion, and every count here
carries its share of them. This is the first source on the map that does this. spec §3.5 says
non-response is reported and not filled; here the FILLING HAS ALREADY HAPPENED upstream and
is not separable, which is a different situation and is recorded rather than corrected
(§14.4 — undoing it would mean inventing a distribution the office did not publish).

Usage:
    python sources/gy.py --fetch    one GET, 3.6 MB
    python sources/gy.py            normalise from data/raw/gy/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "gy")
OUT = os.path.join(ROOT, "data", "normalized", "gy.csv")

SOURCE_ID = "gy_census_2012"
YEAR = 2012
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

URL = ("https://statisticsguyana.gov.gy/wp-content/uploads/2019/10/"
       "Final_2012_Census_Compendium2.pdf")
PDF = os.path.join(RAW, "Final_2012_Census_Compendium2.pdf")

# 0-based page indices. Asserted by title text before anything is read off them, so a
# repagination fails loudly instead of parsing the wrong table (ke.py's rule).
PAGE_REGIONAL = 48          # Table 2.19, counts by region
PAGE_NATIONAL = 41          # Table 2.17, national counts 2002 and 2012

TITLE_REGIONAL = re.compile(
    r"Table\s+2\.19:\s*Distribution of the\s+Population by Religious Affiliation and\s+"
    r"Administrative Regions", re.I)
TITLE_NATIONAL = re.compile(
    r"Table\s+2\.17:\s*Distribution of the Population by Religious Affiliation", re.I)

REGIONS = 10
NATIONAL = 746_955          # 2012 census population, and the table's own grand total

# The category labels exactly as Table 2.19 prints them, in row order. Multi-word labels are
# broken across lines in the PDF ("Roman"/"Catholic", "Seventh"/"Day"/"Adventist"), so the
# reader joins a row's label fragments; these are the joined forms and they are the mapping
# keys taxonomy/gy2012.py uses. Kept verbatim, including `Jehovah Witness` without the
# apostrophe-s and `Other Christians` in the plural, per §12.
CATEGORIES = [
    "Anglican",
    "Methodist",
    "Pentecostal",
    "Roman Catholic",
    "Jehovah Witness",
    "Seventh Day Adventist",
    "Bahai",
    "Muslim",
    "Hindu",
    "Rastafarian",
    "Other Christians",
    "None",
    "Other",
]
TOTAL_ROW = "Total"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) > 3_000_000:
        print("already have", PDF)
        return
    r = requests.get(PDF and URL, timeout=300, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
    r.raise_for_status()
    # §5a: HTTP 200 is not a download. Assert type and size, not the absence of an exception.
    if r.content[:4] != b"%PDF":
        raise SystemExit(f"not a PDF -- starts {r.content[:16]!r}")
    if len(r.content) < 3_000_000:
        raise SystemExit(f"only {len(r.content):,} bytes; expected ~3.6 MB")
    with open(PDF, "wb") as fh:
        fh.write(r.content)
    print(f"wrote {PDF} ({len(r.content):,} bytes)")


def _rows(page):
    """Cluster a page's words into visual rows.

    Reading order is useless on this table -- it interleaves the label fragments with the
    figures -- so rows are rebuilt from y-coordinates and each row is sorted by x.
    """
    out = []
    for w in sorted(page.get_text("words"), key=lambda w: (w[1], w[0])):
        yc = (w[1] + w[3]) / 2.0
        for r in out:
            if abs(r["y"] - yc) < 4.5:
                r["w"].append(w)
                break
        else:
            out.append({"y": yc, "w": [w]})
    for r in out:
        r["w"].sort(key=lambda w: w[0])
    return out


NUM = re.compile(r"^\d{1,3}(?:,\d{3})*$|^\d+$")


def _split(row):
    """A row -> (label words, numeric tokens) split at the first number."""
    words = [w[4] for w in row["w"]]
    for i, t in enumerate(words):
        if NUM.match(t):
            return words[:i], words[i:]
    return words, []


def _records(page, want):
    """Numeric rows, each carrying the label fragments that TRAIL it.

    A multi-word category is broken across lines with the figures on the FIRST of them and
    the rest of the label below: `Other` + the counts, then `Christians` on its own line.
    Accumulating fragments forwards instead reads that row as the category `Other`, which
    also exists two rows later -- the two collide and one silently wins. So a record is a
    row with numbers plus the label-only rows that follow it.

    The extension stops as soon as it would stop looking like a label, because the row
    AFTER `Total` is the table's footnote and its figures are written `'363` and `'16,331`
    with a leading apostrophe -- not numbers to this reader, so an unbounded walk swallows
    the whole note into the `Total` label and then cannot find `Total` at all.
    """
    rows = [_split(r) for r in _rows(page)]
    out = []
    for i, (label, nums) in enumerate(rows):
        if not nums:
            continue
        name = " ".join(label)
        for j in range(i + 1, len(rows)):
            if rows[j][1]:                 # the next row with numbers ends this label
                break
            longer = " ".join([name] + rows[j][0]).strip()
            if not any(w == longer or w.startswith(longer + " ") for w in want):
                break                      # no wanted label continues this way
            name = longer
        out.append((name, nums))
    return out


def _int(tok):
    return int(tok.replace(",", ""))


def read():
    import fitz

    doc = fitz.open(PDF)

    reg_page = doc[PAGE_REGIONAL]
    if not TITLE_REGIONAL.search(" ".join(reg_page.get_text().split())):
        raise SystemExit(
            f"page {PAGE_REGIONAL + 1} is not Table 2.19 -- it starts "
            f"{' '.join(reg_page.get_text().split())[:120]!r}. The compendium has been "
            "repaginated; find the table and update PAGE_REGIONAL.")

    # ---- Table 2.19: counts by region ------------------------------------------------
    # A data row is one whose label fragments accumulate to a known category and which
    # carries 11 numbers: ten regions plus the clipped total.
    want = list(CATEGORIES) + [TOTAL_ROW]
    found, clipped = {}, {}
    for name, nums in _records(reg_page, want):
        if name not in want:
            continue
        if len(nums) != REGIONS + 1:
            raise SystemExit(f"row {name!r} has {len(nums)} numbers, expected {REGIONS + 1} "
                             f"(ten regions and the clipped total): {nums}")
        if name in found:
            raise SystemExit(f"row {name!r} appears twice on page {PAGE_REGIONAL + 1}")
        found[name] = [_int(t) for t in nums[:REGIONS]]
        clipped[name] = nums[REGIONS]

    missing = [c for c in want if c not in found]
    if missing:
        raise SystemExit(f"rows not found on page {PAGE_REGIONAL + 1}: {missing}")

    # ---- Table 2.17: the national figures, for the independent check ------------------
    nat_page = doc[PAGE_NATIONAL]
    if not TITLE_NATIONAL.search(" ".join(nat_page.get_text().split())):
        raise SystemExit(
            f"page {PAGE_NATIONAL + 1} is not Table 2.17 -- it starts "
            f"{' '.join(nat_page.get_text().split())[:120]!r}.")

    published = {}
    for name, nums in _records(nat_page, CATEGORIES):
        if name in CATEGORIES and len(nums) == 6:
            # Male, Female, Total for 2002 then the same for 2012. The last is 2012 Total.
            published[name] = _int(nums[5])

    doc.close()

    rows = []
    for level, ids in (("country", ["0"]), ("region", [str(i) for i in range(1, REGIONS + 1)])):
        for gid in ids:
            # The Total row is emitted as a category so taxonomy/gy2012.py can EXCLUDE it
            # by name and tools/check_mapping.py has a universe to report coverage against.
            for cat in CATEGORIES + [TOTAL_ROW]:
                if level == "country":
                    n = sum(found[cat])
                    name = "Guyana"
                else:
                    n = found[cat][int(gid) - 1]
                    name = f"Region {gid}"
                note = f"level={level}"
                if cat == TOTAL_ROW:
                    note += "; universe total, not a religion category"
                rows.append({"geo_id": gid, "geo_level": level, "geo_name": name,
                             "source_category": cat, "count": n, "basis": BASIS,
                             "year": YEAR, "source_id": SOURCE_ID, "note": note})
    return rows, found, clipped, published


def check(rows, found, clipped, published):
    ok = True

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
    for lv, want in (("region", REGIONS), ("country", 1)):
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<8} {got:>3} units (expected {want})")

    # 1. the grand total is the published census population
    grand = sum(found[TOTAL_ROW])
    good = grand == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} grand total {grand:,} (census {NATIONAL:,})")

    # 2. the thirteen categories sum to the Total row in every region
    bad = []
    for i in range(REGIONS):
        s = sum(found[c][i] for c in CATEGORIES)
        if s != found[TOTAL_ROW][i]:
            bad.append((i + 1, s, found[TOTAL_ROW][i]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the {len(CATEGORIES)} categories sum to Total "
          f"in all {REGIONS} regions ({len(bad)} failures)")
    for r, s, t in bad[:5]:
        print(f"        Region {r}: {s:,} vs {t:,}")

    # 3. THE INDEPENDENT ONE. Table 2.17 is a different page and a different layout, and
    #    the regional parse does not determine it.
    bad = []
    for cat in CATEGORIES:
        if cat not in published:
            continue
        s = sum(found[cat])
        if s != published[cat]:
            bad.append((cat, s, published[cat]))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} {len(published)} categories sum from the "
          f"{REGIONS} regions to Table 2.17's national figure exactly ({len(bad)} failures)")
    for c, s, n in bad[:5]:
        print(f"        {c}: {s:,} vs {n:,}")
    if len(published) < 11:
        ok = False
        print(f"  BAD only {len(published)} of {len(CATEGORIES)} categories were read from "
              "Table 2.17; the independent check is too weak to rely on")

    # 4. the clipped total column must be a PREFIX of the computed sum
    bad = []
    for cat in CATEGORIES + [TOTAL_ROW]:
        s = f"{sum(found[cat]):,}"
        if not s.startswith(clipped[cat]):
            bad.append((cat, clipped[cat], s))
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the clipped Total column is a prefix of the "
          f"recomputed sum on all {len(CATEGORIES) + 1} rows ({len(bad)} failures)")
    for c, t, s in bad[:5]:
        print(f"        {c}: PDF shows {t!r}, regions sum to {s!r}")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES:
        n = sum(found[cat])
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {cat}")
    print(f"    {'-' * 10}")
    print(f"    {NATIONAL:>10,}  100.00%  {TOTAL_ROW}")

    print("\n  Prorated into the above by the Bureau of Statistics and not separable:")
    print("     363 religious affiliation not stated, 16,331 no-contact persons and")
    print("     7,443 institutional population -- 24,137 people, 3.23% of Guyana.")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing -- run: python sources/gy.py --fetch")
    rows, found, clipped, published = read()
    check(rows, found, clipped, published)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
