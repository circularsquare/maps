"""Sri Lanka — Department of Census and Statistics, CPH 2024, religion by GN division.

Reads (or fetches) data/raw/lk/ and writes data/normalized/lk.csv.

One workbook, `GN_Level_Population_by_Religion.xlsx`: **6 religion categories on 14,003
Grama Niladhari divisions**, 21,781,800 people. About 1,555 people per unit, which is the
finest geography in this project outside the US tracts and the German grid, and it arrives
that way with no allocation and no modelling at all.

**The trade is made at the other end.** Six categories is shallow — Buddhist, Hindu, Islam,
Roman Catholic, Other Christian, Other — and the census asks for a religion rather than a
body, so 15.2M Buddhists arrive with no school attached and 266k non-Catholic Christians
share one cell. §3.9's trade-off, at the geography end of it: Sri Lanka buys the finest
grain in the project by having almost nothing to say about branches. See sources/lk.md §2.

TWO THINGS IN THIS FILE WOULD GO WRONG QUIETLY.

- **`-` is an in-band sentinel and it is everywhere** — 39,241 cells, five of the seven
  numeric columns. It means zero. `errors="coerce"` would turn it into NaN and a plain
  `int()` would raise; either way the categories stop summing to the total. Every cell is
  classified one at a time here and anything unrecognised raises (§12).
- **The sheet name is a disclosure rule, not a label.** `By Religon(<10 add to Other)`:
  where a religion has fewer than 10 people in a GN division its count is moved into
  `Other`. So a `-` is "zero OR up to nine people who are now in Other", the small-group
  tail is unrecoverable at this geography, and `Other` is inflated by an unknown amount
  bounded by 9 x 14,003 x 5. §3.8, declared by the source rather than found.

Usage:
    python sources/lk.py --fetch    one workbook, ~1 MB
    python sources/lk.py            normalise from data/raw/lk/
"""

import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lk")
OUT = os.path.join(ROOT, "data", "normalized", "lk.csv")

SOURCE_ID = "lk_cph_2024"
YEAR = 2024
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

URL = ("https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024/"
       "GNLevel/GN_Level_Population_by_Religion")
XLSX = "GN_Level_Population_by_Religion.xlsx"

# The workbook's own column order, row 3. `Total` is the universe and is not a category.
CATEGORIES = ["Buddhist", "Hindu", "Islam", "Roman Catholic", "Other Christian", "Other"]
TOTAL_CAT = "Total"

NATIONAL = 21_781_800          # CPH 2024, and the workbook's own row 4
EXPECTED_GND = 14_003
EXPECTED_DSD = 340
EXPECTED_DISTRICTS = 25

HEADER_ROWS = 5                # rows 0-3 are titles and headers, row 4 is the national line
NATIONAL_ROW = 4


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, XLSX)
    if os.path.exists(dest) and os.path.getsize(dest) > 500_000:
        print("already have", dest)
        return
    print("GET", URL)
    # DCS serves the workbook from a path with no extension, as an attachment.
    r = requests.get(URL, timeout=300, verify=False,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    # §5a: HTTP 200 is not a download. This one is an xlsx or it is nothing.
    if r.content[:2] != b"PK":
        raise SystemExit(f"not a zip container: {r.content[:40]!r}")
    with open(dest, "wb") as fh:
        fh.write(r.content)
    import zipfile
    if not zipfile.is_zipfile(dest):
        raise SystemExit(f"{dest} is not a readable xlsx")
    print(f"  {os.path.getsize(dest):,} bytes -> {dest}")


def _cell(v, where):
    """Classify one numeric cell. `-` is zero; anything else unrecognised raises (§12)."""
    if isinstance(v, (int, float)) and v == v:
        if float(v) != int(v):
            raise SystemExit(f"{where}: non-integer count {v!r}")
        return int(v)
    s = str(v).strip()
    if s == "-":
        return 0
    raise SystemExit(f"{where}: unrecognised cell {v!r} -- a new sentinel, do not guess")


def read():
    import pandas as pd

    p = os.path.join(RAW, XLSX)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    raw = pd.read_excel(p, sheet_name=0, header=None, dtype=object)

    # The two header rows are asserted, not assumed: DCS could reorder the columns and
    # nothing else here would notice.
    head = [str(x).strip() if x == x else "" for x in raw.iloc[3].tolist()]
    want = [""] * 6 + [TOTAL_CAT] + CATEGORIES
    if head != want:
        raise SystemExit(f"header row 3 is {head!r}\n            expected {want!r}")

    # Positional, not by name: three of the column labels contain spaces and attribute
    # access would silently pick up pandas' renamed `_10`-style fields.
    body = raw.iloc[HEADER_ROWS:].to_numpy()
    D, DN, DS, DSN, GN, GNN = 0, 1, 2, 3, 4, 5
    COL = {cat: 6 + i for i, cat in enumerate([TOTAL_CAT] + CATEGORIES)}

    rows = []
    for r in body:
        d, ds, gn = int(r[D]), int(r[DS]), int(r[GN])
        # The census's OWN code, seven digits, deliberately NOT written in COD's `LK…`
        # form: sources/lk_geo.py shows the two disagree on 13 DS divisions and a shared
        # spelling would invite exactly the wrong join.
        geo_id = f"{d:02d}{ds:02d}{gn:03d}"
        name = str(r[GNN]).strip()
        note = (f"district={str(r[DN]).strip()}; dsd={str(r[DSN]).strip()}; "
                f"dsd_id={d:02d}{ds:02d}")
        where = f"{geo_id} {name}"
        for cat in [TOTAL_CAT] + CATEGORIES:
            n = _cell(r[COL[cat]], f"{where}/{cat}")
            note_c = note
            if cat == TOTAL_CAT:
                note_c += "; universe total, not a religion category"
            elif cat == "Other":
                note_c += "; inflated by the <10-per-GND disclosure rule"
            rows.append({"geo_id": geo_id, "geo_level": "gnd", "geo_name": name,
                         "source_category": cat, "count": n, "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": note_c})

    nat = raw.iloc[NATIONAL_ROW].tolist()
    national = {}
    for i, cat in enumerate([TOTAL_CAT] + CATEGORIES):
        national[cat] = _cell(nat[6 + i], f"national/{cat}")
    return rows, national


def check(rows, national):
    ok = True
    import collections

    gnds = {r["geo_id"] for r in rows}
    dsds = {g[:4] for g in gnds}
    districts = {g[:2] for g in gnds}
    for label, got, want in (("GN divisions", len(gnds), EXPECTED_GND),
                             ("DS divisions", len(dsds), EXPECTED_DSD),
                             ("districts", len(districts), EXPECTED_DISTRICTS)):
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label:<13} {got:>6,} (expected {want:,})")

    # Every GN division's six categories sum to its own total. DCS publishes both, so this
    # is an equality per row rather than a national one, and it is 14,003 separate checks.
    per = collections.defaultdict(dict)
    for r in rows:
        per[r["geo_id"]][r["source_category"]] = r["count"]
    bad = [g for g, d in per.items()
           if sum(d[c] for c in CATEGORIES) != d[TOTAL_CAT]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} all {len(per):,} GN divisions: the 6 categories "
          f"sum to the unit's own total ({len(bad)} failures)")

    print()
    for cat in [TOTAL_CAT] + CATEGORIES:
        s = sum(d[cat] for d in per.values())
        good = s == national[cat]
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {cat:<16} GN sum {s:>12,}  "
              f"national row {national[cat]:>12,}")

    good = national[TOTAL_CAT] == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national total {national[TOTAL_CAT]:,} "
          f"(published {NATIONAL:,})")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for cat in CATEGORIES:
        n = national[cat]
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:5.2f}%  {cat}")

    # How much of the map the <10 rule can be hiding, and the bound is exact in the
    # direction that matters: everything the rule moves lands in `Other`, so `Other` is
    # itself the ceiling on the damage. Counting the emptied cells says how widespread it
    # is; `Other` says how many people it can possibly be about (spec §3.8).
    zeros = sum(1 for r in rows
                if r["source_category"] in CATEGORIES[:-1] and r["count"] == 0)
    o = national["Other"]
    print(f"\n  the <10-per-GND rule: {zeros:,} of the {EXPECTED_GND * 5:,} named-category "
          f"cells are 0,\n  each of which is a true zero or up to nine people moved into "
          f"`Other`. Everything moved\n  lands there, so `Other` — {o:,} people, "
          f"{100.0 * o / NATIONAL:.2f}% — is the ceiling on how many\n  people this "
          "misfiles, and the floor on it is 0. Sri Lanka's small-group tail cannot be\n  "
          "recovered at this geography; the effect on everything else is under a third of "
          "a percent.")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, national = read()
    check(rows, national)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
