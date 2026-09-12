"""Vanuatu — VNSO, 2020 National Population and Housing Census, Basic Tables Vol 1, Table 3.5.

Reads (or fetches) data/raw/vu/ and writes data/normalized/vu.csv.

**THE TABLE GOES ALL THE WAY DOWN TO AREA COUNCILS, WHICH NOBODY EXPECTED.** The queue had
Vanuatu down as "6 provinces". Table 3.5's `Region` column is the whole census hierarchy —
VANUATU, URBAN (Port Vila, Luganville), RURAL, the six provinces, and then **64 rural area
councils** underneath them. 66 drawable units for 293,963 people, about 4,500 each, against
the six provinces anyone would have settled for.

**AND THE CATEGORIES SURVIVE THE FINE GEOGRAPHY**, which is the thing Fiji could not have
(§9bd §3): fourteen columns at every one of the 66 units, twelve of them substantive. No
trade-off to make here — VNSO publishes the deep list and the fine geography in the same
table.

**WHAT MAKES VANUATU WORTH DRAWING IS THAT NO CHURCH IS CLOSE TO A MAJORITY AND THE PROVINCES
DISAGREE COMPLETELY.** Presbyterian 27.2%, SDA 14.8%, Catholic 12.1%, Anglican 12.0% — and
Anglican is **77.4% of Torba and 0.2% of Tafea**. Presbyterian runs Malampa and Shefa and is
all but absent from Torba and Penama.

**CUSTOMARY BELIEFS ARE A PRINTED CATEGORY WITH 9,080 PEOPLE, AND THEY ARE ALMOST ALL ON ONE
ISLAND.** 7,757 of the 9,080 are in Tafea — Tanna, where the John Frum and Prince Philip
movements are — and South West Tanna is **30.3% customary**. This is the largest indigenous-
religion share of any unit on this map outside India.

**THE PUBLISHED TABLE DOES NOT ADD UP, AND THE MISS IS NEVER MORE THAN TWO PEOPLE.** 41 of
the 75 printed rows have their fourteen categories summing to one or two away from the printed
`Total`. It is not a parse error: `VANUATU`, `URBAN`, `Port Vila`, `Luganville`, `RURAL` and
`East Santo` were read off the rendered page digit by digit and all agree with what this file
extracts, and East Santo — printed 5,788, categories 5,786 — is wrong on the page itself.
Volume 2 independently reproduces all thirteen national figures. So it is VNSO's arithmetic,
consistent with each cell being rounded on its own; see `sources/vu.md` §4.

The categories are what this map draws, so the CATEGORY SUM is used as each unit's universe
and the printed `Total` is carried alongside for reporting. At a maximum of two people against
units averaging 4,500 this changes nothing anyone can see, but it is asserted rather than
ignored: if a row ever misses by more than 2, something has changed and this file stops.

Usage:
    python sources/vu.py --fetch    two PDFs from vnso.gov.vu (~26 MB)
    python sources/vu.py            normalise from data/raw/vu/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "vu")
OUT = os.path.join(ROOT, "data", "normalized", "vu.csv")

SOURCE_ID = "vu_nphc_2020"
YEAR = 2020
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

BASE = "https://vnso.gov.vu"
VOL1_NAME = "vu_2020_basic_tables_vol1.pdf"
VOL1_URL = (BASE + "/images/Public_Documents/Census_Surveys/Census/2020/Basic_Tables/"
            "2020NPHC_Volume_1_-_Version_2.pdf")
VOL2_NAME = "vu_2020_analytical_vol2.pdf"
VOL2_URL = (BASE + "/images/Public_Documents/Census_Surveys/Census/2020/"
            "2020_Vanuatu_National_Population_and_Housing_Census_-_Analytical_report_"
            "Volume_2.pdf")

# Table 3.5 is split across four pages: 62-63 carry `Total` plus eight religion columns and
# 64-65 the remaining six, with the SAME row list repeated. Asserted in read().
BLOCK_A_PAGES = [62, 63]
BLOCK_B_PAGES = [64, 65]
COLS_A = ["Total", "Presbyterian", "Seventh Day Adventist (SDA)", "Catholic", "Anglican",
          "Churches of Christ", "Assemblies of God (AOG)",
          "Neil Thomas Ministry / Inner Life Ministry", "Customary beliefs"]
COLS_B = ["Apostolic", "Latter Day Saints (Mormon)", "Other churches",
          "No Religion/Faith", "Refuse to answer", "Not Stated"]
COLS = COLS_A + COLS_B
TOTAL_CAT = "Total"
# §3.5 residuals: marked, not filled, and not drawn.
RESIDUALS = ["Refuse to answer", "Not Stated"]
DRAWN = [c for c in COLS[1:] if c not in RESIDUALS]

# Geometry of the page. Everything left of LABEL_X is the `Region` column; a value token must
# start to the right of it. See the two traps in read().
LABEL_X = 125.0
NUM = re.compile(r"^[\d][\d,]*$")

PROVINCES = ["TORBA", "SANMA", "PENAMA", "MALAMPA", "SHEFA", "TAFEA"]
URBAN_UNITS = ["Port Vila", "Luganville"]
STRUCTURAL = ["VANUATU", "URBAN", "RURAL"] + PROVINCES

# The published table rounds each cell on its own, so a row's categories can miss its
# printed Total. Never by more than this; asserted, not assumed.
ROUNDING_TOLERANCE = 2

NATIONAL = 293_963
EXPECTED_UNITS = 66
EXPECTED_COUNCILS = 64

# Volume 2, Table 30, the 2020 column — a different volume of the same census, typeset
# separately. `Not Stated` is not one of its rows; it folds into the 428 its text quotes.
VOL2_TABLE30_2020 = {
    "Anglican": 35_339, "Presbyterian": 80_060, "Catholic": 35_602,
    "Seventh Day Adventist (SDA)": 43_541, "Churches of Christ": 14_588,
    "Assemblies of God (AOG)": 14_450, "Neil Thomas Ministry / Inner Life Ministry": 9_515,
    "Apostolic": 6_894, "Customary beliefs": 9_080, "Latter Day Saints (Mormon)": 5_174,
    "No Religion/Faith": 4_023, "Refuse to answer": 394, "Other churches": 35_270,
}


def fetch():
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
    for name, url in ((VOL1_NAME, VOL1_URL), (VOL2_NAME, VOL2_URL)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print("already have", dest)
            continue
        print("GET", url)
        # vnso.gov.vu's certificate chain does not verify from here; the files are public
        # census PDFs and are checked for a %PDF header and an %%EOF trailer below.
        r = requests.get(url, headers=ua, timeout=1800, verify=False)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise SystemExit(f"vnso returned something that is not a PDF for {name} "
                             f"({len(r.content):,} bytes)")
        if b"%%EOF" not in r.content[-4096:]:
            raise SystemExit(f"{name} has no %%EOF trailer -- truncated at source")
        with open(dest, "wb") as fh:
            fh.write(r.content)
        print(f"  {os.path.getsize(dest):,} bytes")


def _visual_rows(page, ytol=3.0):
    """Words grouped into visual rows: [(y, [(x0, x1, text), ...]), ...]."""
    words = [w for w in page.get_text("words") if w[4].strip()]
    rows = []
    for x0, y0, x1, y1, txt, *_ in words:
        for r in rows:
            if abs(r[0] - y0) <= ytol:
                r[1].append((x0, x1, txt))
                break
        else:
            rows.append([y0, [(x0, x1, txt)]])
    for r in rows:
        r[1].sort(key=lambda t: t[0])
    rows.sort(key=lambda r: r[0])
    return rows


def _parse_page(doc, pno, ncols):
    """One page of Table 3.5 as [(region, [values])].

    The numbers are right-aligned, so the columns are recovered by clustering the RIGHT edge
    of every numeric token. Two traps put a row LABEL into the data if the split is done on
    text order or on a bare "is it numeric" test:

      * `Canal - Fanafo` contains a hyphen, which is also this table's symbol for zero;
      * `Central Pentecost 1` and `Central Pentecost 2` end in a digit.

    Both live in the label column, so the x test is what separates them.
    """
    page = doc[pno - 1]
    rows = _visual_rows(page)

    # The header band is NOT at a fixed height -- page 62 puts `Region` at y=117 and page 63
    # at y=105, so SHEFA (y=118) falls above any cutoff tuned on page 62 and vanishes from
    # the table without a word. Anchor on the header row itself.
    head_y = None
    for y, cells in rows:
        if any(t == "Region" and x0 < LABEL_X for x0, x1, t in cells):
            head_y = y
    if head_y is None:
        raise SystemExit(f"page {pno}: no `Region` header row -- the PDF has changed")

    data = []
    for y, cells in rows:
        if y <= head_y + 3.0:
            continue
        label = " ".join(t for x0, x1, t in cells if x0 < LABEL_X).strip()
        label = " ".join(re.sub(r"\s*-\s*", " - ", label).split())
        nums = [(x1, t) for x0, x1, t in cells
                if x0 >= LABEL_X and (NUM.match(t) or t == "-")]
        if not label or len(nums) < 2:
            continue
        data.append((label, nums))
    if not data:
        raise SystemExit(f"page {pno}: no data rows found below the header")

    edges = sorted(x1 for _, nums in data for x1, _ in nums)
    clusters = [[edges[0]]]
    for e in edges[1:]:
        if e - clusters[-1][-1] <= 12:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    centres = [sum(c) / len(c) for c in clusters]
    if len(centres) != ncols:
        raise SystemExit(f"page {pno}: {len(centres)} column clusters, expected {ncols} "
                         f"-- centres {[round(c) for c in centres]}")

    out = []
    for label, nums in data:
        vals = [None] * ncols
        for x1, t in nums:
            j = min(range(ncols), key=lambda k: abs(centres[k] - x1))
            if vals[j] is not None:
                raise SystemExit(f"page {pno} row {label!r}: two values in column {j}")
            vals[j] = 0 if t == "-" else int(t.replace(",", ""))
        if any(v is None for v in vals):
            raise SystemExit(f"page {pno} row {label!r}: a column came out empty")
        out.append((label, vals))
    return out


def read():
    import fitz

    path = os.path.join(RAW, VOL1_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run with --fetch first")
    doc = fitz.open(path)
    if doc.page_count < max(BLOCK_B_PAGES):
        raise SystemExit(f"{path}: {doc.page_count} pages, expected at least "
                         f"{max(BLOCK_B_PAGES)}")
    a, b = [], []
    for p in BLOCK_A_PAGES:
        a += _parse_page(doc, p, len(COLS_A))
    for p in BLOCK_B_PAGES:
        b += _parse_page(doc, p, len(COLS_B))
    doc.close()

    if len(a) != len(b):
        raise SystemExit(f"the two column blocks disagree on row count: {len(a)} vs {len(b)}")
    bad = [(x[0], y[0]) for x, y in zip(a, b) if x[0] != y[0]]
    if bad:
        raise SystemExit(f"row labels differ between the column blocks: {bad[:6]}")

    order = [lab for lab, _ in a]
    table = {lab: va + vb for (lab, va), (_, vb) in zip(a, b)}
    return order, table


def _councils(order):
    """province -> its area council rows, from the printed row order."""
    idx = {lab: i for i, lab in enumerate(order)}
    missing = [p for p in STRUCTURAL if p not in idx]
    if missing:
        raise SystemExit(f"Table 3.5 is missing structural rows {missing}")
    out = {}
    for p in PROVINCES:
        start = idx[p] + 1
        end = min([idx[q] for q in PROVINCES if idx[q] > idx[p]] + [len(order)])
        out[p] = order[start:end]
    return out


def build(order, table):
    councils = _councils(order)
    rows = []
    for prov, names in councils.items():
        for name in names:
            v = table[name]
            for cat, n in zip(COLS[1:], v[1:]):
                rows.append({"geo_id": name, "geo_level": "area_council", "geo_name": name,
                             "source_category": cat, "count": n, "basis": BASIS,
                             "year": YEAR, "source_id": SOURCE_ID,
                             "note": f"level=area_council; province={prov}"
                                     + ("; §3.5 residual, not drawn" if cat in RESIDUALS
                                        else "")})
            rows.append({"geo_id": name, "geo_level": "area_council", "geo_name": name,
                         "source_category": TOTAL_CAT, "count": v[0], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID,
                         "note": f"level=area_council; province={prov}; printed universe "
                                 "total, not a religion category"})
    for name in URBAN_UNITS:
        v = table[name]
        for cat, n in zip(COLS[1:], v[1:]):
            rows.append({"geo_id": name, "geo_level": "area_council", "geo_name": name,
                         "source_category": cat, "count": n, "basis": BASIS, "year": YEAR,
                         "source_id": SOURCE_ID,
                         "note": "level=area_council; urban municipality"
                                 + ("; §3.5 residual, not drawn" if cat in RESIDUALS
                                    else "")})
        rows.append({"geo_id": name, "geo_level": "area_council", "geo_name": name,
                     "source_category": TOTAL_CAT, "count": v[0], "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID,
                     "note": "level=area_council; urban municipality; printed universe "
                             "total, not a religion category"})
    return rows, councils


def check(order, table, rows, councils):
    ok = True
    units = sorted({r["geo_id"] for r in rows})
    good = len(units) == EXPECTED_UNITS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {len(units)} drawable units "
          f"(expected {EXPECTED_UNITS}: {EXPECTED_COUNCILS} rural area councils + "
          f"{len(URBAN_UNITS)} urban)")

    ncouncil = sum(len(v) for v in councils.values())
    good = ncouncil == EXPECTED_COUNCILS
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} {ncouncil} area councils across the 6 provinces "
          + ", ".join(f"{p} {len(councils[p])}" for p in PROVINCES))

    nat = table["VANUATU"]
    good = nat[0] == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} printed national total {nat[0]:,} "
          f"(expected {NATIONAL:,})")

    # ---- THE ROUNDING, which is the source's and not ours (see the module docstring) ----
    off = [(lab, table[lab][0], sum(table[lab][1:])) for lab in order
           if sum(table[lab][1:]) != table[lab][0]]
    worse = [x for x in off if abs(x[2] - x[1]) > ROUNDING_TOLERANCE]
    ok &= not worse
    hist = {}
    for _, t, s in off:
        hist[s - t] = hist.get(s - t, 0) + 1
    print(f"\n  {'OK ' if not worse else 'BAD'} the fourteen categories sum to the printed "
          f"Total on {len(order) - len(off)}/{len(order)} rows, and the")
    print(f"      other {len(off)} miss by at most {ROUNDING_TOLERANCE}: "
          + ", ".join(f"{d:+d} on {n}" for d, n in sorted(hist.items()))
          + f"  ({len(worse)} exceed {ROUNDING_TOLERANCE})")
    print("      NOT a parse error. Six rows were read off the rendered page digit by "
          "digit and\n      agree with this extraction, and East Santo — printed 5,788, "
          "categories 5,786 — is\n      wrong on the page itself. Volume 2 reproduces every "
          "national figure. The CATEGORY\n      SUM is each unit's universe; the printed "
          "Total is carried for reporting only.")

    # ---- the hierarchy the table prints, at every level ----
    def cmp(label, got, want):
        nonlocal ok
        d = [g - w for g, w in zip(got, want)]
        good = all(abs(x) <= ROUNDING_TOLERANCE for x in d)
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} on "
              f"{sum(1 for x in d if not x)}/{len(COLS)} columns exactly, "
              f"max |diff| {max(abs(x) for x in d)}")

    urb, rur = table["URBAN"], table["RURAL"]
    cmp("URBAN + RURAL = VANUATU",
        [u + r for u, r in zip(urb, rur)], nat)
    cmp("Port Vila + Luganville = URBAN",
        [p + l for p, l in zip(table["Port Vila"], table["Luganville"])], urb)
    cmp("the 6 provinces sum to RURAL",
        [sum(table[p][i] for p in PROVINCES) for i in range(len(COLS))], rur)
    for p in PROVINCES:
        cmp(f"{p}'s {len(councils[p])} area councils sum to {p}",
            [sum(table[c][i] for c in councils[p]) for i in range(len(COLS))], table[p])

    # ---- witness: Volume 2, Table 30, typeset separately from Volume 1 ----
    print("\n  the WITNESS — Volume 2's Table 30, a different volume of the same census:")
    bad = [(c, nat[COLS.index(c)], w) for c, w in VOL2_TABLE30_2020.items()
           if nat[COLS.index(c)] != w]
    ok &= not bad
    print(f"    {'OK ' if not bad else 'BAD'} all {len(VOL2_TABLE30_2020)} of its 2020 "
          f"figures match Table 3.5 exactly ({len(bad)} failures)")
    for c, g, w in bad[:6]:
        print(f"        {c}: Vol 1 {g:,} vs Vol 2 {w:,}")
    print("       Vol 2 has no `Not Stated` row; its text gives 428 for refusals and "
          "non-response,\n       which is exactly this table's 394 + 34.")

    drawn = sum(nat[COLS.index(c)] for c in DRAWN)
    print(f"\n  drawn {drawn:,} of {nat[0]:,} — {100.0 * drawn / nat[0]:.2f}%. "
          f"The {sum(nat[COLS.index(c)] for c in RESIDUALS):,} not drawn are "
          f"{' + '.join(RESIDUALS)} (§3.5).")

    print("\n  the drawn categories, national:")
    for c in sorted(DRAWN, key=lambda c: -nat[COLS.index(c)]):
        print(f"    {nat[COLS.index(c)]:>8,}  {100.0 * nat[COLS.index(c)] / nat[0]:6.2f}%  "
              f"{c}")

    # ---- what the map is for: the provinces disagree completely ----
    print("\n  Anglican and Presbyterian by province — the reason the fine tier matters:")
    for p in PROVINCES:
        v, t = table[p], table[p][0]
        print(f"    {p:<9} Anglican {100.0 * v[COLS.index('Anglican')] / t:5.1f}%   "
              f"Presbyterian {100.0 * v[COLS.index('Presbyterian')] / t:5.1f}%   "
              f"Customary {100.0 * v[COLS.index('Customary beliefs')] / t:5.1f}%")

    cb = COLS.index("Customary beliefs")
    top = sorted(((table[c][cb] / table[c][0], c) for v in councils.values() for c in v
                  if table[c][0] > 0), reverse=True)
    print("\n  `Customary beliefs` is 9,080 people and almost all of them are on Tanna:")
    for share, c in top[:6]:
        print(f"    {c:<20} {table[c][cb]:>6,} of {table[c][0]:>6,}  {100 * share:5.1f}%")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    order, table = read()
    rows, councils = build(order, table)
    check(order, table, rows, councils)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
