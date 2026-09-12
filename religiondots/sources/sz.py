"""Eswatini — Central Statistical Office, 2017 Population and Housing Census, Volume 3.

Reads (or fetches) data/raw/sz/ and writes data/normalized/sz.csv.

**Twenty categories on four regions for 1,093,238 people**, and the twenty are the reason to
want this country: `Zionists` alone are **367,290 people, 33.60% of Eswatini**, which makes
`christianity.africaninstituted` the PLURALITY religion of a country for the first time on
this map. Zimbabwe's Vapostori are 40.3% of a much larger country and are its largest single
answer too, so Eswatini is the second of that shape and the first where the African
Instituted churches are counted alongside a full mission-denomination list rather than as one
undivided cell beside `Protestant`.

**THE HOST EVERY EARLIER NOTE NAMES IS DEAD AND THE FILE IS ON THE GOVERNMENT PORTAL.**
`eswatinistats.org.sz` resolves to 102.23.132.23 and times out on every scheme and both
ports; `swazistats.org.sz` does not resolve; the Wayback CDX has 61 captures for the first
hostname and **not one PDF**. sources.md §11w recorded Eswatini as "no reachable host" on
that evidence and it was right about the office. It is the wrong place to look: the census
volumes are articles on `www.gov.sz`, the Joomla government portal, under
`/images/FinanceDocuments/`. See sources/sz.md §1 for how the article was found, because the
route generalises.

**RELIGION BY REGION IS A CHRISTIAN-ONLY TABLE, AND THE RESIDUAL IS STILL COUNTED.**
Table 3.2.4 is *Christians by Denomination and Regions* — thirteen denominations on four
regions. The nine top-level religions of Table 3.2.1 (Christian, Islam, Hindu, Baha'i,
Traditionalist, Judaism, Other, No religion, Not Stated) are published NATIONALLY ONLY. So
each region's non-Christian total is known exactly, as its published population (Table 5.2.2)
minus its published Christians (Table 3.2.4), and only the SPLIT of that total across the
eight non-Christian answers is carried down from the national table. Those eight rows are
`derived` and the thirteen Christian rows are `measured`; nothing invents a magnitude.

**SIX TABLES ARE READ AND ONLY ONE IS DRAWN.** 3.2.1 (national religion), 3.2.2 (national
Christian denominations), 3.2.3 (the same by urban/rural), 3.2.4 (the same by region, the
drawn one), 3.2.5 (3.2.4 as percentages) and 5.2.2 (region populations). The other five are
read because **every identity inside 3.2.4 survives a consistent permutation of its four
region columns** — Zimbabwe's warning — and the checks that do not are all cross-table:

  * 3.2.4's `Total` column must equal 3.2.2's `Total` column on all thirteen rows;
  * 3.2.3's urban + rural must equal 3.2.2's total on all thirteen rows;
  * 3.2.2's thirteen denominations must sum to 3.2.1's `Christian` cell, 975,757;
  * Male + Female == Total on every row of 3.2.1 and 3.2.2;
  * 3.2.5's printed percentage must reproduce 3.2.4's cell over its region's Christian
    total, on all 52 cells;
  * and **the permutation is tested directly.** Each region's Christians over its Table 5.2.2
    population must land near the national 89.25%; the true pairing gives 88.42-89.77% and
    all 23 wrong pairings of the four columns are checked to fail that band. That check
    crosses two chapters of the volume and is the only one a consistent column swap cannot
    survive.

**AND THE UNSD DEMOGRAPHIC YEARBOOK IS AN INDEPENDENT TRANSCRIPTION OF THE SAME TABLES.**
Table 28 carries Eswatini 2017 with twenty categories summing exactly to 1,093,238 — the CSO
forwarded a merged list (the thirteen denominations, seven non-Christian answers, and `Other
Christians` = this volume's Christian `Other` + Christian `Not Stated`). Every figure is
asserted against it. That is §0.5's third use of the oracle: not what exists and not how
deep, but a second pair of eyes on the parse, keyed to a file nobody here produced.

Usage:
    python sources/sz.py --fetch    one 3.0 MB PDF from www.gov.sz, seconds
    python sources/sz.py            normalise from data/raw/sz/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "sz")
OUT = os.path.join(ROOT, "data", "normalized", "sz.csv")

SOURCE_ID = "sz_phc_2017_v3"
YEAR = 2017
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The Joomla article `2455-eswatini-census-documents` (now 404, last captured 2024-07-24)
# linked Volumes 3, 5 and 6 out of the FINANCE ministry's upload folder. Volume 3 is the one
# with religion in it; 5 is literacy/economic activity and 6 is disability. There is no
# Volume 1 or 4 at this path and Volume 2, the Census Atlas, is under `planningministry/` as
# `census2017.pdf` — tinkhundla maps, no religion. sources/sz.md §1.
PDF_URL = "https://www.gov.sz/images/FinanceDocuments/Volume-3.pdf"
PDF_NAME = "sz_phc2017_volume3.pdf"

# 0-based page indices in Volume 3.
PAGE_321 = 30      # Table 3.2.1 and Table 3.2.2 share a page
PAGE_323 = 31
PAGE_324 = 32      # Table 3.2.4 and Table 3.2.5 share a page
PAGE_522 = 46

TITLES = {
    "3.2.1": re.compile(r"Table\s*3\.2\.1\s*Distribution of the Population by Religious "
                        r"Affiliation", re.I),
    "3.2.2": re.compile(r"Table\s*3\.2\.2\s*Christians by Denomination", re.I),
    "3.2.3": re.compile(r"Table\s*3\.2\.3\s*Christians by Denomination and Residence", re.I),
    "3.2.4": re.compile(r"Table\s*3\.2\.4\s*Distribution of Christians by Denomination and "
                        r"Regions", re.I),
    "3.2.5": re.compile(r"Table\s*3\.2\.5\s*Percent Distribution of Christians by "
                        r"Denomination and Regions", re.I),
    "5.2.2": re.compile(r"Table\s*5\.2\.2\s*Urban-rural ratio by region", re.I),
}

# Table 3.2.1's rows, in print order. `Christian` is the one that is split further.
TOP_LEVEL = ["Christian", "Islam", "Hindu", "Bahai Faith", "Traditionalist", "Judaism",
             "Other", "No religion", "Not Stated"]
NON_CHRISTIAN = [c for c in TOP_LEVEL if c != "Christian"]

# Tables 3.2.2 / 3.2.3 / 3.2.4 / 3.2.5, in print order. The CSO prints trailing spaces on
# several of these labels; `_key` folds whitespace and nothing else (§2.4, transcribe).
DENOMINATIONS = ["Roman Catholic", "Anglican", "Lutheran", "Methodist", "Jehovah Witness",
                 "Evangelical", "Pentecostal", "Zionists", "Apostles", "Nazarene",
                 "Seventh Day Adventist", "Other", "Not Stated"]

REGIONS = ["Hhohho", "Manzini", "Shiselweni", "Lubombo"]

NATIONAL = 1_093_238
CHRISTIANS = 975_757

# The CSV keeps the two tables' identically-named cells apart. Table 3.2.2's `Other` is
# 13,458 other CHRISTIAN denominations and Table 3.2.1's is 3,363 other RELIGIONS; its
# `Not Stated` is 13 Christians who named no denomination and 3.2.1's is 23,925 people who
# named no religion. Four labels, two meanings each, and nothing else distinguishes them.
CHRISTIAN_PREFIX = "Christian: "

# UNSD Demographic Yearbook table 28, Eswatini 2017 — an INDEPENDENT transcription of these
# same tables, forwarded by the CSO and merged to twenty rows. `tools/oracle.py Eswatini`.
UNSD = {
    "Zion Christian Church": 367_290, "Evangelical": 248_233, "Pentecostal": 130_794,
    "No Religion": 80_861, "Apostolic": 53_400, "Nazarene": 44_112, "Methodist": 40_452,
    "Roman Catholic": 35_969, "Not Specified": 23_925, "Anglican": 15_364,
    "Other Christians": 13_471, "Jehovah Witness": 11_896, "Seventh Day Adventist": 8_783,
    "Lutheran": 5_993, "Traditional": 4_869, "Islam": 3_626, "Other": 3_363,
    "Baha'i": 430, "Hindu": 244, "Judaism": 163,
}
# UNSD's label -> how this volume prints it. `Other Christians` has no single counterpart:
# it is the Christian `Other` plus the Christian `Not Stated`, asserted separately.
UNSD_TO_VOLUME = {
    "Zion Christian Church": ("den", "Zionists"),
    "Evangelical": ("den", "Evangelical"),
    "Pentecostal": ("den", "Pentecostal"),
    "Apostolic": ("den", "Apostles"),
    "Nazarene": ("den", "Nazarene"),
    "Methodist": ("den", "Methodist"),
    "Roman Catholic": ("den", "Roman Catholic"),
    "Anglican": ("den", "Anglican"),
    "Jehovah Witness": ("den", "Jehovah Witness"),
    "Seventh Day Adventist": ("den", "Seventh Day Adventist"),
    "Lutheran": ("den", "Lutheran"),
    "No Religion": ("top", "No religion"),
    "Not Specified": ("top", "Not Stated"),
    "Traditional": ("top", "Traditionalist"),
    "Islam": ("top", "Islam"),
    "Other": ("top", "Other"),
    "Baha'i": ("top", "Bahai Faith"),
    "Hindu": ("top", "Hindu"),
    "Judaism": ("top", "Judaism"),
}

NUM = re.compile(r"^[\d,]+(?:\.\d+)?$")


def _key(s):
    """Fold runs of whitespace, including the tabs the CSO leaves in figure cells."""
    return " ".join(str(s).split())


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) > 2_000_000:
        print("already have", dest)
        return
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, timeout=900, stream=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    # §5a: HTTP 200 is not a download. Assert magic AND the trailer, because a PDF can be
    # truncated at source with a Content-Length that matches ([[reference_pdf_truncated_at_source]]).
    with open(dest, "rb") as fh:
        blob = fh.read()
    if blob[:5] != b"%PDF-":
        raise SystemExit(f"{dest} is not a PDF -- starts {blob[:5]!r}, {len(blob):,} bytes")
    if b"%%EOF" not in blob[-4096:]:
        raise SystemExit(f"{dest} has no %%EOF trailer -- truncated at source, {len(blob):,} bytes")
    print(f"  {len(blob):,} bytes")


def _num(tok, where):
    t = _key(tok)
    if not NUM.match(t):
        raise SystemExit(f"{where}: {tok!r} is not a figure -- Volume 3 has been re-typeset "
                         "and the row would silently shift")
    return float(t.replace(",", "")) if "." in t else int(t.replace(",", ""))


def _read_table(lines, title_re, rows, ncols, label):
    """One table off a page's flat line list. Label, then `ncols` figures, one per line.

    The text layer of Volume 3 emits every cell on its own line in reading order, so there
    is no geometry to do — Zimbabwe's shape rather than Malawi's rotated page or Benin's
    column bands. It is still not trusted: the title is matched first, and each row's label
    is asserted in print order before its figures are taken, so a re-ordered or renamed row
    stops the run instead of relabelling the map.
    """
    start = next((j for j in range(len(lines))
                  if title_re.search(_key(lines[j]))), None)
    if start is None:
        raise SystemExit(f"{label}: no line matching its title -- the CSO has re-typeset "
                         "Volume 3; find the table and update the page constants")
    # The header band varies per table (`Sex`/`Number 2017`/`Regions` spanners), so skip to
    # the first ROW LABEL rather than asserting a header shape that differs six ways.
    i = next((j for j in range(start + 1, len(lines))
              if _key(lines[j]) == _key(rows[0])), None)
    if i is None:
        raise SystemExit(f"{label}: never reached its first row {rows[0]!r} after the title")

    out = {}
    for want in rows:
        while i < len(lines) and not _key(lines[i]):
            i += 1
        got = _key(lines[i]) if i < len(lines) else "<eof>"
        if got != _key(want):
            raise SystemExit(f"{label}: expected the {want!r} row and read {got!r} -- the "
                             "row order or the category list has changed, and "
                             "taxonomy/sz2017.py must be revisited")
        i += 1
        vals = []
        while len(vals) < ncols and i < len(lines):
            txt = _key(lines[i])
            i += 1
            if not txt:
                continue
            vals.append(_num(txt, f"{label} {want!r}"))
        if len(vals) != ncols:
            raise SystemExit(f"{label} {want!r}: read {len(vals)} figures, expected {ncols}")
        out[want] = vals
    return out


def read():
    import fitz

    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)
    if doc.page_count < PAGE_522 + 1:
        raise SystemExit(f"{p} has {doc.page_count} pages, expected at least {PAGE_522 + 1}")

    def lines(pg):
        return doc[pg].get_text().splitlines()

    l321, l323, l324, l522 = (lines(PAGE_321), lines(PAGE_323),
                              lines(PAGE_324), lines(PAGE_522))

    t = {}
    # Male, Female, Total, Percent
    t["3.2.1"] = _read_table(l321, TITLES["3.2.1"], TOP_LEVEL + ["Total"], 4, "Table 3.2.1")
    t["3.2.2"] = _read_table(l321, TITLES["3.2.2"], DENOMINATIONS + ["Total"], 4,
                             "Table 3.2.2")
    # Urban n, Rural n, Urban %, Rural %
    t["3.2.3"] = _read_table(l323, TITLES["3.2.3"], DENOMINATIONS + ["Total"], 4,
                             "Table 3.2.3")
    # Hhohho, Manzini, Shiselweni, Lubombo, Total
    t["3.2.4"] = _read_table(l324, TITLES["3.2.4"], DENOMINATIONS + ["Total"], 5,
                             "Table 3.2.4")
    # the same four as percentages, no Total column
    t["3.2.5"] = _read_table(l324, TITLES["3.2.5"], DENOMINATIONS + ["Total"], 4,
                             "Table 3.2.5")
    # Total, Urban, Rural, urban-rural ratio
    t["5.2.2"] = _read_table(l522, TITLES["5.2.2"], ["Eswatini"] + REGIONS, 4, "Table 5.2.2")
    return t


def _allocate(residuals, shares):
    """Split each region's non-Christian residual across the eight national columns.

    Every region's non-Christian people are a COUNTED magnitude — its published population
    minus its published Christians — so this only decides which of the eight columns each
    one falls in, at the national rate. §14.4 rule 1: no magnitude is estimated.

    **BOTH MARGINS ARE HELD EXACTLY, and a per-region largest remainder does not do that.**
    Rounding each region on its own reproduces its residual and misses the national column
    by a person or two: `No religion` came out 80,860 against a published 80,861 and
    `Not Stated` 23,926 against 23,925. Nothing about the map changes at that size and the
    check does: a column that no longer equals the figure Table 3.2.1 prints cannot be
    asserted against it, and an assertion given up is a whole class of error let through.
    So this is a CONTROLLED ROUNDING over the 4x8 matrix — floor every cell, then hand out
    the shortfall to the largest fractional parts subject to BOTH the region totals and the
    national column totals, which leaves every published figure on both margins intact.
    """
    base = sum(shares.values())
    regions = list(residuals)
    cats = list(shares)
    exact = {(r, c): residuals[r] * shares[c] / base for r in regions for c in cats}
    out = {k: int(v) for k, v in exact.items()}

    # What each margin is still short of its published figure.
    short_r = {r: residuals[r] - sum(out[(r, c)] for c in cats) for r in regions}
    short_c = {c: shares[c] - sum(out[(r, c)] for r in regions) for c in cats}
    if sum(short_r.values()) != sum(short_c.values()):
        raise SystemExit("the two margins disagree before rounding, which means the "
                         "residuals do not sum to the national non-Christian total")

    order = sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))
    for _ in range(len(order) + 1):
        if not any(short_r.values()):
            break
        for r, c in order:
            if short_r[r] > 0 and short_c[c] > 0:
                out[(r, c)] += 1
                short_r[r] -= 1
                short_c[c] -= 1
    if any(short_r.values()) or any(short_c.values()):
        raise SystemExit(f"controlled rounding did not close: rows {short_r}, cols {short_c}")
    for r in regions:
        if sum(out[(r, c)] for c in cats) != residuals[r]:
            raise SystemExit(f"{r}: allocation does not sum to its residual")
    for c in cats:
        if sum(out[(r, c)] for r in regions) != shares[c]:
            raise SystemExit(f"{c}: allocation does not sum to its national figure")
    return out


def build(t):
    """The drawn rows: 13 Christian denominations measured, 8 residual columns derived."""
    reg_pop = {r: int(t["5.2.2"][r][0]) for r in REGIONS}
    reg_chr = {r: int(t["3.2.4"]["Total"][i]) for i, r in enumerate(REGIONS)}
    nat_non = {c: int(t["3.2.1"][c][2]) for c in NON_CHRISTIAN}

    residuals = {}
    for i, region in enumerate(REGIONS):
        residuals[region] = reg_pop[region] - reg_chr[region]
        if residuals[region] < 0:
            raise SystemExit(f"{region}: more Christians than people")
    if sum(residuals.values()) != sum(nat_non.values()):
        raise SystemExit(f"the four residuals sum to {sum(residuals.values()):,} and Table "
                         f"3.2.1's eight non-Christian cells to {sum(nat_non.values()):,}")
    alloc = _allocate(residuals, nat_non)

    rows = []
    for i, region in enumerate(REGIONS, start=1):
        gid = f"SZ{i:02d}"
        for den in DENOMINATIONS:
            rows.append({
                "geo_id": gid, "geo_level": "region", "geo_name": region,
                "source_category": CHRISTIAN_PREFIX + den,
                "count": int(t["3.2.4"][den][i - 1]),
                "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                "note": "level=region; Table 3.2.4, Christians by denomination and region"})

        for cat in NON_CHRISTIAN:
            rows.append({
                "geo_id": gid, "geo_level": "region", "geo_name": region,
                "source_category": cat, "count": alloc[(region, cat)],
                "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                "note": ("level=region; tier=derived; the region's non-Christian total is "
                         "its Table 5.2.2 population minus its Table 3.2.4 Christians, "
                         "split at the national rates of Table 3.2.1; "
                         "parent_column=non-Christian residual")})

        rows.append({
            "geo_id": gid, "geo_level": "region", "geo_name": region,
            "source_category": "Total", "count": reg_pop[region],
            "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
            "note": "level=region; universe total, not a religion category; Table 5.2.2"})
    return rows


def check(t, rows):
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    # ---- Table 3.2.1, the national religion table ----
    bad = [c for c in TOP_LEVEL + ["Total"]
           if t["3.2.1"][c][0] + t["3.2.1"][c][1] != t["3.2.1"][c][2]]
    say(not bad, f"3.2.1  Male + Female == Total on all {len(TOP_LEVEL) + 1} rows {bad[:3]}")
    s = sum(t["3.2.1"][c][2] for c in TOP_LEVEL)
    say(s == NATIONAL == t["3.2.1"]["Total"][2],
        f"3.2.1  the 9 religions sum to {s:,} == the census {NATIONAL:,}")
    bad = [(c, t["3.2.1"][c][3], round(100.0 * t["3.2.1"][c][2] / NATIONAL, 2))
           for c in TOP_LEVEL
           if abs(t["3.2.1"][c][3] - 100.0 * t["3.2.1"][c][2] / NATIONAL) > 0.006]
    say(not bad, f"3.2.1  the printed percent reproduces count/{NATIONAL:,} on all "
                 f"{len(TOP_LEVEL)} rows {bad[:3]}")

    # ---- Table 3.2.2, the national Christian denominations ----
    bad = [d for d in DENOMINATIONS + ["Total"]
           if t["3.2.2"][d][0] + t["3.2.2"][d][1] != t["3.2.2"][d][2]]
    say(not bad, f"3.2.2  Male + Female == Total on all {len(DENOMINATIONS) + 1} rows {bad[:3]}")
    s = sum(t["3.2.2"][d][2] for d in DENOMINATIONS)
    say(s == CHRISTIANS == t["3.2.1"]["Christian"][2],
        f"3.2.2  the 13 denominations sum to {s:,} == 3.2.1's Christian cell "
        f"{t['3.2.1']['Christian'][2]:,}  <- CROSS-TABLE")

    # ---- Table 3.2.3, urban/rural. Cross-table and cross-page. ----
    bad = [(d, t["3.2.3"][d][0] + t["3.2.3"][d][1], t["3.2.2"][d][2])
           for d in DENOMINATIONS + ["Total"]
           if t["3.2.3"][d][0] + t["3.2.3"][d][1] != t["3.2.2"][d][2]]
    say(not bad, f"3.2.3  urban + rural == 3.2.2's Total on all {len(DENOMINATIONS) + 1} "
                 f"rows {bad[:3]}  <- CROSS-TABLE")

    # ---- Table 3.2.4, the drawn one ----
    bad = [(d, sum(t["3.2.4"][d][:4]), t["3.2.4"][d][4]) for d in DENOMINATIONS + ["Total"]
           if sum(t["3.2.4"][d][:4]) != t["3.2.4"][d][4]]
    say(not bad, f"3.2.4  the 4 regions sum to its own Total column on all "
                 f"{len(DENOMINATIONS) + 1} rows {bad[:3]}")
    bad = [(d, t["3.2.4"][d][4], t["3.2.2"][d][2]) for d in DENOMINATIONS + ["Total"]
           if t["3.2.4"][d][4] != t["3.2.2"][d][2]]
    say(not bad, f"3.2.4  its Total column == 3.2.2's Total column on all "
                 f"{len(DENOMINATIONS) + 1} rows {bad[:3]}  <- CROSS-TABLE")

    # ---- Table 3.2.5, the per-cell check (Cambodia's) ----
    bad = []
    for d in DENOMINATIONS:
        for i, r in enumerate(REGIONS):
            want = 100.0 * t["3.2.4"][d][i] / t["3.2.4"]["Total"][i]
            if abs(t["3.2.5"][d][i] - want) > 0.006:
                bad.append((d, r, t["3.2.5"][d][i], round(want, 3)))
    n_cells = len(DENOMINATIONS) * len(REGIONS)
    say(not bad, f"3.2.5  its printed percent reproduces 3.2.4 / the region's Christians on "
                 f"all {n_cells} cells ({len(bad)} off) {bad[:3]}")

    # ---- Table 5.2.2, the region populations ----
    bad = [r for r in ["Eswatini"] + REGIONS
           if t["5.2.2"][r][1] + t["5.2.2"][r][2] != t["5.2.2"][r][0]]
    say(not bad, f"5.2.2  urban + rural == total on all {len(REGIONS) + 1} rows {bad[:3]}")
    s = sum(int(t["5.2.2"][r][0]) for r in REGIONS)
    say(s == NATIONAL == int(t["5.2.2"]["Eswatini"][0]),
        f"5.2.2  the 4 regions sum to {s:,} == the census {NATIONAL:,}")

    # ---- THE PERMUTATION TEST, which is the only check a consistent column swap fails ----
    # Every identity inside 3.2.4 holds whichever order its four region columns were read
    # in. This one does not, because it pairs 3.2.4 (chapter 3) against 5.2.2 (chapter 5):
    # a region's Christians over its population must land near the national 89.25%.
    import itertools
    nat_share = 100.0 * CHRISTIANS / NATIONAL
    pops = [int(t["5.2.2"][r][0]) for r in REGIONS]
    chrs = [int(t["3.2.4"]["Total"][i]) for i in range(len(REGIONS))]
    BAND = 2.0
    shares = [100.0 * c / p for c, p in zip(chrs, pops)]
    say(all(abs(s - nat_share) <= BAND for s in shares),
        "3.2.4 x 5.2.2  every region's Christian share is within "
        f"{BAND:g} points of the national {nat_share:.2f}%: "
        + ", ".join(f"{r} {s:.2f}%" for r, s in zip(REGIONS, shares)))
    survivors = [p for p in itertools.permutations(range(len(REGIONS)))
                 if all(abs(100.0 * chrs[j] / pops[i] - nat_share) <= BAND
                        for i, j in enumerate(p))]
    say(len(survivors) == 1,
        f"3.2.4 x 5.2.2  and exactly {len(survivors)} of "
        f"{len(list(itertools.permutations(range(len(REGIONS)))))} column orderings "
        "survives that band, so the region columns cannot be silently transposed")

    # ---- the UNSD oracle, an independent transcription (§0.5) ----
    bad = []
    for label, want in UNSD.items():
        if label == "Other Christians":
            got = t["3.2.2"]["Other"][2] + t["3.2.2"]["Not Stated"][2]
        else:
            which, row = UNSD_TO_VOLUME[label]
            got = t["3.2.2" if which == "den" else "3.2.1"][row][2]
        if got != want:
            bad.append((label, got, want))
    say(not bad, f"UNSD   all {len(UNSD)} Demographic Yearbook figures reproduce this "
                 f"volume exactly {bad[:3]}  <- INDEPENDENT TRANSCRIPTION")
    say(sum(UNSD.values()) == NATIONAL,
        f"UNSD   and its 20 categories partition {sum(UNSD.values()):,} == {NATIONAL:,}")

    # ---- the built rows ----
    drawn = [r for r in rows if r["source_category"] != "Total"]
    per_region = {}
    for r in drawn:
        per_region[r["geo_name"]] = per_region.get(r["geo_name"], 0) + r["count"]
    bad = [(r, per_region[r], int(t["5.2.2"][r][0])) for r in REGIONS
           if per_region[r] != int(t["5.2.2"][r][0])]
    say(not bad, f"built  each region's 21 drawn rows sum to its census population {bad[:3]}")
    say(sum(per_region.values()) == NATIONAL,
        f"built  and to {sum(per_region.values()):,} == the census {NATIONAL:,}, so "
        "100% of Eswatini is placed")

    # The derived rows must reproduce the national non-Christian totals exactly.
    bad = []
    for cat in NON_CHRISTIAN:
        s = sum(r["count"] for r in drawn if r["source_category"] == cat)
        if s != t["3.2.1"][cat][2]:
            bad.append((cat, s, t["3.2.1"][cat][2]))
    say(not bad, f"built  the 8 derived columns sum back to Table 3.2.1's own national "
                 f"figures {bad[:3]}")

    derived = sum(r["count"] for r in drawn if "tier=derived" in r["note"])
    print(f"\n  {len(rows):,} rows. {derived:,} people ({100.0 * derived / NATIONAL:.2f}%) "
          f"are `derived`; the other {100.0 * (NATIONAL - derived) / NATIONAL:.2f}% is "
          "Table 3.2.4 read at the region it was printed for.")
    print("\n  national, as drawn:")
    tot = {}
    for r in drawn:
        tot[r["source_category"]] = tot.get(r["source_category"], 0) + r["count"]
    for cat, n in sorted(tot.items(), key=lambda kv: -kv[1]):
        mark = "  derived" if cat in NON_CHRISTIAN else ""
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {cat}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    t = read()
    rows = build(t)
    check(t, rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
