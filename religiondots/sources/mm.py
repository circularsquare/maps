"""Myanmar — Department of Population, 2014 Census Report Volume 2-C (Religion), Table 1.

Reads (or fetches) data/raw/mm/ and writes data/normalized/mm.csv.

**Seven religion categories plus an eighth column that is the point of the country: an
estimated non-enumerated population of 1,206,353, of which Rakhine is exactly 1,090,000.**

**THE ENTIRE PUBLISHED RELIGION OUTPUT OF THE 2014 CENSUS IS TWO TABLES IN A 17-PAGE
REPORT.** Table 1 is Union plus 15 States/Regions; Table 2 is a 1973/1983/2014 time series.
There is no district or township religion table anywhere — not withheld at a finer tier, not
published and hard to find: it does not exist. So 15 units is not §14's ceiling being applied,
it is all there is, and §3.9b is what makes it drawable.

**THE NON-ENUMERATED ARE DRAWN AS THEIR OWN CATEGORY — Anita's call, 2026-09-07.** The
report says who they are and why, in its own words:

    "In Rakhine, an estimated 1.09 million people were not enumerated in the Census because
     they were not allowed to self-identify using a name not recognized by the Government.
     It is assumed that the non-enumerated population in Rakhine is mainly affiliated with
     the Islamic faith."

**What is NOT done here is applying that assumption.** DOP applies it only at the Union level
(Figure 3 and Table 2's second 2014 column: Islam 2.3% → 4.3%), and it publishes no religious
breakdown of the non-enumerated at any geography. Assigning them to `islam` would be inventing
a magnitude the source does not publish at the level it is drawn — §14 rule 1 — so they go to
`unenumerated`, a node that says only that these people were not counted. The state's own
assumption is quoted in `note_public` so a reader is told what DOP itself concluded.

**Drawing them at all is the whole reason this is worth doing.** Enumerated Rakhine is
2,098,807 people of whom 28,731 are Muslim, so a map built from the enumerated columns alone
renders Rakhine **96.2% Buddhist** — which is not a modelling slip but the census's own
exclusion reproduced as fact, and §14.2's second risk exactly. With the column drawn, Rakhine
is 34% not-enumerated and the map says so.

**THE PARSE HAS TWO INDEPENDENT CHECKS AND IT NEEDED THEM**, because every identity inside
Table 1 — the seven religions summing to each Total, the fifteen rows summing to the Union —
survives a consistent column permutation (§12's Zimbabwe warning):

  * **THE PRINTED PERCENTAGES ARE A PER-CELL CHECK.** DOP prints a `%` row under every
    `Number` row, so `count / total` must reproduce it to one decimal place on all 105 cells.
    A swapped pair of religion columns fails this immediately, and no within-table sum does.
  * **MIMU PUBLISHED AN INDEPENDENT TRANSCRIPTION** of the same table, p-coded
    (`BaselineData_Census_Religion_Union_Pcode_MIMU_21Jul2016.xlsx`). Every cell is compared
    and must agree exactly. Two organisations reading the same page is the strongest check
    available here, and it is also where the p-codes come from.

**AND THE STATE ROWS ARE PAIRED TO MIMU'S BY THEIR NUMBERS, NOT BY NAME OR POSITION.** Each
row's eight figures are matched against MIMU's, 1:1 both ways — a pairing the data itself
proves. Names then only have to be *reported*, which is just as well: DOP writes
**`Ayeyawady`** and MIMU and OCHA write **`Ayeyarwady`**.

Usage:
    python sources/mm.py --fetch    one 3.6 MB PDF and one 18 KB xlsx, seconds
    python sources/mm.py            normalise from data/raw/mm/
"""

import csv
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "mm")
OUT = os.path.join(ROOT, "data", "normalized", "mm.csv")

SOURCE_ID = "mm_census_2014_v2c"
YEAR = 2014
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

PDF_URL = ("https://myanmar.unfpa.org/sites/default/files/pub-pdf/"
           "UNION_2C_Religion_EN.pdf")
PDF_NAME = "mm_union_2c_religion_en.pdf"

# MIMU's p-coded transcription of the same Table 1 — the cross-source check AND the p-codes.
XLSX_URL = ("https://themimu.info/sites/themimu.info/files/documents/"
            "BaselineData_Census_Religion_Union_Pcode_MIMU_21Jul2016.xlsx")
XLSX_NAME = "mm_religion_pcode_mimu.xlsx"

TITLE = re.compile(r"Table\s*1\s*Number and percentage of persons by religion and "
                   r"State/Region", re.I)

# In the order DOP prints them, left to right.
CATEGORIES = ["Buddhist", "Christian", "Islam", "Hindu", "Animist", "Other religion",
              "No religion"]
TOTAL_CAT = "Total"
NONENUM_CAT = "Estimated Non-enumerated population"

# In the order DOP prints them, which is also MIMU's p-code order — asserted, not assumed.
STATES = [
    "Kachin", "Kayah", "Kayin", "Chin", "Sagaing", "Tanintharyi", "Bago", "Magway",
    "Mandalay", "Mon", "Rakhine", "Yangon", "Shan", "Ayeyawady", "Nay Pyi Taw",
]
EXPECTED_STATES = 15

ENUMERATED = 50_279_900        # Table 1's own Union `Total`
NON_ENUMERATED = 1_206_353     # Table 1's own Union non-enumerated cell
UNIVERSE = 51_486_253          # the report's own figure for the two together (page 4)

# **DOP IS INCONSISTENT ABOUT ROUNDING VS TRUNCATING ITS PERCENTAGES**, measured over all 110
# cells of Table 1: 53 where the two agree, 51 that match rounding only, 5 that match
# truncation only (Union/Buddhist 89.8678 -> 89.8, Sagaing/Christian 6.5606 -> 6.5, Bago/Hindu
# 2.0579 -> 2.0, Shan/Hindu 0.093 -> 0.0, Ayeyawady/Buddhist 92.1556 -> 92.1). So the check
# accepts EITHER, which is a rule rather than a tolerance chosen to pass — and it still leaves
# the check enormous margin, because a swapped pair of religion columns moves a percentage by
# whole points (Kachin's Christian 33.8 against its Islam 1.6), not by hundredths.
#
# **AND ONE CELL IS NEITHER, WHICH IS AN ERROR IN THE PUBLISHED REPORT.** Kachin's Hindu count
# is 5,738 of 1,642,841 = 0.3493%, and DOP prints **0.4**. It is not a parse artefact: the
# seven Kachin counts sum to its Total exactly, MIMU's independent transcription carries the
# same 5,738 and the same 0.4, and no denominator in the table yields 0.4 (including the total
# with the non-enumerated added back). The COUNT is what this map draws, so nothing downstream
# is affected. It is listed rather than tolerated, so that a SECOND such cell fails the build.
PCT_EXCEPTIONS = {
    ("Kachin", "Hindu"): 0.4,
}

NUM = re.compile(r"^[\d,]+$")
PCT = re.compile(r"^\d+(?:\.\d+)?$")


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    for url, name, magic, floor in (
            (PDF_URL, PDF_NAME, b"%PDF-", 1_000_000),
            (XLSX_URL, XLSX_NAME, b"PK\x03\x04", 5_000)):
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > floor:
            print("already have", dest)
            continue
        print("GET", url)
        r = requests.get(url, timeout=900, stream=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        # §5a: HTTP 200 is not a download. Assert size AND type.
        with open(dest, "rb") as fh:
            got = fh.read(len(magic))
        if got != magic:
            raise SystemExit(f"{dest} starts {got!r}, expected {magic!r} -- "
                             f"{os.path.getsize(dest):,} bytes")
        if os.path.getsize(dest) < floor:
            raise SystemExit(f"{dest} is {os.path.getsize(dest):,} bytes -- truncated")
        print(f"  {os.path.getsize(dest):,} bytes")


def _cell(tok, where):
    if not NUM.match(tok):
        raise SystemExit(f"{where}: {tok!r} is not a figure -- Table 1 has been re-typeset")
    return int(tok.replace(",", ""))


def read_pdf():
    """Table 1 -> {row label: (numbers, percentages)}.

    The text layer emits the row label, the literal `Number`, that row's figures one per
    line, the literal `%`, and then its percentages. **The non-enumerated column is BLANK for
    the eleven states that have none**, so a Number row is 8 or 9 figures and the count is
    what says which — never assume nine.
    """
    import fitz

    p = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    doc = fitz.open(p)

    # The caption appears twice — once in the List of Tables on page 10 and once over the
    # table itself. Discriminate on the BODY: the real page carries a standalone `Number`
    # line and every state name.
    hits = []
    for n in range(doc.page_count):
        text = doc[n].get_text()
        if not TITLE.search(" ".join(text.split())):
            continue
        body = [ln.strip() for ln in text.splitlines()]
        if "Number" in body and all(s in body for s in STATES):
            hits.append(n)
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one page carrying Table 1's body, found "
                         f"{hits} -- DOP has re-typeset or re-paginated Volume 2-C")
    pno = hits[0]
    lines = [ln.strip() for ln in doc[pno].get_text().splitlines()]
    doc.close()

    head = next(j for j in range(len(lines))
                if TITLE.search(" ".join(" ".join(lines[j:j + 3]).split())))

    # The header block wraps `Other religion` and `No religion` over two lines each, so it is
    # asserted as an ordered sequence inside the joined text rather than line by line.
    start = next(j for j in range(head, len(lines)) if lines[j] == "Union")
    header = " ".join(" ".join(lines[head:start]).split())
    cursor = 0
    for want in ["State/Region", "Total", "Religion"] + CATEGORIES:
        k = header.find(want, cursor)
        if k < 0:
            raise SystemExit(f"Table 1's header does not carry {want!r} in order -- "
                             f"read {header[:200]!r}. The column list has changed and "
                             "taxonomy/mm2014.py must be revisited.")
        cursor = k + len(want)

    out, i = {}, start
    for label in ["Union"] + STATES:
        while i < len(lines) and not lines[i]:
            i += 1
        got = lines[i] if i < len(lines) else "<eof>"
        if got != label:
            raise SystemExit(f"expected the {label!r} row and read {got!r} -- Table 1's "
                             "row order has changed")
        i += 1
        while i < len(lines) and lines[i] != "Number":
            if lines[i]:
                raise SystemExit(f"{label}: expected `Number` and read {lines[i]!r}")
            i += 1
        i += 1
        nums = []
        while i < len(lines) and lines[i] != "%":
            if lines[i]:
                nums.append(_cell(lines[i], f"{label} numbers"))
            i += 1
        i += 1
        pcts = []
        while i < len(lines) and (not lines[i] or PCT.match(lines[i]) or lines[i] == "-"):
            if lines[i]:
                pcts.append(None if lines[i] == "-" else float(lines[i]))
            i += 1
        if len(nums) not in (1 + len(CATEGORIES), 2 + len(CATEGORIES)):
            raise SystemExit(f"{label}: read {len(nums)} figures, expected 8 or 9")
        if len(pcts) != len(CATEGORIES):
            raise SystemExit(f"{label}: read {len(pcts)} percentages, expected "
                             f"{len(CATEGORIES)}")
        out[label] = (nums, pcts)
    return out, pno


def read_mimu():
    """MIMU's p-coded transcription -> {label: (pcode, numbers)} in file order."""
    import openpyxl

    p = os.path.join(RAW, XLSX_NAME)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run with --fetch first")
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    if "SR" not in wb.sheetnames:
        raise SystemExit(f"no `SR` sheet in {p}: {wb.sheetnames}")
    ws = wb["SR"]

    rows, order = {}, []
    for r in ws.iter_rows(values_only=True):
        if r[2] != "Number" or not r[1]:
            continue
        name = str(r[1]).strip()
        nums = [int(v) for v in r[3:11]]                      # total + 7 religions
        nonenum = r[11]
        if nonenum not in (None, ""):
            nums.append(int(nonenum))
        rows[name] = (str(r[0]).strip(), nums)
        order.append(name)
    if len(rows) != EXPECTED_STATES + 1:
        raise SystemExit(f"MIMU sheet has {len(rows)} Number rows, expected "
                         f"{EXPECTED_STATES + 1}")
    return rows, order


def build(pdf, mimu, mimu_order):
    """Pair DOP's rows to MIMU's BY THEIR NUMBERS, 1:1 both ways."""
    by_nums = {}
    for name, (pcode, nums) in mimu.items():
        key = tuple(nums)
        if key in by_nums:
            raise SystemExit(f"MIMU rows {by_nums[key][0]!r} and {name!r} have identical "
                             "figures -- they cannot be told apart by value")
        by_nums[key] = (name, pcode)

    pairs, unmatched = {}, []
    for label in ["Union"] + STATES:
        key = tuple(pdf[label][0])
        if key in by_nums:
            pairs[label] = by_nums[key]
        else:
            unmatched.append(label)
    return pairs, unmatched, by_nums


def check(pdf, mimu, mimu_order, pairs, unmatched, pno):
    ok = True
    print(f"  Table 1 on page {pno + 1}: {len(pdf)} rows "
          f"({EXPECTED_STATES} states + Union)")

    good = pdf["Union"][0][0] == ENUMERATED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} enumerated universe {pdf['Union'][0][0]:,} "
          f"(expected {ENUMERATED:,})")

    good = pdf["Union"][0][8] == NON_ENUMERATED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} non-enumerated       {pdf['Union'][0][8]:,} "
          f"(expected {NON_ENUMERATED:,})")

    good = ENUMERATED + NON_ENUMERATED == UNIVERSE
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the two sum to the report's own overall figure "
          f"{UNIVERSE:,} (page 4)")

    bad = [k for k in ["Union"] + STATES
           if sum(pdf[k][0][1:8]) != pdf[k][0][0]]
    ok &= not bad
    print(f"  {'OK ' if not bad else 'BAD'} the 7 religions sum to Total on all "
          f"{len(STATES) + 1} rows ({len(bad)} failures) {bad[:4]}")

    for j, cat in enumerate([TOTAL_CAT] + CATEGORIES):
        s = sum(pdf[k][0][j] for k in STATES)
        if s != pdf["Union"][0][j]:
            ok = False
            print(f"  BAD the 15 states sum to the Union row on {cat}: {s:,} vs "
                  f"{pdf['Union'][0][j]:,}")
    s = sum(pdf[k][0][8] for k in STATES if len(pdf[k][0]) > 8)
    good = s == NON_ENUMERATED
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 15 states sum to the Union row on all "
          f"{len(CATEGORIES) + 2} columns (non-enumerated {s:,})")

    # --- the per-cell check a consistent column permutation cannot survive ---
    import math

    bad, exercised = [], set()
    for k in ["Union"] + STATES:
        nums, pcts = pdf[k]
        for j, cat in enumerate(CATEGORIES):
            if pcts[j] is None:
                continue
            got = 100.0 * nums[1 + j] / nums[0]
            allowed = (round(got, 1), math.floor(got * 10) / 10)
            if any(abs(pcts[j] - a) < 1e-9 for a in allowed):
                continue
            if PCT_EXCEPTIONS.get((k, cat)) == pcts[j]:
                exercised.add((k, cat))
                continue
            bad.append((k, cat, round(got, 4), pcts[j]))
    ok &= not bad
    cells = sum(1 for k in ["Union"] + STATES for v in pdf[k][1] if v is not None)
    print(f"  {'OK ' if not bad else 'BAD'} count/total reproduces DOP's own printed "
          f"percentage — rounded OR truncated — on {cells - len(PCT_EXCEPTIONS)} of "
          f"{cells} cells ({len(bad)} unexplained)")
    for k, c, g, w in bad[:5]:
        print(f"        {k}/{c}: computed {g} vs printed {w}")
    print("      THIS is the check a consistent column permutation cannot survive; every "
          "sum above\n      would still hold if two religion columns had been swapped "
          "throughout (§12).")

    stale = set(PCT_EXCEPTIONS) - exercised
    ok &= not stale
    print(f"  {'OK ' if not stale else 'BAD'} the {len(PCT_EXCEPTIONS)} known DOP "
          f"percentage error is still present ({sorted(exercised)})")
    if stale:
        print(f"        no longer failing: {sorted(stale)} -- DOP has corrected the file, "
              "so remove it from PCT_EXCEPTIONS")
    print("      Kachin's Hindu cell is 5,738/1,642,841 = 0.3493% and DOP prints 0.4. The "
          "COUNT is\n      what is drawn, so nothing on the map is affected; it is listed "
          "so a SECOND one fails.")

    # --- the cross-source check ---
    ok &= not unmatched
    print(f"\n  {'OK ' if not unmatched else 'BAD'} every DOP row pairs to a MIMU row by "
          f"its FIGURES, 1:1 ({len(unmatched)} unmatched) {unmatched}")
    spare = sorted(set(mimu) - {v[0] for v in pairs.values()})
    ok &= not spare
    print(f"  {'OK ' if not spare else 'BAD'} MIMU rows with no DOP row: {spare}")
    print("      MIMU transcribed the same table independently and p-coded it. Matching on "
          "the\n      NUMBERS rather than on names or position means the data proves its "
          "own pairing.")

    renamed = [(k, pairs[k][0]) for k in pairs if k != pairs[k][0]]
    print(f"\n    {len(pairs) - len(renamed)}/{len(pairs)} names agree; {len(renamed)} "
          "differ by romanisation:")
    for a, b in renamed:
        print(f"      DOP {a!r:<16} MIMU/OCHA {b!r}")

    # --- print order against MIMU's own file order ---
    want = [pairs[k][0] for k in ["Union"] + STATES]
    good = want == mimu_order
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} DOP's printed row order reproduces MIMU's file "
          "order on all 16 rows")
    if not good:
        print(f"        DOP  {want}")
        print(f"        MIMU {mimu_order}")

    print(f"\n  Categories, national (enumerated {ENUMERATED:,}):")
    for j, cat in enumerate(CATEGORIES):
        n = pdf["Union"][0][1 + j]
        print(f"    {n:>11,}  {100.0 * n / ENUMERATED:6.2f}%  {cat}")
    print(f"    {NON_ENUMERATED:>11,}  {100.0 * NON_ENUMERATED / UNIVERSE:6.2f}%  "
          f"{NONENUM_CAT}  <- of the {UNIVERSE:,} overall")

    print(f"\n  Non-enumerated by state, and it is why the country is worth drawing:")
    for k in STATES:
        if len(pdf[k][0]) > 8 and pdf[k][0][8]:
            tot = pdf[k][0][0] + pdf[k][0][8]
            print(f"    {k:<14} {pdf[k][0][8]:>9,}  {100.0 * pdf[k][0][8] / tot:5.1f}% of "
                  f"that state's {tot:,} people")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def rows_for(pdf, pairs):
    rows = []
    for idx, state in enumerate(STATES, start=1):
        nums = pdf[state][0]
        gid = f"MM-{idx:02d}"
        base = "level=state_region"
        for j, cat in enumerate(CATEGORIES):
            rows.append({"geo_id": gid, "geo_level": "state_region", "geo_name": state,
                         "source_category": cat, "count": nums[1 + j], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID, "note": base})
        rows.append({"geo_id": gid, "geo_level": "state_region", "geo_name": state,
                     "source_category": TOTAL_CAT, "count": nums[0], "basis": BASIS,
                     "year": YEAR, "source_id": SOURCE_ID,
                     "note": base + "; ENUMERATED total, not the universe -- the "
                                    "non-enumerated are outside it"})
        if len(nums) > 8 and nums[8]:
            rows.append({"geo_id": gid, "geo_level": "state_region", "geo_name": state,
                         "source_category": NONENUM_CAT, "count": nums[8], "basis": BASIS,
                         "year": YEAR, "source_id": SOURCE_ID,
                         "note": base + "; DOP's own estimate of people the census did not "
                                        "enumerate; no religion is published for them"})
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    pdf, pno = read_pdf()
    mimu, mimu_order = read_mimu()
    pairs, unmatched, _ = build(pdf, mimu, mimu_order)
    check(pdf, mimu, mimu_order, pairs, unmatched, pno)

    rows = rows_for(pdf, pairs)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")

    # the p-code bridge, for sources/mm_geo.py
    lut = os.path.join(ROOT, "data", "normalized", "mm_pcodes.csv")
    with open(lut, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_name", "mimu_name", "pcode"])
        for idx, state in enumerate(STATES, start=1):
            w.writerow([f"MM-{idx:02d}", state, pairs[state][0], pairs[state][1]])
    print("wrote", lut, f"({EXPECTED_STATES} rows)")


if __name__ == "__main__":
    main()
