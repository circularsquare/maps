"""American Samoa - 2015 Household Income and Expenditure Survey, religion by county.

Reads (or fetches) the survey report into data/normalized/as.csv. `sources/as.md` is the write-up;
`sources/as_geo.py` builds the ten counties, the 2020 census count per county, and the placement
grid. `taxonomy/as2015.py` holds the mapping and the pinned verdict of the county test below.

## WHY A SURVEY

The census cannot ask. The 2010 American Samoa summary file documentation says "The Census Bureau
cannot collect information on religion", and the 2020 Island Areas forms have no religion item
(sources.md §scout-2026-09-14-asia-oceania). UNSD table 28 has no American Samoa row.

## THE TABLE

American Samoa Department of Commerce, Statistics Division, *2015 American Samoa Household Income
and Expenditure Survey Report*, section 11, **Table 1.6, "Religion by County, American Samoa:
2015"**, printed p.57 (PDF p.58), published on doi.gov. Weighted persons for the territory and ten
counties: the nine counties of Tutuila and Aunu'u, and Manu'a as one column (its five counties).
Sixteen rows, from a write-in answer for every household member (printed p.23).

## THE SAMPLE (printed pp.17-18)

A systematic 20 percent sample of housing units on all islands, 2000 census geography; 1,838 of
2,098 selected housing units completed a form. ONE weight, 5.99668, "the average of the sample
coverage of the three districts", applied to housing and population alike. So every cell is
round(k x 5.99668) for a whole number k of sampled persons, 9,578 in all, and the county totals are
the sample's completion by county times one number, NOT populations: against the 2010 census they
run 0.83x (Saole) to 1.37x (Lealataua). countries/as.py lays the shares on the 2020 census instead.

## THE CHECKS

    the PDF pinned; 189 pages, a text layer on every page
    the header's split column names rebuild the ten counties in order
    Table 1.6 parsed off the page equals the transcription below
    every cell within half a person of a whole multiple of the weight
    the sixteen rows sum to each printed county total within 8 (half a person per cell), and the
        ten counties to each row's total within 5
    Table 1.1 (age by county) prints the same county totals
    Table A (printed p.19) prints CCCAS, Catholic and LDS as county percents; each equals Table
        1.6's count over its total to the printed decimal
    Tables 2.3 (by age), 3.3 (by birthplace) and 4.3 (by citizenship) print the same national
        column; 4.3's males and females sum to it within 1 and its `NR` rows are 0
    the county test's verdict equals taxonomy/as2015.py::OWN_GEOGRAPHY

Usage:
    python sources/as.py --fetch    the report (6.9 MB) from doi.gov
    python sources/as.py            normalise from data/raw/as/
"""

import csv
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "as")
OUT = os.path.join(ROOT, "data", "normalized", "as.csv")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))

from fetch_checks import FetchCheckError, check_body, check_pdf_doc, digest   # noqa: E402

SOURCE_ID = "as_hies2015_t1_6"
YEAR = 2015
BASIS = "self_id"
COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

URL = ("https://www.doi.gov/sites/default/files/uploads/"
       "american-samoa-2015-household-income-and-expenditure-report.pdf")
PDF = os.path.join(RAW, "as_hies2015_report.pdf")
PDF_SIZE = 6_856_994
PDF_DIGEST = "K3PPMVA4Z24KN3ZPHPTQRFYCFTEAYLWS"
PDF_PAGES = 189

# 0-based page indices and the caption each must carry
PAGE_METHOD = 17        # printed p.17: the sample and its weight
PAGE_WRITEIN = 23       # printed p.23: "The survey used a write-in entry for religion"
PAGE_A = 19             # printed p.19: Table A
PAGE_11 = 55            # printed p.55: Table 1.1
PAGE_16 = 57            # printed p.57: Table 1.6
PAGE_23 = 68            # Table 2.3
PAGE_33 = 80            # Table 3.3
PAGE_43 = 93            # Table 4.3
CAP_A = "Table A. Selected Population Characteristics by County"
CAP_11 = "Table 1.1. Age by County"
CAP_16 = "Table 1.6. Religion by County, American Samoa: 2015"
CAP_23 = "Table 2.3. Religion by Age"
CAP_33 = "Table 3.3. Religion by Birthplace"
CAP_43 = "Table 4.3. Religion by Citizenship"

WEIGHT = 5.99668
HOUSEHOLDS = 1_838
PERSONS = 9_578

COUNTIES = ["Ituau", "Maoputasi", "Saole", "Sua", "Vaifanua", "Lealataua", "Leasina", "Tualatai",
            "Tualauta", "Manu'a"]
CATS = ["CCCAS", "Catholic", "Methodist", "SDA", "LDS _ Mormons", "Assembly of God", "Baha’i",
        "Full Gospel", "Jehovah's Witness", "Orthodox", "Jewish", "Pentecostal", "Nazarene",
        "Baptist", "Other religion", "No religion"]
# the same rows as the cross-tabulations spell them
ALIAS = {"Bahai": "Baha’i", "Jehovah's Witnes": "Jehovah's Witness"}

# Table 1.6, transcribed from printed p.57: territory total, then COUNTIES in order.
T16_TOTAL = (57_436, 5_607, 11_052, 1_811, 3_274, 2_489, 6_968, 1_541, 3_892, 19_519, 1_283)
T16 = {
    "CCCAS":             (19_147, 1_751, 2_818, 864, 1_331, 1_163, 2_501, 858, 1_811, 4_917, 1_133),
    "Catholic":          (10_410, 1_115, 2_860, 102, 480, 252, 2_123, 276, 678, 2_525, 0),
    "Methodist":         (4_342, 432, 624, 84, 228, 114, 390, 102, 162, 2_141, 66),
    "SDA":               (1_661, 192, 198, 0, 42, 84, 414, 30, 234, 468, 0),
    "LDS _ Mormons":     (9_091, 846, 1_091, 324, 276, 426, 774, 144, 336, 4_875, 0),
    "Assembly of God":   (5_451, 534, 1_139, 318, 384, 306, 366, 96, 462, 1_805, 42),
    "Baha’i":       (294, 0, 108, 36, 6, 6, 30, 0, 36, 72, 0),
    "Full Gospel":       (762, 6, 126, 0, 132, 78, 36, 0, 36, 324, 24),
    "Jehovah's Witness": (714, 120, 120, 0, 186, 0, 36, 0, 0, 240, 12),
    "Orthodox":          (24, 0, 12, 0, 0, 0, 6, 0, 6, 0, 0),
    "Jewish":            (84, 6, 6, 0, 6, 12, 18, 0, 6, 30, 0),
    "Pentecostal":       (558, 36, 168, 0, 0, 48, 84, 24, 18, 174, 6),
    "Nazarene":          (246, 138, 0, 0, 0, 0, 0, 0, 0, 108, 0),
    "Baptist":           (774, 180, 228, 0, 0, 0, 48, 0, 6, 312, 0),
    "Other religion":    (3_178, 222, 1_277, 0, 150, 0, 96, 12, 96, 1_325, 0),
    "No religion":       (702, 30, 276, 84, 54, 0, 48, 0, 6, 204, 0),
}
# Table A's religion percents, printed p.19, same column order
TA_PCT = {
    "CCCAS":         (33.3, 31.2, 25.5, 47.7, 40.7, 46.7, 35.9, 55.7, 46.5, 25.2, 88.3),
    "Catholic":      (18.1, 19.9, 25.9, 5.6, 14.7, 10.1, 30.5, 17.9, 17.4, 12.9, 0.0),
    "LDS _ Mormons": (15.8, 15.1, 9.9, 17.9, 8.4, 17.1, 11.1, 9.3, 8.6, 25.0, 0.0),
}

# The county test (taxonomy/as2015.py docstring)
TEST_DRAWS = 20_000
TEST_SEED = 20260915
TEST_ALPHA = 0.05

NUM = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0"}


def fetch():
    import urllib.request

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) == PDF_SIZE:
        print("already have", PDF)
        return
    with urllib.request.urlopen(urllib.request.Request(URL, headers=UA), timeout=600) as r:
        body = r.read()
    try:
        check_body(body, "pdf", where=URL, pin_size=PDF_SIZE, pin_digest=PDF_DIGEST)
    except FetchCheckError as e:
        raise SystemExit(f"{os.path.basename(PDF)}: {e}")
    with open(PDF + ".part", "wb") as fh:
        fh.write(body)
    os.replace(PDF + ".part", PDF)
    print(f"wrote {PDF} ({len(body):,} bytes)")


def _lines(doc, index):
    return [x for x in (" ".join(s.split()) for s in doc.load_page(index).get_text().splitlines()) if x]


def block(doc, index, caption):
    """The text lines between `caption` and the next `Source` line on one page."""
    lines = _lines(doc, index)
    at = [i for i, x in enumerate(lines) if x.startswith(caption)]
    if len(at) != 1:
        raise SystemExit(f"page index {index}: caption {caption!r} found {len(at)} times")
    end = next((j for j in range(at[0] + 1, len(lines)) if lines[j].startswith("Source")), None)
    if end is None:
        raise SystemExit(f"page index {index}: no Source line after {caption!r}")
    return lines[at[0] + 1:end]


def _num(s):
    return float(s.replace(",", "")) if "." in s else int(s.replace(",", ""))


def rows(lines):
    """[(label lines, numbers)]: the text layer prints a row's label, then its numbers one a line.
    A label can take two lines (`Jehovah's` / `Witness`); header lines land in the first label."""
    out, label, nums = [], [], []
    for x in lines:
        if NUM.match(x):
            nums.append(_num(x))
        else:
            if nums:
                out.append((label, nums))
                label, nums = [], []
            label.append(x)
    if nums:
        out.append((label, nums))
    return out


def label_of(label, wanted):
    """The wanted label this row carries (its last one or two lines), else None."""
    for w in wanted:
        for n in (1, 2):
            got = " ".join(label[-n:])
            if ALIAS.get(got, got) == w:
                return w
    return None


def header_counties(label):
    """Rebuild the county names from Table 1.6's two header rows, which the text layer prints as
    the first-row fragments (`Maopu-`, `Vai-`, ...) and then the second row (`Ituau`, `tasi`, ...)."""
    try:
        i = label.index("Religion")
    except ValueError:
        raise SystemExit(f"Table 1.6 header has no `Religion`: {label}")
    frags = [x for x in label[:i] if x.endswith("-")]
    second = label[i + 2:-1]                   # after `Religion`, `Total`; before the row's `Total`
    names = []
    for tok in second:
        if tok[:1].islower():
            if not frags:
                raise SystemExit(f"Table 1.6 header: nothing to join {tok!r} to")
            names.append(frags.pop(0)[:-1] + tok)
        else:
            names.append(tok)
    if frags:
        raise SystemExit(f"Table 1.6 header: fragments left over {frags}")
    return names


def read_t16(doc):
    rs = rows(block(doc, PAGE_16, CAP_16))
    if not rs or rs[0][0][-1] != "Total":
        raise SystemExit("Table 1.6: the first row is not `Total`")
    names = header_counties(rs[0][0])
    got = {"Total": tuple(rs[0][1])}
    for label, nums in rs[1:]:
        w = label_of(label, CATS)
        if w is None or w in got:
            raise SystemExit(f"Table 1.6: unexpected or repeated row {label}")
        got[w] = tuple(nums)
    return names, got


def national_columns(doc, index, caption):
    """Blocks of {religion: first number} from a cross-tabulation, split at Total/Males/Females.

    Rows before the first block are header: Table 2.3's age bands print `9`, `14` and so on as
    lines of their own, which read as numbers."""
    blocks, cur = [], None
    for label, nums in rows(block(doc, index, caption)):
        last = label[-1]
        if last in ("Total", "Males", "Females"):
            cur = {"_block": last, "_total": nums[0]}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        w = label_of(label, CATS + ["NR"])
        if w is None:
            raise SystemExit(f"{caption}: row {label} is not a religion")
        cur[w] = nums[0]
    return blocks


def geography_test():
    """{category: p}: does the category's county pattern exceed what sampling households would give?

    Households per county are the sampled persons over the mean household size; a household counts
    once. Under the null each draws the category at the territory rate (binomial); the statistic is
    the 2 x 10 chi-square on households. Conservative: no finite-population correction for a
    one-in-six sample, and a household is taken to share one answer."""
    import numpy as np

    rng = np.random.default_rng(TEST_SEED)
    tot = np.array(T16_TOTAL[1:], dtype=float)
    persons = np.round(tot / WEIGHT)
    m = persons.sum() / HOUSEHOLDS
    hh = np.round(persons / m)

    def stat(a):
        p = a.sum() / hh.sum()
        if p <= 0 or p >= 1:
            return 0.0
        e1, e0 = hh * p, hh * (1 - p)
        return float((((a - e1) ** 2) / e1).sum() + ((((hh - a) - e0) ** 2) / e0).sum())

    out = {}
    for c in CATS:
        a = np.round(np.array(T16[c][1:], dtype=float) / WEIGHT) / m
        obs = stat(a)
        sims = rng.binomial(hh.astype(int)[None, :], a.sum() / hh.sum(),
                            size=(TEST_DRAWS, len(hh))).astype(float)
        null = np.array([stat(s) for s in sims])
        out[c] = (1 + int((null >= obs).sum())) / (1 + TEST_DRAWS)
    return out, m, hh


def check(doc):
    import as2015

    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("American Samoa - 2015 HIES report, Table 1.6\n")
    with open(PDF, "rb") as fh:
        body = fh.read()
    say(len(body) == PDF_SIZE and digest(body) == PDF_DIGEST,
        f"{os.path.basename(PDF)} is the pinned {PDF_SIZE:,} bytes, digest {PDF_DIGEST}")
    for good, msg in check_pdf_doc(doc, PDF_PAGES):
        say(good, msg)
    meth = " ".join(_lines(doc, PAGE_METHOD))
    say("1,838 households" in meth and "5.99668" in meth,
        "printed p.17 states 1,838 completed households and the weight 5.99668")
    say("write-in entry for religion" in " ".join(_lines(doc, PAGE_WRITEIN)),
        "printed p.23 states the survey used a write-in entry for religion")

    # 1. the page and the transcription are one table, in the header's column order
    names, t16 = read_t16(doc)
    say(names == COUNTIES, f"Table 1.6's header rebuilds the counties in order: {names}")
    want = dict(T16, Total=T16_TOTAL)
    say(t16 == want, f"Table 1.6 parsed off printed p.57: {len(t16)} rows x 11, equal to the transcription")
    if t16 != want:
        for k in want:
            if t16.get(k) != want[k]:
                print(f"        {k!r}: page {t16.get(k)} transcription {want[k]}")

    # 2. the weight
    cells = [v for c in CATS for v in T16[c]] + list(T16_TOTAL)
    worst = max(abs(v - round(v / WEIGHT) * WEIGHT) for v in cells)
    say(worst <= 0.5, f"every cell is within half a person of a whole multiple of {WEIGHT} "
                      f"(worst {worst:.3f})")
    persons = [round(v / WEIGHT) for v in T16_TOTAL]
    say(persons[0] == PERSONS and sum(persons[1:]) == PERSONS,
        f"{PERSONS:,} sampled persons, and the counties' own sum to the same: {persons[1:]}")

    # 3. closure
    col = [sum(T16[c][i] for c in CATS) - T16_TOTAL[i] for i in range(11)]
    say(all(abs(d) <= 8 for d in col), f"the 16 rows sum to every printed total within 8: {col}")
    row = {c: sum(T16[c][1:]) - T16[c][0] for c in CATS}
    say(all(abs(d) <= 5 for d in row.values()),
        f"the ten counties sum to every row's total within 5: {[d for d in row.values() if d]}")

    # 4. Table 1.1 and Table A
    t11 = rows(block(doc, PAGE_11, CAP_11))
    first = next((nums for label, nums in t11 if label[-1] == "Total"), None)
    say(first is not None and tuple(first) == T16_TOTAL, "Table 1.1 prints the same county totals")
    ta = {}
    for label, nums in rows(block(doc, PAGE_A, CAP_A)):
        w = label_of(label, list(TA_PCT))
        if w and w not in ta:
            ta[w] = tuple(nums)
    say(ta == TA_PCT, "Table A's religion percents parse as transcribed")
    bad = [(c, i) for c in TA_PCT for i in range(11)
           if abs(100.0 * T16[c][i] / T16_TOTAL[i] - TA_PCT[c][i]) > 0.05 + 1e-9]
    say(not bad, f"each Table A percent is Table 1.6's count over its total, to the decimal {bad or ''}")

    # 5. the cross-tabulations
    national = {c: T16[c][0] for c in CATS}
    for index, cap in ((PAGE_23, CAP_23), (PAGE_33, CAP_33)):
        b = national_columns(doc, index, cap)
        say(len(b) == 1 and b[0]["_total"] == T16_TOTAL[0]
            and {c: b[0].get(c) for c in CATS} == national,
            f"{cap}: the same national column")
    b = national_columns(doc, PAGE_43, CAP_43)
    say([x["_block"] for x in b] == ["Total", "Males", "Females"], f"{CAP_43}: three blocks")
    if len(b) == 3:
        say({c: b[0].get(c) for c in CATS} == national, f"{CAP_43}: the same national column")
        say(all(x.get("NR") == 0 for x in b), f"{CAP_43}: `NR` is 0 in all three blocks")
        diff = {c: b[1][c] + b[2][c] - b[0][c] for c in CATS}
        say(all(abs(d) <= 1 for d in diff.values())
            and abs(b[1]["_total"] + b[2]["_total"] - b[0]["_total"]) <= 1,
            f"{CAP_43}: males + females = total within 1 "
            f"({sum(1 for d in diff.values() if d)} rows off by one)")

    # 6. the county test
    p, m, hh = geography_test()
    passing = tuple(c for c in CATS if p[c] < TEST_ALPHA)
    print(f"\n  county test: {HOUSEHOLDS:,} households, {m:.2f} sampled persons each; households "
          f"per county {hh.astype(int).tolist()}")
    for c in CATS:
        share = " ".join(f"{100.0 * T16[c][i] / T16_TOTAL[i]:5.1f}" for i in range(1, 11))
        print(f"    {c:<18} persons {round(T16[c][0] / WEIGHT):>5}  p={p[c]:.4f}  "
              f"{'own shares' if p[c] < TEST_ALPHA else 'territory rate'}   {share}")
    say(set(passing) == set(as2015.OWN_GEOGRAPHY),
        f"the verdict equals taxonomy/as2015.py OWN_GEOGRAPHY: {passing}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def emit():
    out = []
    for i, county in enumerate(COUNTIES, start=1):
        for c in CATS:
            n = T16[c][i]
            if n <= 0:
                continue
            out.append({
                "geo_id": county, "geo_level": "county", "geo_name": county,
                "source_category": c, "count": n, "basis": BASIS, "year": YEAR,
                "source_id": SOURCE_ID,
                "note": (f"Table 1.6 weighted persons ({round(n / WEIGHT)} sampled x {WEIGHT}); "
                         f"county total {T16_TOTAL[i]}"),
            })
    return out


def main():
    import fitz

    if "--fetch" in sys.argv:
        fetch()
    if not os.path.exists(PDF):
        raise SystemExit(f"{PDF} missing - run: python sources/as.py --fetch")
    check(fitz.open(PDF))
    out = emit()

    print(f"\n  10 counties, {sum(r['count'] for r in out):,} weighted people")
    for c in CATS:
        print(f"    {T16[c][0]:>7,}  {100.0 * T16[c][0] / T16_TOTAL[0]:6.2f}%  {c}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(OUT + ".part", OUT)
    print("\nwrote", OUT, f"({len(out)} rows)")


if __name__ == "__main__":
    main()
