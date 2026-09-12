"""Cyprus — the 2001 census's religion-by-district table, read as a CHECK and never drawn.

Reads data/raw/cy/census_2001_vol1.pdf and prints a comparison. Writes nothing to
data/normalized/, on purpose.

**WHY THIS FILE EXISTS.** `sources/cy.py` allocates 2021's national religion counts to 396
communities using citizenship-group composition, and that arithmetic rests on one assumption
it cannot test from 2021 data alone: **that religion does not vary between places WITHIN a
citizenship group.** Table 29 of the 2001 census, *Population by sex, religion, district and
urban/rural area*, is the only tabulation in any Cypriot census that measures religion against
a geography, so it is the only thing that can put a number on how wrong that assumption is.
2011 does not have one and neither does 2021; `sources/cy.md` §5 records the search.

**IT IS A CHECK AND NOT A SOURCE, FOR A REASON THAT IS EASY TO STATE.** Cyprus was 94.8%
Orthodox in 2001 and is 74.5% now, on a population a third larger. Every part of the
difference arrived after 2001. Drawing 2001 would put a Cyprus on the map that no longer
exists, and the map already has India's 2011 as its oldest source; this would be worse.

**WHAT IT SAYS, AND IT IS EXACTLY THE DIAGNOSIS `cy.py` PREDICTED.** Run it. The short version
is that the 2001 concentrations split cleanly in two. The ones that live in the CITIZENSHIP
dimension are reproduced by the 2021 allocation, because that is the dimension it allocates
on; the ones that live INSIDE the Cypriot population are invisible to it, and those are the
Armenians and the Maronites, both about twice as concentrated in Lefkosia as the population is.

**THE PDF IS A FIXED-WIDTH PIPE TABLE AND ITS GREEK IS MOJIBAKE.** CYSTAT typeset the 2001
volumes in a legacy Greek symbol font, so the Greek half of every label extracts as Latin
lookalikes (`ĬȇǾȈȀǼȊȂǹ` is ΘΡΗΣΚΕΥΜΑ). The English half of each caption, every column header
and every digit extract correctly, so the parse anchors on the English and never on the Greek.
Numbers use `.` as the thousands separator.

Usage:
    python sources/cy_2001.py --fetch    one 5 MB PDF from library.cystat.gov.cy
    python sources/cy_2001.py            parse and compare
"""

import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "6")          # [[feedback_cap_cpu]]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cy")
PDF = os.path.join(RAW, "census_2001_vol1.pdf")

PDF_URL = ("https://library.cystat.gov.cy/Documents/Publication/"
           "CENSUS%20OF%20POPULATION%202001-VOL.1.pdf")

# Table 29 begins on this page, 0-indexed. Asserted by its own English caption rather than
# trusted, so a re-issued volume fails here instead of parsing the wrong table.
PAGE = 252
CAPTION = re.compile(r"TABLE\s+29\.\s+POPULATION BY SEX, RELIGION, DISTRICT", re.I)

# The nine religion columns in print order, from the English header row.
CATEGORIES_2001 = ["Orthodox", "Armenians", "Maronites", "Roman catholic", "Moslems",
                   "Protestants", "Atheists", "Other", "Not Stated"]

DISTRICTS = ["Lefkosia", "Ammochostos", "Larnaka", "Lemesos", "Pafos"]

# UNSD Demographic Yearbook table 28's Cyprus 2001 return, forwarded by CYSTAT and NOT read
# off this PDF. Every one of the nine is asserted against it.
UNSD_2001 = {"Orthodox": 653635, "Armenians": 1741, "Maronites": 3930,
             "Roman catholic": 10240, "Moslems": 4182, "Protestants": 6839,
             "Atheists": 1500, "Other": 6505, "Not Stated": 993}
TOTAL_2001 = 689565

# 2001 category -> the 2021 category that is the same thing. Buddhists, Sikhs and Hindus have
# no 2001 row at all (they are inside `Other`), which is itself the story: they are 11,809
# people in 2021 and were too few to name in 2001.
SAME = {
    "Orthodox": "Christian Orthodox",
    "Armenians": "Armenian church",
    "Maronites": "Maronite church",
    "Roman catholic": "Roman Catholic",
    "Moslems": "Muslim",
    "Protestants": "Anglican/Protestant",
    "Atheists": "Atheist/No Religion",
}

# community code leading digit -> district, the same map sources/cy_geo.py uses
DISTRICT_OF = {"1": "Lefkosia", "3": "Ammochostos", "4": "Larnaka",
               "5": "Lemesos", "6": "Pafos"}


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(PDF) and os.path.getsize(PDF) > 4_000_000:
        print("already have", PDF)
        return
    print("GET", PDF_URL)
    r = requests.get(PDF_URL, timeout=900, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(PDF + ".part", "wb") as fh:
        fh.write(r.content)
    os.replace(PDF + ".part", PDF)
    # [[reference_pdf_truncated_at_source]]: a matching Content-Length is not an intact file.
    with open(PDF, "rb") as fh:
        fh.seek(-2048, os.SEEK_END)
        if b"%%EOF" not in fh.read():
            raise SystemExit(f"{PDF} has no %%EOF trailer -- truncated at source")
    print(f"  {os.path.getsize(PDF):,} bytes")


def _num(s):
    s = s.strip()
    if not s:
        return None
    return int(s.replace(".", "").replace(",", ""))


def parse_2001():
    """{district or 'Total': {category: count}} for males and females, urban and rural."""
    import fitz

    if not os.path.exists(PDF):
        raise SystemExit(f"missing {PDF} -- run with --fetch first")
    doc = fitz.open(PDF)
    text = doc[PAGE].get_text()
    if not CAPTION.search(text):
        raise SystemExit(f"PDF page {PAGE + 1} is not Table 29 -- the volume has been "
                         "re-issued and PAGE must be re-found by its English caption")
    # The English header row names the columns; assert it before reading a number.
    #
    # TWO OF THE NINE CANNOT BE MATCHED LITERALLY AND THAT IS THE FONT, NOT A TYPO.
    # `Orthodox` extracts as `Ȅrthodox`: its capital O is a Greek omicron glyph from the
    # legacy symbol font, so the English word is half Greek and `"Orthodox" in text` is
    # False on a page that plainly says Orthodox. And `Roman catholic` is typeset over two
    # header lines (`Roman-` / `catholic`), so it is never one string. Both are matched on
    # the part that survives, which is the whole point of anchoring on the header at all.
    HEADER_PATTERNS = [r"rthodox", r"Armenians", r"Maronites", r"catholic", r"Moslems",
                       r"Protestants", r"Atheists", r"Other", r"Not Stated"]
    missing = [p for p in HEADER_PATTERNS if not re.search(p, text)]
    if missing:
        raise SystemExit(f"column headers missing from the page: {missing}")

    out, seen = {}, set()
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        label = cells[1] if len(cells) > 1 else ""
        # `Total` and the five districts appear three times each on the page: once for
        # males-and-females, once for urban, once for rural. Only the FIRST block is the
        # whole enumerated population, so each label is taken once and then locked out.
        if label.endswith("Total"):
            key = "Total"
        else:
            key = next((d for d in DISTRICTS if label.endswith("- " + d)), None)
        if key is None or key in seen:
            continue
        nums = [_num(c) for c in cells[2:2 + 1 + len(CATEGORIES_2001)]]
        if any(n is None for n in nums) or len(nums) != 1 + len(CATEGORIES_2001):
            continue
        seen.add(key)
        row = dict(zip(CATEGORIES_2001, nums[1:]))
        row["_total"] = nums[0]
        out[key] = row

    if sorted(out) != sorted(["Total"] + DISTRICTS):
        raise SystemExit(f"parsed {sorted(out)}, expected Total plus the five districts")

    # the five districts must sum to the printed Total, on every column
    for c in CATEGORIES_2001 + ["_total"]:
        s = sum(out[d][c] for d in DISTRICTS)
        if s != out["Total"][c]:
            raise SystemExit(f"!! {c}: districts sum to {s:,}, printed total "
                             f"{out['Total'][c]:,}")
    # and the printed Total must be UNSD's own return
    if out["Total"]["_total"] != TOTAL_2001:
        raise SystemExit(f"!! total {out['Total']['_total']:,}, expected {TOTAL_2001:,}")
    for c, want in UNSD_2001.items():
        if out["Total"][c] != want:
            raise SystemExit(f"!! {c}: PDF {out['Total'][c]:,} vs UNSD {want:,}")
    if sum(out["Total"][c] for c in CATEGORIES_2001) != TOTAL_2001:
        raise SystemExit("!! the 2001 categories do not partition the total")
    print(f"Table 29 parsed: {TOTAL_2001:,} people, {len(CATEGORIES_2001)} categories, "
          "5 districts")
    print("  every column's districts sum to its printed total, and all nine totals "
          "match UNSD table 28")
    return out


def allocation_2021():
    """{district: {2021 category: count}} from what sources/cy.py actually wrote."""
    import csv
    from collections import defaultdict

    p = os.path.join(ROOT, "data", "normalized", "cy.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run `python sources/cy.py` first")
    out = defaultdict(lambda: defaultdict(float))
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = DISTRICT_OF[row["geo_id"][0]]
            out[d][row["source_category"]] += float(row["count"])
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()

    old = parse_2001()
    new = allocation_2021()

    tot01 = {d: old[d]["_total"] for d in DISTRICTS}
    tot21 = {d: sum(new[d].values()) for d in DISTRICTS}
    n01, n21 = sum(tot01.values()), sum(tot21.values())

    print("\npopulation by district, 2001 measured vs 2021 measured:")
    for d in DISTRICTS:
        print(f"  {d:<14} {tot01[d]:>9,} ({100*tot01[d]/n01:5.1f}%)   "
              f"{tot21[d]:>9,.0f} ({100*tot21[d]/n21:5.1f}%)")

    print("\nCONCENTRATION IN A DISTRICT, as a multiple of that district's population share.")
    print("1.00 means the religion is spread exactly like the population. The 2001 column is")
    print("MEASURED; the 2021 column is what sources/cy.py's allocation produces.")
    for c01, c21 in SAME.items():
        print(f"\n  {c01} (2001)  /  {c21} (2021)")
        print(f"    {'district':<14} {'2001 n':>8} {'2001 x':>8} {'2021 x':>8}   verdict")
        for d in DISTRICTS:
            share01 = tot01[d] / n01
            share21 = tot21[d] / n21
            r01 = (old[d][c01] / old["Total"][c01]) / share01 if old["Total"][c01] else 0
            got21 = new[d].get(c21, 0.0)
            nat21 = sum(new[x].get(c21, 0.0) for x in DISTRICTS)
            r21 = (got21 / nat21) / share21 if nat21 else 0
            gap = abs(r01 - r21)
            verdict = "reproduced" if gap < 0.25 else ("MISSED" if r01 > 1.25 or r01 < 0.75
                                                       else "")
            print(f"    {d:<14} {old[d][c01]:>8,} {r01:>8.2f} {r21:>8.2f}   {verdict}")

    print("\nREAD IT LIKE THIS. Where a 2001 concentration lives in the citizenship")
    print("dimension, the allocation finds it, because that is the dimension it allocates on.")
    print("Where it lives inside the Cypriot population, the allocation cannot see it at all")
    print("and returns a flat 1.00. Those rows are the cost of drawing Cyprus this way, and")
    print("they are named in taxonomy/cy2021.py's REVIEW and in the country note.")
    print("\nTHE 2001 COLUMN IS NOT GROUND TRUTH FOR 2021 EITHER. Twenty-four years and")
    print("280,000 arrivals sit between them, so a gap here is an upper bound on the error")
    print("and not a measurement of it.")


if __name__ == "__main__":
    main()
