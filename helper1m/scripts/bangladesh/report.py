"""Parse the upazila and city-corporation-thana tables out of the PHC 2022
National Report (Volume I), BBS, November 2023.

  Table P33  City Corporation by Thana   (PDF pages 433-434, printed 386-387)
  Table P35  Upazila (except City Corporation areas)   (PDF pages 442-451, printed 395-404)

The PDF is text, not a scan: every cell comes out of PyMuPDF as its own line, a
row label followed by its numbers. Labels that wrap (e.g. "Gazipur District
(Except" / "City Corporation)") are joined back.

Writes data/bangladesh/phc2022_units.csv:
  table, division, district, cc, name, hh, pop, male, female

`pop` is the table's Total column, which is male + female: the report's
third-gender (hijra) count, 8,124 nationally, is not carried at upazila level.
Enumerated population, not the PEC-adjusted figure (169,828,911).
"""
import csv
import re
from pathlib import Path

import fitz

HELPER = Path(__file__).resolve().parents[2]
PDF = HELPER / "data/bangladesh/raw/phc2022_national_report_vol1.pdf"
OUT = HELPER / "data/bangladesh/phc2022_units.csv"

P33_PAGES = range(433, 435)
P35_PAGES = range(442, 452)

NUM = re.compile(r"^-?[\d.,]+$|^-$")
SKIP = {"Upazila", "Total", "Household", "Population", "Size (General)",
        "Literacy Rate (7+ yrs.)", "Male", "Female", "City Corporation/Thana",
        "Literacy Rate", "(7+ yrs.)", "Size", "(General)"}


def rows(doc, pages):
    """Yield (page, label, [numbers]) for every table row on the pages."""
    for p in pages:
        lines = [l.strip() for l in doc[p].get_text().splitlines()]
        label, nums = [], []
        for l in lines:
            if not l or "Population and Housing Census" in l or l in SKIP \
                    or l.startswith("Table") or l.startswith("by Thana"):
                continue
            if NUM.match(l):
                if label:
                    nums.append(l)
                continue  # the 1 2 4 5 ... column-number header
            if label and nums:
                yield p, " ".join(label), nums
                label, nums = [], []
            label.append(l)
        if label and nums:
            yield p, " ".join(label), nums


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def parse():
    doc = fitz.open(PDF)
    out = []

    # --- P33: city corporations by thana
    cc = None
    for p, label, nums in rows(doc, P33_PAGES):
        label = clean(label)
        hh, pop, male, female = (int(x) for x in nums[:4])
        if label == "Total":
            continue
        if label.endswith("City Corporation"):
            cc = label
            out.append(dict(table="P33", division="", district="", cc=cc, name="",
                            hh=hh, pop=pop, male=male, female=female))
            continue
        out.append(dict(table="P33", division="", district="", cc=cc, name=label,
                        hh=hh, pop=pop, male=male, female=female))

    # --- P35: upazilas outside city corporations
    division = district = None
    for p, label, nums in rows(doc, P35_PAGES):
        label = clean(label)
        hh, pop, male, female = (int(x) for x in nums[:4])
        if label.startswith("Total"):
            out.append(dict(table="P35", division="", district="", cc="", name="TOTAL",
                            hh=hh, pop=pop, male=male, female=female))
            continue
        m = re.match(r"^(?:(\w[\w' ]*?) Division )?(.+?) District( \(Except City Corporation\))?$", label)
        if m:
            if m.group(1):
                division = m.group(1)
            district = m.group(2)
            out.append(dict(table="P35", division=division, district=district, cc="",
                            name="", hh=hh, pop=pop, male=male, female=female))
            continue
        out.append(dict(table="P35", division=division, district=district, cc="",
                        name=label, hh=hh, pop=pop, male=male, female=female))
    return out


def main():
    recs = parse()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"{len(recs)} rows -> {OUT}")


if __name__ == "__main__":
    main()
