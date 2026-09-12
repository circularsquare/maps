"""Ireland — CSO Census 2022 religion.

Writes data/normalized/ie.csv from two CSO releases that cover the same universe
(usually resident population, 5,149,139) and reconcile to the person:

  * FY106  — Population by Religion x Administrative County.  24 real categories
             plus "All religions".  Finest geography for the DETAILED split.
  * SAPS   — Small Area Population Statistics 2022, Theme 2 Table 4.  Only FOUR
             categories (Catholic / Other religion / No religion / Not stated)
             but published down to Small Area (18,919 units).

That is spec.md 3.4's "structure from one source, totals from another" shape,
except that here both sources are the same census and the same year, so no
interpolation is involved -- only a change of geography.

The two nest exactly:
    SAPS Catholic      == FY106 "Roman Catholic"          3,540,412
    SAPS No religion   == FY106 "No religion"               755,455
    SAPS Not stated    == FY106 "Not stated"                345,165
    SAPS Other religion == the other 21 FY106 categories     508,107
Note what that means: FY106's "Atheist", "Agnostic" and "Lapsed (Roman)
Catholic" land in SAPS's OTHER RELIGION bucket, not in "No religion".

Categories are left exactly as CSO writes them (spec.md 2.4).  No taxonomy
mapping happens here.

Re-fetch: see sources/ie.md.
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ie")
OUT = os.path.join(ROOT, "data", "normalized", "ie.csv")

SOURCE_ID = "ie_census_2022"
YEAR = 2022
BASIS = "self_id"  # census self-declaration, every row

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# SAPS Theme 2 Table 4 column -> the category label CSO gives it in the glossary.
SAPS_RELIGION = [
    ("T2_4CA", "Catholic"),
    ("T2_4OR", "Other religion"),
    ("T2_4NR", "No religion"),
    ("T2_4NS", "Not stated"),
    ("T2_4T", "Total"),
]

TOTAL_NOTE = "universe total, not a religion category"


def read_fy106():
    """County + State rows, 24 detailed categories, from the PxStat CSV."""
    path = os.path.join(RAW, "FY106.csv")
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["STATISTIC"] != "FY106C01":  # C02 is the percentage statistic
                continue
            geo_id = r["C03789V04537"]
            name = r["Administrative Counties"]
            level = "state" if geo_id == "IE0" else "county"
            cat = r["Religion"]
            note = "CSO PxStat FY106"
            if cat == "All religions":
                note += "; " + TOTAL_NOTE
            rows.append({
                "geo_id": geo_id,
                "geo_level": level,
                "geo_name": name,
                "source_category": cat,
                "count": int(r["VALUE"]),
                "basis": BASIS,
                "year": YEAR,
                "source_id": SOURCE_ID,
                "note": note,
            })
    return rows


def read_saps(filename, level, note_prefix, skip_geogid=("Ireland",)):
    """Small Area or Electoral Division rows, 4 categories, from a SAPS CSV."""
    path = os.path.join(RAW, filename)
    rows = []
    with open(path, encoding="latin-1", newline="") as fh:
        for r in csv.DictReader(fh):
            # Both SAPS files carry a whole-State row; FY106 already supplies it
            # at a finer category split, so drop it here rather than emit the
            # same national figure twice on two different category systems.
            if r["GEOGID"] in skip_geogid:
                continue
            for col, cat in SAPS_RELIGION:
                note = note_prefix
                if cat == "Total":
                    note += "; " + TOTAL_NOTE
                rows.append({
                    "geo_id": r["GUID"],
                    "geo_level": level,
                    "geo_name": r["GEOGDESC"],
                    "source_category": cat,
                    "count": int(r[col]),
                    "basis": BASIS,
                    "year": YEAR,
                    "source_id": SOURCE_ID,
                    "note": note,
                })
    return rows


def check(rows):
    """Reconcile against CSO's published Census 2022 figures."""
    published = {
        "All religions": 5149139,
        "Roman Catholic": 3540412,
        "No religion": 755455,
        "Not stated": 345165,
    }

    def total(level, cat):
        return sum(r["count"] for r in rows
                   if r["geo_level"] == level and r["source_category"] == cat)

    print("national reconciliation")
    ok = True
    for cat, want in published.items():
        got = total("state", cat)
        flag = "OK " if got == want else "BAD"
        ok &= got == want
        print(f"  {flag} state  {cat:<18} {got:>10,}  (published {want:,})")

    for cat, want in published.items():
        got = total("county", cat)
        flag = "OK " if got == want else "BAD"
        ok &= got == want
        print(f"  {flag} county {cat:<18} {got:>10,}")

    # Every detailed category except Roman Catholic / No religion / Not stated
    # must add up to the SAPS "Other religion" bucket.
    lumped = {"All religions", "Roman Catholic", "No religion", "Not stated"}
    other = sum(r["count"] for r in rows
                if r["geo_level"] == "state" and r["source_category"] not in lumped)
    for level in ("electoral_division", "small_area"):
        for cat, want in [("Total", 5149139), ("Catholic", 3540412),
                          ("No religion", 755455), ("Not stated", 345165),
                          ("Other religion", other)]:
            got = total(level, cat)
            flag = "OK " if got == want else "BAD"
            ok &= got == want
            print(f"  {flag} {level:<18} {cat:<15} {got:>10,}")

    units = {}
    for r in rows:
        units.setdefault(r["geo_level"], set()).add(r["geo_id"])
    print("\nunits per level:", {k: len(v) for k, v in sorted(units.items())})
    print("rows:", len(rows))
    print("categories:",
          {lv: len({r["source_category"] for r in rows if r["geo_level"] == lv})
           for lv in sorted(units)})
    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    rows = []
    rows += read_fy106()
    rows += read_saps("SAPS_2022_CSOED3270923.csv", "electoral_division",
                      "SAPS 2022 Theme 2 Table 4; CSO Electoral Divisions 2022")
    rows += read_saps("SAPS_2022_Small_Area_UR_171024.csv", "small_area",
                      "SAPS 2022 Theme 2 Table 4; CSO Small Areas 2022")
    check(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
