"""Cayman Islands: first language from the 2021 census's country of birth by district
-> data/normalized/ky.csv.

    python sources/ky_census.py

NO CENSUS LANGUAGE QUESTION (2021). Built as Barbados (sources/bb.md): the native-born on the
local language (here English; Caymanian speech is an English dialect and no creole node or
Glottolog entry separates it), immigrants on their birth country's language. Every row
`derived`.

THE TABLES: Economics and Statistics Office, *Cayman Islands' 2021 Census Report*
(religiondots' data/raw/ky/ky_census_report_2021.pdf, read only), Tables 4.12C-D and 4.13E-I
(pp. 128-134), population by country of birth, sex and status, one per district: George Town,
West Bay, Bodden Town, North Side, East End, Cayman Brac, Little Cayman. Only the first
number of each row (Total, both sexes, all statuses) is read; "-" is zero. Cayman Brac and
Little Cayman are summed into religiondots' Sister Islands unit.

CHECKS: each district's rows against its printed Total: over by at most 3 (rounding), and a
shortfall over 3 (rows the table leaves out: East End 7, Cayman Brac 26, Little Cayman 23) kept
as `Not listed in the district table` on `other`; the printed district Totals sum to the census
total 68,811 (Table 4.12A); each district within 6 of religiondots' district total.
"""
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

PDF = RD / "data" / "raw" / "ky" / "ky_census_report_2021.pdf"
OUT = HERE / "data" / "normalized" / "ky.csv"
TOTAL = 68_811
# table district -> religiondots unit name
UNIT = {"George Town": "George Town", "West Bay": "West Bay", "Bodden Town": "Bodden Town",
        "North Side": "North Side", "East End": "East End", "Cayman Brac": "Sister Islands",
        "Little Cayman": "Sister Islands"}
TITLE = re.compile(r"Table 4\.1[23][C-I]: (.+?) Population by Country of Birth, Sex and Status, 2021$")
NUM = re.compile(r"^(-|[\d,]+)$")


def _n(s):
    return 0 if s == "-" else int(s.replace(",", ""))


def parse():
    import fitz
    tables = {}
    for page in fitz.open(PDF):
        lines = [x.strip() for x in page.get_text().split("\n") if x.strip()]
        title = next((TITLE.match(x) for x in lines if TITLE.match(x)), None)
        if not title:
            continue
        dist = title.group(1)
        i = lines.index(dist) + 1          # the district's name heads the body
        rows = {}
        while i + 13 <= len(lines):
            lab, vals = lines[i], lines[i + 1:i + 13]
            if not all(NUM.match(v) for v in vals):   # the column header that ends the page
                break
            rows[lab] = _n(vals[0])
            i += 13
        total = rows.pop("Total")
        # Rows can miss the printed Total: the ESO rounds weighted counts (George Town's rows are
        # one over), and some district tables leave out a row their Total includes (East End has
        # no Guyana row, Cayman Brac no India or Costa Rica). A shortfall over 3 is kept as
        # `Not listed in the district table`, drawn on `other`; smaller gaps are rounding.
        gap = total - sum(rows.values())
        assert gap >= -3, (dist, gap)
        if gap:
            print(f"  {dist}: rows sum to {sum(rows.values()):,}, printed Total {total:,}")
        if gap > 3:
            rows["Not listed in the district table"] = gap
        tables[dist] = (total, rows)
    assert set(tables) == set(UNIT), set(tables) ^ set(UNIT)
    assert sum(t for t, _ in tables.values()) == TOTAL
    return tables


def main():
    import ky2021
    tables = parse()
    rd = pd.read_csv(RD / "data" / "normalized" / "ky.csv", keep_default_na=False, na_values=[])
    rd = rd[(rd["geo_level"] == "district") & (rd["source_category"] == "Total")]
    ids = dict(zip(rd["geo_name"], rd["geo_id"]))
    rdtot = dict(zip(rd["geo_name"], pd.to_numeric(rd["count"])))
    acc = {}
    for dist, (total, rows) in tables.items():
        u = UNIT[dist]
        for lab, n in rows.items():
            acc[(u, lab)] = acc.get((u, lab), 0) + n
    out = []
    for (u, lab), n in acc.items():
        if lab == "DK/NS" or n == 0:
            continue
        ky2021.resolve(lab)
        out.append(dict(geo_id=ids[u], geo_level="district", geo_name=u, source_category=lab,
                        count=n, tier="derived", year=2021))
    for u in set(UNIT.values()):
        assert abs(sum(n for (uu, _), n in acc.items() if uu == u) - rdtot[u]) <= 6, u
    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False, encoding="utf-8")
    dk = sum(n for (u, lab), n in acc.items() if lab == "DK/NS")
    print(f"wrote {OUT}: {df['geo_id'].nunique()} districts, {df['count'].sum():,} people "
          f"({dk} DK/NS not drawn)")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE / "taxonomy"))
    main()
