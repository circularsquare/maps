"""Antigua and Barbuda: first language from the 2011 census's country of birth, national
-> data/normalized/ag.csv.

    python sources/ag_census.py

NO CENSUS LANGUAGE QUESTION (2011). Built as Barbados (sources/bb.md): the native-born on the
national creole, immigrants on their birth country's language. Every row `derived`.

THE TABLE: 2011 Population and Housing Census, Q58 country of birth, all of Antigua and Barbuda,
as run on the Statistics Division's Redatam WebServer and saved as PDF on 2024-12-15
(redatam.org/redatg/tempo/46442/~tmp_4644201.pdf, which redatam.org's 2025 relaunch dropped;
fetched from the Wayback Machine, data/raw/ag/ag_2011_country_of_birth.pdf). National only:
the WebServer that could cross birthplace by parish is no longer online.

CHECKS: the categories sum to 84,818, two more than the printed Total 84,816 (as printed).
"""
import re
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "ag" / "ag_2011_country_of_birth.pdf"
OUT = HERE / "data" / "normalized" / "ag.csv"
TOTAL = 84_816


def main():
    import fitz
    lines = [x.strip() for p in fitz.open(RAW) for x in p.get_text().split("\n") if x.strip()]
    i = lines.index("Cumul %") + 1
    rows = {}
    while lines[i] != "Total":
        lab, n = lines[i], int(re.sub(r"\D", "", lines[i + 1]))   # thousands use a narrow space
        assert re.match(r"^[\d.]+%$", lines[i + 2]) and re.match(r"^[\d.]+%$", lines[i + 3]), lines[i:i + 4]
        rows[lab] = n
        i += 4
    # the rows sum to 84,818, two more than the printed Total; the output prints it that way
    assert int(re.sub(r"\D", "", lines[i + 1])) == TOTAL and sum(rows.values()) == TOTAL + 2
    out = [dict(geo_id="AG", geo_level="country", geo_name="Antigua and Barbuda",
                source_category=k, count=v, tier="derived", year=2011)
           for k, v in rows.items() if k != "Not Stated"]
    df = pd.DataFrame(out)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} birthplaces, {df['count'].sum():,} people "
          f"({rows['Not Stated']:,} not stated, not drawn)")


if __name__ == "__main__":
    main()
