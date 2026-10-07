"""Anguilla: first language from the 2001 census's citizenship, national -> data/normalized/ai.csv.

    python sources/ai_census.py

NO CENSUS LANGUAGE QUESTION asking the first language. The Anguilla Statistics Department's
"1.1 Population.xlsx" (statistics.gov.ai/AllDocuments, saved to data/raw/ai/) has:
  Table 1.01.1.6  population 2001 by district, citizenship and age group: Anguillian, USA,
                  St. Kitts, Dominican Republic, Jamaica, Other Caribbean, UK, Other; three age
                  blocks (under 15, 15-50, 51+), summed here. 11,430 people.
  Table 1.01.1.9  2001, "second language(s) spoken": 10,376 of 11,430 (91%) speak one
                  language only; Spanish is the commonest second language (654). Context only.
No country-of-birth table for 2001, 2011 or 2022 is published; Anguilla has no REDATAM base.

MAPPING (St Kitts and Montserrat conventions, sources/kn_census.py, ms_census.py): Anguillian
and Kittitian citizens on the Leeward creole (Glottolog's Antigua and Barbuda Creole English,
anti1245, whose countries include AI; node `antiguan`, bb.txt); US and UK citizens on English
(many are Anguillians born or naturalised abroad); Dominican Republic on Spanish; Jamaica on
Jamaican Creole; "Other Caribbean" on the creoles group node (it mixes English, French and Dutch
Caribbean creoles: an unnamed remainder); "Other" on `other`. Every row `derived`.

CHECKS: the three age blocks' district rows sum to their block totals; blocks sum to 11,430.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "data" / "raw" / "ai" / "1.1 Population.xlsx"
OUT = ROOT / "data" / "normalized" / "ai.csv"
COLS = ["Anguillian", "USA", "St. Kitts", "Dominican Republic", "Jamaica", "Other Caribbean",
        "UK", "Other"]
TOTAL = 11_430


def main():
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)["Table 1.01.1.6"]
    rows = [[c for c in r if c is not None] for r in ws.iter_rows(values_only=True)]
    blocks = []   # [(block total row, district rows)]; the total row heads its block
    for r in rows:
        if not r or not isinstance(r[0], str):
            continue
        if r[0].startswith("Total"):
            blocks.append(([int(x) for x in r[1:10]], []))
        elif "(EDs" in r[0] and blocks:
            blocks[-1][1].append([int(x) for x in r[1:10]])
    assert len(blocks) == 3, len(blocks)
    for blk, dist in blocks:
        assert len(dist) == 14, len(dist)
        assert [sum(d[k] for d in dist) for k in range(9)] == blk, blk
        assert sum(blk[:8]) == blk[8], blk
    blocks = [b for b, _ in blocks]
    nat = [sum(b[k] for b in blocks) for k in range(9)]
    assert nat[8] == TOTAL, nat
    print("2001 citizenship: " + ", ".join(f"{c} {n:,}" for c, n in zip(COLS, nat[:8])))
    df = pd.DataFrame([dict(geo_id="AI", geo_level="country", geo_name="Anguilla",
                            source_category=c, count=n, tier="derived", year=2001,
                            source_id="aia_census_2001_t1.01.1.6") for c, n in zip(COLS, nat[:8])])
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.name}: {df['count'].sum():,} people")


if __name__ == "__main__":
    main()
