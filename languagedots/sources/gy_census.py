"""Guyana, Population and Housing Census 2012 (Bureau of Statistics): ethnic background by
administrative region -> data/normalized/gy.csv.

    python sources/gy_census.py

THE TABLE is Table 2.3 of the 2012 Census Compendium 2 ("Regional Distribution of the Population
by Nationality Background/Ethnicity"), religiondots' copy (religiondots/data/raw/gy/
Final_2012_Census_Compendium2.pdf, p. 6 printed / PDF page 8, read-only), typed below. Its note:
'Not Stated' (321) and estimated 'No-Contact Persons' (16,331) were added and prorated, so the
table covers the whole 2012 population, 746,955.

NO LANGUAGE QUESTION in the 2012 census (the compendium has no language table). taxonomy/gy2012.py
reads each group as a language (AGENT_BRIEF section 2, ethnicity rule).

CHECKS: every row sums to its printed Total, every column to its printed Total, and the printed
grand total is 746,955; the region totals match religiondots' gy_lookup.csv census column.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

REGIONS = list(range(1, 11))
TABLE_2_3 = {   # regions 1..10, then the printed Total
    "African / Black": [635, 5891, 22774, 126378, 16472, 23383, 2135, 858, 353, 19604, 218483],
    "Amerindian": [17846, 8834, 2820, 7066, 1270, 1801, 6833, 8009, 20808, 3205, 78492],
    "Chinese": [14, 41, 192, 737, 44, 178, 25, 9, 10, 127, 1377],
    "East Indian": [472, 20861, 64183, 109105, 27234, 72406, 1569, 282, 253, 1128, 297493],
    "Mixed": [8616, 11046, 17652, 66844, 4740, 11727, 7514, 1838, 2708, 15847, 148532],
    "Portuguese": [46, 105, 84, 1148, 41, 73, 223, 76, 73, 41, 1910],
    "White": [12, 31, 31, 192, 16, 60, 9, 5, 29, 30, 415],
    "Other": [2, 1, 49, 93, 3, 24, 67, 0, 4, 10, 253],
}
TOTAL = [27643, 46810, 107785, 311563, 49820, 109652, 18375, 11077, 24238, 39992, 746955]


def main():
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'BAD'} {msg}")

    bad = [k for k, v in TABLE_2_3.items() if sum(v[:10]) != v[10]]
    check(not bad, f"every group's regions sum to its printed total ({bad})")
    cols = [sum(v[i] for v in TABLE_2_3.values()) for i in range(11)]
    check(cols == TOTAL, f"every region's groups sum to its printed total ({cols} vs {TOTAL})")
    lut = pd.read_csv(RD / "data" / "geo" / "gy" / "gy_lookup.csv")
    check(list(lut["census"]) == TOTAL[:10], "region totals equal religiondots' gy_lookup census")
    if not ok:
        raise SystemExit("reconciliation FAILED")
    rows = [dict(geo_level="region", geo_id=str(r), source_category=k, count=v[i])
            for k, v in TABLE_2_3.items() for i, r in enumerate(REGIONS) if v[i]]
    df = pd.DataFrame(rows)
    out = HERE / "data" / "normalized" / "gy.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {df['count'].sum():,} people)")


if __name__ == "__main__":
    main()
