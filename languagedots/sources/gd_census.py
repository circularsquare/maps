"""Grenada: Grenadian Creole and English per parish from the 2021 census's ethnicity table
-> data/normalized/gd.csv.

    python sources/gd_census.py

NO CENSUS LANGUAGE QUESTION (2021 preliminary report, every table listed; 2011 likewise).
Built as Barbados (sources/bb.md): the national creole for everyone, except white Grenadians on
English. Every row `derived`.

THE TABLES (religiondots' copy of the CSO's 2021 preliminary report,
`../religiondots/data/raw/gd/gd_census_2021_preliminary.pdf`, read only), on the
non-institutional population in private dwellings (108,279 of 109,021):
  * parish totals: religiondots' normalized gd.csv `parish` TOTAL rows (the census's Town of St
    George folded into St George, as religiondots draws it);
  * Table 22 (p.19), population by ethnicity and parish: the WHITE/CAUCASIAN column, typed in
    below and checked against Table 22's own total and Table 16's national 973.

Everyone else (African descent, Mixed, East Indian, ..., and the 1,721 not stated) is drawn as
Grenadian Creole. Born abroad (Table 19: 5,526, 5.1%) is national only and has no country, so
immigrants cannot be told apart and are drawn as Grenadian Creole too (sources/gd.md).
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

OUT = HERE / "data" / "normalized" / "gd.csv"
TOTAL = 108_279
# Table 22: (TOTAL, WHITE/CAUCASIAN) per census unit, typed from p.19
TABLE22 = {
    "St. George": (42_096 + 2_681, 713 + 18),     # rest of St George + Town of St George
    "St. John": (7_773, 12),
    "St. Mark": (3_938, 7),
    "St. Patrick": (7_846, 34),
    "St. Andrew": (24_755, 18),
    "St. David": (14_443, 89),
    "Carriacou and Petite Martinique": (4_747, 82),
}
WHITE_NATIONAL = 973   # Table 16 and Table 22's TOTAL row


def main():
    assert sum(t for t, _ in TABLE22.values()) == TOTAL
    assert sum(w for _, w in TABLE22.values()) == WHITE_NATIONAL
    rd = pd.read_csv(RD / "data" / "normalized" / "gd.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[])
    rd = rd[(rd["geo_level"] == "parish") & (rd["source_category"] == "TOTAL")]
    assert set(rd["geo_name"]) == set(TABLE22), set(rd["geo_name"]) ^ set(TABLE22)
    out = []
    for _, r in rd.iterrows():
        tot, white = TABLE22[r["geo_name"]]
        assert int(r["count"]) == tot, (r["geo_name"], r["count"], tot)
        for lab, n in (("White/Caucasian", white), ("Everyone else", tot - white)):
            out.append(dict(geo_id=r["geo_id"], geo_level="parish", geo_name=r["geo_name"],
                            source_category=lab, count=n, tier="derived", year=2021))
    df = pd.DataFrame(out)
    assert df["count"].sum() == TOTAL
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {df['geo_id'].nunique()} parishes, {TOTAL:,} people, "
          f"{WHITE_NATIONAL} on English")


if __name__ == "__main__":
    main()
