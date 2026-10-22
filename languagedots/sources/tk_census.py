"""Tokelau: language spoken most often at home, 2016 census, national -> data/normalized/tk.csv.

    python sources/tk_census.py

THE QUESTION. 2016 Tokelau Census of Population and Dwellings: "language spoken most often at
home", asked of the usual residents present in Tokelau on census night aged 5 and over; plus the
languages in which the person can hold a conversation (several allowed), and the language first
learnt and still understood. The first is the home-language question, one answer per person.

THE TABLE. Tokelau National Statistics Office and Statistics New Zealand, 2016 Tokelau Census,
social profile tables (religiondots' copy, data/raw/tk/tk2016_tables_social_profile.xlsx, read in
place), Table 5.5, "Language spoken most often at home, by whether lived overseas for six months or
more": Tokelauan 877, Samoan 21, English 90, Tuvaluan 54, Kiribati 4, Other language 16, None (eg too
young to talk) 1, Not stated 6; total 1,069 aged 5 and over. It is national: no language table is
published by atoll (the report's only atoll figure on language is the share who speak two or more
languages).

THE DRAWING. 1,062 people (1,069 less the 6 not stated and the 1 who cannot talk), every row
`measured`, one national unit. Tokelau's 1,197 residents present on census night (aged 5+: 1,069)
are a subset of the 1,499 usual residents; the other 302 were overseas (religiondots'
`countries/tk.py`) and were not asked.

CHECKS (asserted): the rows sum to the printed total 1,069 and, by the "lived overseas" columns,
to 620 / 428 / 21; Table 5.4's "Total stated" + "Not stated" is 1,197 (the universe present), and
its 5+ rows sum to 1,069 people (1,197 less the 128 aged 0-4).
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

XLSX = RD / "data" / "raw" / "tk" / "tk2016_tables_social_profile.xlsx"
OUT = HERE / "data" / "normalized" / "tk.csv"


def main():
    x = pd.ExcelFile(XLSX)
    d = x.parse("Table 5.5", header=None).dropna(how="all").dropna(axis=1, how="all")
    rows = {}
    for _, r in d.iterrows():
        v = r.tolist()
        if isinstance(v[0], str) and len(v) == 5 and all(isinstance(n, (int, float)) and not pd.isna(n) for n in v[1:]):
            rows[v[0]] = [int(n) for n in v[1:]]
    assert list(rows) == ["Tokelauan", "Samoan", "English", "Tuvaluan", "Kiribati", "Other language",
                          "None (eg too young to talk)", "Not stated", "Total"], list(rows)
    tot = rows.pop("Total")
    assert tot == [620, 428, 21, 1069], tot
    assert [sum(r[k] for r in rows.values()) for k in range(4)] == [r for r in tot[:3]] + [1069]
    assert all(r[0] + r[1] + r[2] == r[3] for r in rows.values())
    d4 = x.parse("Table 5.4", header=None).dropna(how="all").dropna(axis=1, how="all")
    ages = d4[d4[0].astype(str).str.match(r"^(\d+–\d+|\d+\+)$")]
    under5 = int(ages[ages[0] == "0–4"].iloc[0, -1])
    total_present = int(ages.iloc[:, -1].sum())
    assert total_present == 1197 and under5 == 128, (total_present, under5)
    assert total_present - under5 == 1069
    drawn = {k: v[3] for k, v in rows.items() if k not in ("Not stated", "None (eg too young to talk)")}
    df = pd.DataFrame([dict(geo_id="TK", geo_level="country", geo_name="Tokelau", source_category=k, count=n,
                            tier="measured", year=2016) for k, n in drawn.items()])
    assert df["count"].sum() == 1069 - 6 - 1
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {df['count'].sum():,} people aged 5+ present on census night")
    print(df[["source_category", "count"]].to_string(index=False))


if __name__ == "__main__":
    main()
