"""Falkland Islands: language spoken at home, 2016 census, by location -> data/normalized/fk.csv.

    python sources/fk_census.py

THE QUESTION. Falkland Islands Census 2016 asked every person their first language and, for the
people whose first language is not English, which language(s) they speak at home and how well they
speak English. The report summarises it as "English is the first language for 85 per cent of the
population. 496 people in the Islands speak a language other than English in their homes".

THE TABLES. Falkland Islands Government, *Falkland Islands Census 2016, Full Report*
(religiondots' copy, data/raw/terr/, read in place), the statistical annex:
  Table 5a (PDF p. 76), Population by language other than English, sex and location: 31 languages
  by Stanley, Camp and the Mount Pleasant Complex (MPC), the same three locations religiondots
  uses. The column sums are 561 mentions: 466 Stanley, 56 Camp, 39 MPC.
  Table 4a/14a (pp. 74-75) give the population by location: 2,458, 381, 359 = 3,198.
  Table 4.2 (p. 31) groups the same 561 mentions (Spanish 325, Shona 73, Pilipino 64, French 26,
  Italian 10, German 7, other 56), and Table 4.3/5b the 496 people.

THE DRAWING. 561 mentions by 496 people: about 65 people named two languages. Within each
location the mentions are scaled by 496/561 (people per mention, the same at every location since
the report gives no persons by location), everyone else is a first-language English speaker. All
rows `derived` (multi-answer scaling, AGENT_BRIEF §2). The 2016 report is used because it is the
one the census office makes available to this project; the 2021 census (3,662 usual residents)
exists but its report could not be reached (sources/fk.md).

CHECKS (asserted): Table 5a's rows sum to 561 mentions, by location 466 / 56 / 39; Table 4.2's
groups (Spanish, Shona, Pilipino = Filipino + Tagalog + Bisaya/Cebuano, French, Italian, German)
equal the sums of Table 5a's rows; the three locations sum to 3,198 (page 74 total row); in every
location the people with another language are fewer than its population; the 496 is Table 5b's
total.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

PDF = RD / "data" / "raw" / "terr" / "Falkland_Islands_Census_2016_-_Full_Report.pdf"
OUT = HERE / "data" / "normalized" / "fk.csv"
LOCS = {"Stanley": "FK-STANLEY", "Camp": "FK-CAMP", "MPC": "FK-MOUNT-PLEASANT-COMPLEX"}
POP = {"Stanley": 2458, "Camp": 381, "MPC": 359}
PERSONS, MENTIONS = 496, 561


def table5a(doc):
    t = doc[75].get_text()
    a, b = t.index("Table 5a"), t.index("Table 5b")
    lines = [x.strip() for x in t[a:b].split("\n") if x.strip()]
    i = lines.index("Afrikaans")
    rows = {}
    while i < len(lines):
        vals = lines[i + 1:i + 13]
        if len(vals) < 12 or not all(re.fullmatch(r"\d+", v) for v in vals):
            break
        rows[lines[i]] = [int(v) for v in vals]
        i += 13
    return rows


def main():
    import fitz
    doc = fitz.open(PDF)
    rows = table5a(doc)
    assert len(rows) == 31, len(rows)
    A = pd.DataFrame(rows, index=["T", "TM", "TF", "S", "SM", "SF", "C", "CM", "CF", "P", "PM", "PF"]).T
    assert (A["TM"] + A["TF"] == A["T"]).all() and (A["T"] == A["S"] + A["C"] + A["P"]).all()
    assert A["T"].sum() == MENTIONS and (A["S"].sum(), A["C"].sum(), A["P"].sum()) == (466, 56, 39)
    # Table 4.2 (p. 31)
    t42 = {"Spanish": 325, "Shona": 73, "Pilipino": 64, "French": 26, "Italian": 10, "German": 7, "Other": 56}
    pil = A.loc[["Filipino", "Tagalog", "Bisaya/Cebuano"], "T"].sum()
    other = MENTIONS - sum(v for k, v in t42.items() if k != "Other")
    assert A.loc["Spanish", "T"] == 325 and A.loc["Shona", "T"] == 73 and pil == 64
    assert A.loc["French", "T"] == 26 and A.loc["Italian", "T"] == 10 and A.loc["German", "T"] == 7
    assert other == 56
    tx = doc[30].get_text()
    assert "496 people" in tx and "Table 4.2" in tx
    t4a = doc[73].get_text()
    assert re.search(r"Total\s+3,198\s+1,687\s+1,511\s+2,458\s+1,261\s+1,197\s+381\s+188\s+193\s+359", t4a), "Table 4a totals"
    assert sum(POP.values()) == 3198
    out = []
    f = PERSONS / MENTIONS
    for loc, col in (("Stanley", "S"), ("Camp", "C"), ("MPC", "P")):
        people = {lang: A.loc[lang, col] * f for lang in A.index if A.loc[lang, col] > 0}
        other_total = sum(people.values())
        assert other_total < POP[loc]
        for lang, n in people.items():
            out.append(dict(geo_id=LOCS[loc], geo_level="location", geo_name=loc, source_category=lang, count=n))
        out.append(dict(geo_id=LOCS[loc], geo_level="location", geo_name=loc, source_category="English",
                        count=POP[loc] - other_total))
    df = pd.DataFrame(out)
    df["tier"] = "derived"
    df["year"] = 2016
    assert abs(df["count"].sum() - 3198) < 1e-6
    assert abs(df.loc[df["source_category"] != "English", "count"].sum() - PERSONS) < 1e-6
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, {df['count'].sum():,.0f} people in 3 locations "
          f"({PERSONS} with a language other than English at home)")
    for loc in LOCS:
        d = df[df["geo_name"] == loc]
        e = d.loc[d["source_category"] == "English", "count"].iloc[0]
        print(f"    {loc:8s} {POP[loc]:>5,} people, English {e:,.0f} ({e / POP[loc]:.1%}); "
              f"Spanish {d.loc[d['source_category'] == 'Spanish', 'count'].sum():,.0f}")


if __name__ == "__main__":
    main()
