"""Jamaica: Jamaican Creole (Patwa) and English from the Jamaican Language Unit's Language
Competence Survey of Jamaica (2006), as national shares on the 2011 census parish populations
-> data/normalized/jm.csv.

    python sources/jm_jlu.py

NO CENSUS LANGUAGE QUESTION (2011 asks ethnic origin and religion; scout 2026-10-05).

THE SHARE. JLU, *The Language Competence Survey of Jamaica: Data Analysis* (UWI Mona, September
2007; data/raw/jm/jlu_language_competence_survey_2006.pdf), Table 4: of 1,000 adults stratified
by region, urban/rural, age and sex, 17.1% spoke only English in the interview, 36.5% only
Jamaican, 46.4% demonstrated both. Jamaican is the language acquired at home by nearly all
Jamaicans and English is acquired at school, so the bilinguals are drawn with the Jamaican
monolinguals as Jamaican Creole first-language speakers (82.9%), and the English monolinguals
as English (17.1%). sources/jm.md argues it; the 2005 Language Attitude Survey's self-declared
"English only" (10.9%) is the lower bound recorded beside it.

THE POPULATION. The 2011 census parish totals as religiondots normalised them (its jm.csv,
parish rows, every religion category summed incl. not stated), read only.

CHECKS: 14 parishes; 2.67-2.70 million people; the two shares pinned to the report's counts.
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

OUT = HERE / "data" / "normalized" / "jm.csv"
# Table 4, counts of 1,000
ENGLISH_ONLY, PATWA_ONLY, BOTH = 171, 365, 464
SHARES = {"Jamaican (Patwa)": (PATWA_ONLY + BOTH) / 1000, "English": ENGLISH_ONLY / 1000}


def main():
    assert ENGLISH_ONLY + PATWA_ONLY + BOTH == 1000
    d = pd.read_csv(RD / "data" / "normalized" / "jm.csv", dtype={"geo_id": str})
    d = d[d["geo_level"] == "parish"]
    pop = d.groupby(["geo_id", "geo_name"], as_index=False)["count"].sum()
    assert len(pop) == 14, len(pop)
    total = pop["count"].sum()
    assert 2_670_000 < total < 2_700_000, total
    rows = []
    for _, r in pop.iterrows():
        eng = round(r["count"] * SHARES["English"])
        for lab, n in (("English", eng), ("Jamaican (Patwa)", r["count"] - eng)):
            rows.append(dict(geo_id=r["geo_id"], geo_level="parish",
                             geo_name=r["geo_name"].title(), source_category=lab, count=n,
                             tier="modelled", source_id="jlu_lcs_2006_table4", year=2006,
                             note=f"national share {SHARES[lab]:.3f}; 2011 census parish "
                                  f"population {r['count']}"))
    df = pd.DataFrame(rows)
    assert df["count"].sum() == total
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT}: {len(df)} rows, 14 parishes, {total:,} people")


if __name__ == "__main__":
    main()
