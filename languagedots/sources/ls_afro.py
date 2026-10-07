"""Lesotho: home language from six pooled Afrobarometer rounds (R4-R9, 2008-2022), by district,
on the 2016 census district populations -> data/normalized/ls.csv.

    python sources/ls_afro.py

NO LANGUAGE QUESTION IN THE CENSUS. The 2016 census (IHSN catalog 8293) has no language or
ethnicity item. So this is the survey route of AGENT_BRIEF §2: shares from the Afrobarometer's
home-language question (R4-R6 "Which language is your home language?", R7-R9 "Language spoken
in home"), one answer, times a population base, every row `modelled`.

7,197 respondents, pooled: Sesotho 98.19%, Sethepu 0.92%, English 0.48%, Sephuthi 0.33%, other
0.08%. SETHEPU is the Sesotho name for isiXhosa as spoken by Lesotho's Thembu-descended Xhosa
(Quthing, Qacha's Nek), so it is put on Xhosa. SEPHUTHI is Phuthi, a Nguni language of Quthing
(Glottolog phut1246, under Nguni). R4's "Others" and R7's "Other" are one answer.

POPULATION. The 2016 census district counts, religiondots' ls_lookup.csv (read-only), 2,007,201
people on 10 districts. REGION -> district is religiondots' sources/ls.py NORM.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD_GEO  # noqa: E402
from mono_afro import extract, unit_counts  # noqa: E402

OUT = HERE / "data" / "normalized" / "ls.csv"
CENSUS_2016 = 2_007_201
NORM = {
    "maseru": "LSA", "buthabuthe": "LSB", "bothabothe": "LSB", "buthebuthe": "LSB",
    "leribe": "LSC", "berea": "LSD", "mafeteng": "LSE", "mohaleshoek": "LSF", "quthing": "LSG",
    "qachasnek": "LSH", "mokhotlong": "LSJ", "thabatseka": "LSK",
}
LABELS = {"Sesotho": "Sesotho", "Sethepu": "Sethepu", "Sephuthi": "Sephuthi",
          "English": "English", "Other": "Other", "Others": "Other"}


def main():
    # Anita's ruling on ask 018 (2026-10-05): English, the lingua franca, at R7's mother-tongue
    # question (Q2A), its national share in every district (4 answers: too few for districts)
    a = extract({"lesotho"}, r7q="Q2A")
    assert len(a) == 7197 and sorted(a["round"].unique()) == [4, 5, 6, 7, 8, 9], (len(a),)
    lut = pd.read_csv(RD_GEO / "ls" / "ls_lookup.csv", dtype={"geo_id": str})
    assert len(lut) == 10 and int(lut["pop"].sum()) == CENSUS_2016
    pop = {r.geo_id: (r.name, int(r.pop)) for r in lut.itertuples()}
    df = unit_counts(a, NORM, LABELS, pop, "afrobarometer_r4_r9_lesotho", "2008-2022",
                     "district", lf=["English"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {df['geo_id'].nunique()} districts, {df['count'].sum():,} people")
    print((nat / nat.sum() * 100).round(2).to_string())
    print(df[df["source_category"] != "Sesotho"][["geo_name", "source_category", "count",
                                                  "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
