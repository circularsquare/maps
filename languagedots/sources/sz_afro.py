"""Eswatini: home language from five pooled Afrobarometer rounds (R5-R9, 2011-2022), by region,
on the 2017 census region populations -> data/normalized/sz.csv.

    python sources/sz_afro.py

NO LANGUAGE QUESTION IN THE CENSUS. The 2007 questionnaire's P15 asks literacy only (read/write/
understand siSwati, English...); the 2017 volumes have no language table (scout 2026-10-05). So
this is the survey route of AGENT_BRIEF §2: shares from the Afrobarometer's home-language
question, one answer, times a population base, every row `modelled`. Swaziland was not in R4.

6,000 respondents, pooled: siSwati 96.05% (R7 spells it "Siswati"), English 2.94%, Zulu 0.62%
(R5 "Isizulu"), Other 0.20%, Shangaan 0.11%, Portuguese 0.02%, Tonga 0.01%, refused 0.04%
(dropped). SHANGAAN is Xitsonga's name in Mozambique and Eswatini; TONGA, one R5 respondent in
Hhohho, is put with it as "Thonga", the older name for the same people (there is no Zambian or
Malawian Tonga community in Eswatini that a source counts).

POPULATION. 2017 census region populations (Volume 3, Table 5.2.2): Hhohho 320,651, Manzini
355,945, Shiselweni 204,111, Lubombo 212,531, 1,093,238 in all. Checked against religiondots'
normalized sz.csv (read-only), whose four regions sum to the same figures.
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
from rdlink import RD  # noqa: E402
from mono_afro import extract, unit_counts  # noqa: E402

OUT = HERE / "data" / "normalized" / "sz.csv"
POP_2017 = {"SZ1": ("Hhohho", 320_651), "SZ2": ("Manzini", 355_945),
            "SZ3": ("Shiselweni", 204_111), "SZ4": ("Lubombo", 212_531)}
NORM = {"hhohho": "SZ1", "manzini": "SZ2", "shiselweni": "SZ3", "lubombo": "SZ4"}
LABELS = {"siSwati": "siSwati", "Siswati": "siSwati", "English": "English", "Zulu": "Zulu",
          "Isizulu": "Zulu", "Shangaan": "Shangaan", "Tonga": "Shangaan",
          "Portuguese": "Portuguese", "Other": "Other", "Refused To Answer": None}


def main():
    rd = pd.read_csv(RD / "data" / "normalized" / "sz.csv", dtype={"geo_id": str})
    tot = rd[rd["source_category"] == "Total"].set_index("geo_id")["count"]   # Table 5.2.2
    for gid, u in (("SZ01", "SZ1"), ("SZ02", "SZ2"), ("SZ03", "SZ3"), ("SZ04", "SZ4")):
        assert int(tot[gid]) == POP_2017[u][1], (gid, int(tot[gid]))
    assert sum(n for _, n in POP_2017.values()) == 1_093_238
    # Anita's ruling on ask 018 (2026-10-05): English, the lingua franca, at R7's mother-tongue
    # question (Q2A), its national share in every region (7 answers: too few for regions)
    a = extract({"eswatini", "swaziland"}, r7q="Q2A")
    assert len(a) == 6000 and sorted(a["round"].unique()) == [5, 6, 7, 8, 9], (len(a),)
    df = unit_counts(a, NORM, LABELS, POP_2017, "afrobarometer_r5_r9_eswatini", "2011-2022",
                     "region", lf=["English"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {df['geo_id'].nunique()} regions, {df['count'].sum():,} people")
    print((nat / nat.sum() * 100).round(2).to_string())
    print(df[["geo_name", "source_category", "count", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
