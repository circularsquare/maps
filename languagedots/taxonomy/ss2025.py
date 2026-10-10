"""South Sudan, Jonglei, Unity and Upper Nile, 2025: sources/ss_homeland.py writes node ids straight
into data/normalized/ss_homeland.csv, so this mapping is the identity, checked against the nodes it
wrote. Each county on its homeland group's language (CSRF county profiles), Malakal split by
JICA's 2014 household survey, Kacipo-Balesi and Opo from Joshua Project (sources/ss.md section 7).

  * Nuer (nuer1246), Dinka (one node, as the 2015 survey's card has it), Shilluk (shil1265),
    Murle (murl1244), Anuak (anua1242), Mabaan (maba1273), Kacipo-Balesi (kaci1244), Opo
    (opuu1239).
  * Malakal's unnamed "others" on `africa_other`.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "ss_homeland.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"ss2025: {label!r} is not a node sources/ss_homeland.py wrote")
    return label
