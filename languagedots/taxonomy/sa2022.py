"""Saudi Arabia, 2022 census: sources/sa_census.py writes node ids straight into
data/normalized/sa.csv, so this mapping is the identity, checked against the nodes it wrote.

No census or survey asks Saudi residents a language. The nodes come from:
  * Saudi citizens: `afroasiatic.saudi_arabic`, one node for the Najdi, Hejazi, Gulf and
    southern varieties citizens speak (Glottolog najd1235, hija1235, gulf1241, and the
    Yemeni-type dialects of Asir and Jazan), which nothing counts apart; a node of its own beside
    `arabic` as Iraq's and Sudan's Arabic are.
  * non-Saudis: their nationality's language or language mix (sources/sa_census.py, sources/sa.md):
    Yemeni, Egyptian and Levantine Arabic as nodes of their own beside `arabic`; India, Pakistan
    and twelve other origins at their home mixes, which bring in the nodes those countries'
    own mappings use.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "sa.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"sa2022: {label!r} is not a node sources/sa_census.py wrote")
    return label
