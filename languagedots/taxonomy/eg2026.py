"""Egypt, 2026: sources/eg_build.py writes node ids straight into data/normalized/eg.csv, so this
mapping is the identity, checked against the nodes it wrote.

No census or survey counts Egypt's languages. The nodes come from (sources/eg.md):
  * Egyptian Arabic (egyp1253) and Sa'idi Arabic (said1239): the remainder of each governorate,
    Sa'idi in Minya and the governorates south of it plus New Valley, beside `arabic` as the
    other national varieties are.
  * Eastern Egyptian Bedawi Arabic (east2690), Sinai; Libyan Arabic (liby1240, whose dialect
    Western Egyptian Bedawi, west2774, the Awlad Ali speak), Matrouh.
  * Siwi (siwi1239), Beja (beja1238), Nobiin (nobi1240), Kenzi (kenu1243, Kenuzi-Dongola
    kenu1236), Domari (doma1258): Ethnologue speaker figures on their home governorates.
  * Refugees: UNHCR's registrations by nationality, Sudan and South Sudan at their drawn mixes.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "eg.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"eg2026: {label!r} is not a node sources/eg_build.py wrote")
    return label
