"""South Korea, 2020: sources/kr_census.py writes node ids straight into data/normalized/kr.csv,
so this mapping is the identity, checked against the nodes it wrote.

No Korean census or survey asks a language. The nodes come from:
  * Korean nationals: `koreanic.korean`; 7,500 of them in Jeju on `koreanic.jejueo` (Glottolog
    jeju1234, Koreanic), from the Jejueo Project's speaker estimate (sources/kr.md §3).
  * foreign nationals: their nationality's language or home mix (sources/kr_census.py): Korean-
    Chinese on Korean, Russian Koreans and Central Asian Koryo-saram on Russian, unnamed
    remainders on `other` or `africa_other`; home mixes bring in the nodes those countries' own
    mappings use.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "kr.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"kr2020: {label!r} is not a node sources/kr_census.py wrote")
    return label
