"""Kuwait, 2021 census: sources/kw_build.py writes node ids straight into
data/normalized/kw.csv, so this mapping is the identity, checked against the nodes it wrote.

No source asks Kuwait's residents a language. Kuwaitis on `afroasiatic.gulf_arabic` (gulf1241,
shared with the UAE, Qatar and Bahrain); non-Kuwaitis on their nationality's language or home
mix (sources/gulf_mix.py); South Americans, whom no table names by country, on
`indoeuropean.romance`.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "kw.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"kw2021: {label!r} is not a node sources/kw_build.py wrote")
    return label
