"""Oman, register end 2024: sources/om_build.py writes node ids straight into
data/normalized/om.csv, so this mapping is the identity, checked against the nodes it wrote.

No source asks Oman's residents a language. Omanis on `afroasiatic.omani_arabic` (oman1239);
expatriates on their nationality's language or home mix (sources/gulf_mix.py); the government
sector's "Other Arabs" on `afroasiatic.arabic`.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "om.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"om2024: {label!r} is not a node sources/om_build.py wrote")
    return label
