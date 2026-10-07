"""Nauru: sources/nr_census.py writes node ids straight into data/normalized/nr.csv (Nauruan,
Kosraean, other, and each foreign ethnicity's origin mix), so this mapping is the identity.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "nr.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique()) \
    if _CSV.exists() else []


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"nr2021: {label!r} is not a node sources/nr_census.py wrote")
    return label
