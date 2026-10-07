"""Montserrat: sources/ms_census.py writes node ids straight into data/normalized/ms.csv (the
Leeward creole for the Montserrat-born, each foreign birthplace's node or origin mix), so this
mapping is the identity.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "ms.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique()) \
    if _CSV.exists() else []


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"ms2011: {label!r} is not a node sources/ms_census.py wrote")
    return label
