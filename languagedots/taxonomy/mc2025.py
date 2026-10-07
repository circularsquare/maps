"""Monaco: sources/mc_census.py writes node ids straight into data/normalized/mc.csv (French for
Monegasques, each nationality's origin mix, other), so this mapping is the identity.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "mc.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique()) \
    if _CSV.exists() else []


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"mc2025: {label!r} is not a node sources/mc_census.py wrote")
    return label
