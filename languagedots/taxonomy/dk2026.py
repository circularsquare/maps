"""Denmark: sources/dk_dst.py writes node ids straight into data/normalized/dk.csv (Danish,
German, the Greenland- and Faroe-born's home mixes, and each country of origin's mix), so this
mapping is the identity, checked against the nodes it wrote.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "dk.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique()) \
    if _CSV.exists() else []


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"dk2026: {label!r} is not a node sources/dk_dst.py wrote")
    return label
