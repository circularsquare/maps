"""Maldives, 2022 census: sources/mv_census.py writes node ids straight into
data/normalized/mv.csv, so this mapping is the identity, checked against the nodes it wrote.

No language question. Maldivians on Dhivehi; foreign residents by nationality (migration report
Table 6.1) at their countries' drawn mixes on this map (India, Sri Lanka, Bangladesh, the
Philippines, Indonesia, Nepal), which bring in those countries' nodes; "Others" on `other`.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "mv.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"mv2022: {label!r} is not a node sources/mv_census.py wrote")
    return label
