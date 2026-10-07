"""Bahrain, 2020 census: sources/bh_build.py writes node ids straight into
data/normalized/bh.csv, so this mapping is the identity, checked against the nodes it wrote.

No source asks Bahrain's residents a language. Shia Bahrainis on `afroasiatic.baharna_arabic`
(baha1259), other Bahrainis on `afroasiatic.gulf_arabic` (gulf1241, shared with the UAE, Kuwait
and Qatar); non-Bahrainis on their origin's language or home mix (sources/gulf_mix.py); the
census's `Others` nationality group on `other`.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "bh.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"bh2020: {label!r} is not a node sources/bh_build.py wrote")
    return label
