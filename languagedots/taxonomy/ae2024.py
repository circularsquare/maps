"""United Arab Emirates, 2024: sources/ae_build.py writes node ids straight into
data/normalized/ae.csv, so this mapping is the identity, checked against the nodes it wrote.

No source asks UAE residents a language. Emiratis on `afroasiatic.gulf_arabic` (gulf1241, shared
with Kuwait, Qatar and Bahrain); non-Emiratis on their origin's language or home mix
(sources/gulf_mix.py), which brings in the nodes those countries' own mappings use; UN DESA's
unnamed `Others` on `other`.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "ae.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"ae2024: {label!r} is not a node sources/ae_build.py wrote")
    return label
