"""Qatar, 2020 census: sources/qa_build.py writes node ids straight into
data/normalized/qa.csv, so this mapping is the identity, checked against the nodes it wrote.

No source asks Qatar's residents a language. Qataris (estimated) on `afroasiatic.gulf_arabic`
(gulf1241, shared with the UAE, Kuwait and Bahrain); non-Qataris on their origin's language or
home mix (sources/gulf_mix.py); UN DESA's unnamed `Others` on `other`.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "qa.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"qa2020: {label!r} is not a node sources/qa_build.py wrote")
    return label
