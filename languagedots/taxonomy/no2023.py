"""Norway: sources/no_ssb.py writes node ids straight into data/normalized/no.csv (Norwegian,
North Sami, and each country background's origin mix), and sources/no_svalbard.py the same into
no_svalbard.csv, so this mapping is the identity, checked against the nodes they wrote.
"""
from pathlib import Path

import pandas as pd

_NORM = Path(__file__).resolve().parent.parent / "data" / "normalized"
NAMES = {}
EXTRA_NODES = sorted({n for f in ("no.csv", "no_svalbard.csv") if (_NORM / f).exists()
                      for n in pd.read_csv(_NORM / f, usecols=["source_category"])["source_category"]})


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"no2023: {label!r} is not a node sources/no_ssb.py or no_svalbard.py wrote")
    return label
