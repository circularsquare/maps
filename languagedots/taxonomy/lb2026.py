"""Lebanon: sources/lb_build.py writes node ids straight into data/normalized/lb.csv, so this
mapping is the identity, checked against the nodes it wrote.

Lebanese: Arab Barometer II-IV first language and WVS 7 language at home, pooled per mohafaza:
"Arabic" -> Levantine Arabic (nort3139); Armenian ("Armenian; Hayeren") -> Armenian; French,
English, Spanish, Ukrainian, Kurdish as answered; "Syriac" -> afroasiatic.syriac. Syrians and
Palestinians (OCHA 2026) on Levantine Arabic.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "lb.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"lb2026: {label!r} is not a node sources/lb_build.py wrote")
    return label
