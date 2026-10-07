"""Brunei, 2021: sources/bn_build.py writes node ids straight into data/normalized/bn.csv, so this
mapping is the identity, checked against the nodes it wrote.

The census asked home language (E27) but never tabulated it. The base is Table A3, race by
district (Malays / Chinese / Others), split into languages with published estimates
(sources/bn.md):
  * Malays: Tutong, Kedayan, Brunei Dusun (Bisaya inside it), Belait and Lun Bawang (Murut) from
    speaker estimates on their home districts; the rest of each district Brunei Malay.
  * Chinese: one national mix, English 16% and Mandarin 30% (Asia Harvest), the rest Min Nan,
    Cantonese, Min Dong and Hakka by Joshua Project's Brunei figures.
  * Others: Iban 15,800 (Omniglot) over Belait, Tutong and Temburong; the rest foreign residents
    by nationality, each at origin_mix's home mix.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "bn.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"bn2021: {label!r} is not a node sources/bn_build.py wrote")
    return label
