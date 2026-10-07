"""Iceland, Hagstofa MAN04203 (1 January 2026): sources/is_pop.py writes node ids straight into
data/normalized/is.csv, so this mapping is the identity, checked against the nodes it wrote.

No language question. Icelandic citizens on Icelandic; foreign citizens on their country's
language (fr_build.COUNTRY_LANG via fr2023.NAMES) or, for 22 multilingual origins, that country's
own drawn mix on this map, which brings in the nodes those countries' mappings use; stateless
and unspecified on `other` (sources/is.md).
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "is.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"is2026: {label!r} is not a node sources/is_pop.py wrote")
    return label
