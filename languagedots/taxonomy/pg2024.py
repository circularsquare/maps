"""Papua New Guinea, 2024 people: sources/pg_build.py writes node ids straight into
data/normalized/pg.csv, so this mapping is the identity, checked against the nodes it wrote.

No PNG census publishes first languages by area. The map is a language-area model: each
province's people shared among the languages whose Glottolog point falls in it, by Joshua
Project's (Ethnologue-based) population figures. The nodes, and taxonomy/tree.d/pg.txt, are
generated from Glottolog's classification; sources/pg.md is the record.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "pg.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"pg2024: {label!r} is not a node sources/pg_build.py wrote")
    return label
