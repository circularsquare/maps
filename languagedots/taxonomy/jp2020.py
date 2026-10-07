"""Japan, 2020 census: sources/jp_census.py writes node ids straight into data/normalized/jp.csv,
so this mapping is the identity, checked against the nodes it wrote.

No census or survey asks Japan's residents a language. The nodes come from:
  * Japanese nationals: `japonic.japanese`, less the Ryukyuan speakers below.
  * the Ryukyuan languages (`japonic.ryukyuan.*`, taxonomy/tree.d/jp.txt): Okinawa from the
    prefecture's 2023 and 2024 shimakutuba surveys by region, each municipality on its
    traditional language; the Amami Islands from Ethnologue's speaker figures.
  * foreign residents: their nationality's language or language mix (sources/jp_census.py,
    sources/jp.md), less the share drawn as Japanese (Aichi 2022 survey); China, the
    Philippines, Vietnam and sixteen other origins at their home mixes, which bring in the
    nodes those countries' own mappings use.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "jp.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"jp2020: {label!r} is not a node sources/jp_census.py wrote")
    return label
