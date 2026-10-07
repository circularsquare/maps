"""Sudan, speaker estimates (2022 base): sources/sd_estimate.py writes node ids straight into
data/normalized/sd.csv, so this mapping is the identity, checked against the nodes it wrote.

Replaces sd2022.py (the Afrobarometer / Arab Barometer mapping), retired 2026-10-05 when ask 019
allowed published estimates. Node choices (Glottolog codes in sources/sd_estimate.py's NODE):
  * Sudanese Arabic (suda1236): the remainder of every state.
  * Beja on `cushitic.beja`; Hausa, Sokoro under Chadic; Fulfulde on `atlantic.fulah`.
  * Nubian languages as separate leaves under Nilo-Saharan, as eg.txt has Nobiin and Kenzi:
    Nobiin, Dongolawi, Midob, and the Hill Nubian Ghulfan, Kadaru, Karko, Dilling, Dair, Wali.
  * Kordofanian (Heiban, Talodi, Rashad, Katla-Tima, Lafofa): a group under Niger-Congo, as
    most readers know them; Moro on au.txt's existing `nigercongo.moro`.
  * Kadugli-Krongo: a group `nilosaharan.kadu`, where it is usually placed.
  * Darfur and Blue Nile languages under their usual Nilo-Saharan branches; Ganza on et.txt's
    Mao leaf (Blue Nile Mao, Omotic); Ngambay on cf.txt's Sara leaf.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "sd.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"sd2026: {label!r} is not a node sources/sd_estimate.py wrote")
    return label
