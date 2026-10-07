"""Djibouti, RGPH-3 2024 (Tome 4, Tableau 7): sources/dj_rgph.py writes node ids straight into
data/normalized/dj.csv, so this mapping is the identity, checked against the nodes it wrote.

The census's labels and their nodes (sources/dj.md):
  * Somali (soma1255), Afar (afar1241), Oromo: Lowland East Cushitic, as et.txt has them.
  * Amharique: Amharic. Oromo and Amharic are mostly Ethiopian residents' languages.
  * Langue des signes: `signlanguage`. Autres langues non listees: `other`.
  * Francais, Anglais, Arabe: learned languages, folded (AGENT_BRIEF section 2).
  * Native Arabic (not a census label): Joshua Project's "Arab, Yemeni" on `yemeni_arabic`
    (Ta'izzi-Adeni, taiz1242, the variety it names) and "Arab, Omani" on the generic `arabic`
    leaf.
"""
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "dj.csv"
NAMES = {}
EXTRA_NODES = sorted(pd.read_csv(_CSV, usecols=["source_category"])["source_category"].unique())


def resolve(label):
    if label not in EXTRA_NODES:
        raise KeyError(f"dj2024: {label!r} is not a node sources/dj_rgph.py wrote")
    return label
