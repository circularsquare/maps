"""Mauritania: Afrobarometer R10 Q2 home language (sources/mr_afro.py, sources/mr.md).

"Arabe / Hassaniya" on Hassaniya (hass1238), as Mali's "Maure/Hasaniya"; Pulaar on Fula, as
Mali and Senegal draw it; Soninke, Wolof. Foreign residents' rows are node ids already (origin
mixes, Mali's census for the refugees), passed through as EXTRA_NODES.
"""
from pathlib import Path

import pandas as pd

NAMES = {
    "Hassaniya": "afroasiatic.hassaniya",
    "Pulaar": "nigercongo.atlantic.fulah",
    "Soninke": "nigercongo.mande.soninke",
    "Wolof": "nigercongo.atlantic.wolof",
}

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "mr.csv"
EXTRA_NODES = sorted(set(pd.read_csv(_CSV, usecols=["source_category"])["source_category"])
                     - set(NAMES)) if _CSV.exists() else []


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    if label in EXTRA_NODES:
        return label
    raise KeyError(f"mr2024: unmapped label {label!r}")
