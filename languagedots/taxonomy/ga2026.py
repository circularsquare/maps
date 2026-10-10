"""Gabon: Afrobarometer R6-R9 home-language answers (sources/ga_afro.py, sources/ga.md). Card labels merged only for spelling ("Kélé"/"Kélè"). Glottolog: Punu punu1239,
Njebi njeb1242 (the card's Nzébi), Mbere-Mbamba mber1257 (the card's Mbédè, Haut-Ogooué), Kota
(Gabon) kota1274, Tsogo tsog1243, Myene myen1241, Lumbu lumb1249 (Baloumbou; not Zambia's Lumbu),
Kélé kele1257, Sangu (Gabon) sang1333 (Masangu; not Tanzania's Sangu), Sira sira1266 (Eshira),
Vumbu vumb1238 (Bavungu), Vili vili1238. "Bateke" on cd.txt's Teke leaf: the card does not say
which Teke, and Gabon's are Latege and Teke-Tsaayi on the Congo border.

Foreign residents' rows are node ids already (origin mixes), passed through as EXTRA_NODES.
"""
from pathlib import Path

import pandas as pd

B = "nigercongo.bantu"
NAMES = {
    "French": "indoeuropean.romance.french",
    "Fang": f"{B}.fang",
    "Punu": f"{B}.punu",
    "Nzebi": f"{B}.nzebi",
    "Mbede": f"{B}.mbere",
    "Kota": f"{B}.kota",   # Congo's node: same language (kota1274); merged 2026-10-08
    "Tsogho": f"{B}.tsogo",
    "Myene": f"{B}.myene",
    "Baloumbou": f"{B}.lumbu_gabon",
    "Kele": f"{B}.kele_gabon",
    "Masangu": f"{B}.sangu_gabon",
    "Eshira": f"{B}.sira",
    "Bateke": f"{B}.teke",
    "Bavungu": f"{B}.vumbu",
    "Vili": f"{B}.vili",
    "English": "indoeuropean.germanic.english",
    "Other": "other",
}

_CSV = Path(__file__).resolve().parent.parent / "data" / "normalized" / "ga.csv"
EXTRA_NODES = sorted(set(pd.read_csv(_CSV, usecols=["source_category"])["source_category"])
                     - set(NAMES)) if _CSV.exists() else []


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    if label in EXTRA_NODES:
        return label
    raise KeyError(f"ga2026: unmapped label {label!r}")
