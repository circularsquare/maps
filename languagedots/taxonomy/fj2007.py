"""Fiji 2007 census, Fijian / Indian / Other per province, read as language (sources/fj_census.py,
sources/fj.md).
"""
NAMES = {
    "Fijian": "austronesian.oceanic.fijian",
    "Indian": "indoeuropean.indoaryan.eastcentral.fiji_hindi",
    "Other (Rotuma)": "austronesian.oceanic.rotuman",
    "Other (Cakaudrove)": "austronesian.oceanic.gilbertese",   # Rabi's Banabans
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"fj2007: unmapped label {label!r}")
