"""Tuvalu 2022 census, ethnicity (Figure 6) read as language, Nui apart (sources/tv_census.py,
sources/tv.md). "Not stated" is not drawn (None).
"""
NAMES = {
    "Nui (Nuian)": "austronesian.oceanic.gilbertese",   # Nuian, a Gilbertese dialect
    "Tuvaluan": "austronesian.oceanic.tuvaluan",
    "Tuvaluan/I-Kiribati": "austronesian.oceanic.tuvaluan",
    "Tuvaluan/Other": "austronesian.oceanic.tuvaluan",
    "Other": "other",
    "Not stated": None,
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"tv2022: unmapped label {label!r}")
