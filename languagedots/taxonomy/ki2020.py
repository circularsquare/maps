"""Kiribati, 2020 census island populations (sources/ki_pop.py, sources/ki.md). No home-language
question: everyone drawn as Gilbertese (Kiribati language)."""
NAMES = {"Gilbertese": "austronesian.oceanic.gilbertese"}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ki2020: unmapped label {label!r}")
