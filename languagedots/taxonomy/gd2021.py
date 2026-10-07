"""Grenada, 2021 census ethnicity by parish (sources/gd_census.py, sources/gd.md). No language
question: White/Caucasian Grenadians on English, everyone else on Grenadian Creole (Glottolog
gren1247; node defined in tree.d/bb.txt, repeated in tree.d/gd.txt)."""
NAMES = {
    "Everyone else": "creole.english_based.grenadian",
    "White/Caucasian": "indoeuropean.germanic.english",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"gd2021: unmapped label {label!r}")
