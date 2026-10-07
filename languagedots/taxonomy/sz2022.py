"""Eswatini: Afrobarometer R5-R9 home-language answers (sources/sz_afro.py, sources/sz.md).
Spelling variants are merged in sz_afro.py's LABELS (siSwati/Siswati, Zulu/Isizulu, and Tonga
with Shangaan as Xitsonga). "Other" is an unnamed remainder of anything, so it sits on `other`.
"""
NAMES = {
    "siSwati": "nigercongo.bantu.nguni.siswati",
    "English": "indoeuropean.germanic.english",
    "Zulu": "nigercongo.bantu.nguni.zulu",
    "Shangaan": "nigercongo.bantu.tswa_ronga.tsonga",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"sz2022: unmapped label {label!r}")
