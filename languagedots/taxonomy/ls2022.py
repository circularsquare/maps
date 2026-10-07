"""Lesotho: Afrobarometer R4-R9 home-language answers (sources/ls_afro.py, sources/ls.md).

Sethepu is the Sesotho name for isiXhosa as Lesotho's Thembu-descended Xhosa speak it (Quthing,
Qacha's Nek): a name for the language, not a separate one, so it goes on Xhosa. Sephuthi is
Phuthi (Glottolog phut1246, under Nguni), a node of its own. "Other"/"Others" is an unnamed
remainder of anything, so it sits on `other`.
"""
NAMES = {
    "Sesotho": "nigercongo.bantu.sotho_tswana.sesotho",
    "Sethepu": "nigercongo.bantu.nguni.xhosa",
    "Sephuthi": "nigercongo.bantu.nguni.phuthi",
    "English": "indoeuropean.germanic.english",
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ls2022: unmapped label {label!r}")
