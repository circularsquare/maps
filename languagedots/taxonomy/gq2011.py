"""Equatorial Guinea: DHS 2011 ethnic group read as language (sources/gq_dhs.py, sources/gq.md).

Fang (fang1246), Bubi (Bube, bube1242), Annobonese (Fa d'Ambo, fada1250, a Portuguese-based
creole, beside Sao Tome's Forro), Bisio (Kwasio, kwas1243: the Bisio/Bujeba of the Litoral coast).
"Ndowe" is the coastal people speaking Kombe (Ngumbi, ngum1255), Benga (beng1282) and smaller
relatives; the DHS names only the people, so it gets one leaf of its own, not a group node.
"Extranjero" (foreign, no nationality given) and "Otro" go on `other`.
"""
NAMES = {
    "Fang": "nigercongo.bantu.fang",
    "Bubi": "nigercongo.bantu.bube",
    "Ndowe": "nigercongo.bantu.ndowe",
    "Bisio": "nigercongo.bantu.kwasio",
    "Annobones": "creole.portuguese_based.fadambo",
    "Extranjero": "other",
    "Otro": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"gq2011: unmapped label {label!r}")
