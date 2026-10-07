"""Eritrea: EPHS 2010 ethnic group read as language (sources/er_ephs.py, sources/er.md).

Each of the nine groups has a language of its own: Tigrinya, Tigre, Saho, Afar, Bilen (Agaw),
Kunama, Nara (Nilo-Saharan; nara1262), Hedareb (the Beja of Eritrea, speaking Beja/To Bedawie),
Rashaida (Hijazi Arabic, drawn on sa.txt's Saudi Arabic node, which holds Hijazi). "Other" on
`other`.
"""
NAMES = {
    "Tigrigna": "afroasiatic.ethiosemitic.tigrinya",
    "Tigre": "afroasiatic.ethiosemitic.tigre",
    "Saho": "afroasiatic.cushitic.lowland.saho",
    "Afar": "afroasiatic.cushitic.lowland.afar",
    "Bilen": "afroasiatic.cushitic.agaw.bilen",
    "Hedarib": "afroasiatic.cushitic.beja",
    "Kunama": "nilosaharan.kunama",
    "Nara": "nilosaharan.nara",
    "Rashaida": "afroasiatic.saudi_arabic",
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"er2010: unmapped label {label!r}")
