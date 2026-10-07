"""Uzbekistan 2026 census, native language by region (sources/uz_census.py) -> node.

Keyed by data/normalized/uz.csv's `source_category`, the table's eight columns in English (the Uzbek
edition's o'zbek, qoraqalpoq, qozoq, tojik, qirg'iz, rus, turkman, boshqa).

CALLS.
  Every named column is a language with a node other countries already draw; nothing new.
  Tajik: `indoeuropean.iranian.tajik`, as Kyrgyzstan's and the migrant tables'. The census does
    not split Tajik by variety, and the Pamiri languages (whose speakers in Uzbekistan are few)
    have no column; anyone who named one is in "other".
  other (boshqa): `other`. It is every language the census did not print a column for (Tatar,
    Korean, Ukrainian, Uyghur, Azerbaijani, Armenian and the rest, 88,244 people, 0.23%), so the
    narrowest node holding all of it is the root `other`. Not a group under Turkic: Korean,
    Ukrainian and Armenian are in it too.
"""
NAMES = {
    "Uzbek": "turkic.uzbek",
    "Karakalpak": "turkic.karakalpak",
    "Kazakh": "turkic.kazakh",
    "Tajik": "indoeuropean.iranian.tajik",
    "Kyrgyz": "turkic.kyrgyz",
    "Russian": "indoeuropean.slavic.east.russian",
    "Turkmen": "turkic.turkmen",
    "other": "other",
}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"uz2026: unmapped label {label!r}")
    return NAMES[label]
