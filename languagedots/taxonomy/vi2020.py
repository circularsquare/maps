"""US Virgin Islands, 2020 Island Areas Census, PBG5 language spoken at home -> node.
sources/vi_census.py; keyed by the table's own cell labels, as data/normalized/vi.csv carries them.

CALLS (sources/vi.md says more):
  Speak only English: English. It holds Virgin Islands Creole English (Crucian, St Thomian),
    which the questionnaire gives no way to name apart from English.
  Speak Spanish: Spanish (Puerto Rican and Dominican, mostly on St Croix).
  Speak French, Haitian, or Cajun (7,101): `creole.french_based`, drawn as "language not named".
    The group holds French, Haitian Creole, Cajun and the Antillean Creole (Kweyol) of Dominica
    and St Lucia, which the Bureau codes with Haitian. In the USVI it is almost all creole: the
    same census counts 4,329 people born in Dominica, 2,881 in St Lucia and 2,397 in Haiti, and
    719 born anywhere in Europe. Strictly, French makes the narrowest node holding the group the
    root; the creole node is chosen because French speakers are a small part of it, and a
    root-level node would draw 9% of the territory as unclassified.
  Speak other languages (3,349): `other`. No breakdown is published at any level; it will hold
    Arabic (St Croix's Palestinian community), Indian languages, Papiamento, Dutch and more.
"""
NAMES = {
    "Speak only English": "indoeuropean.germanic.english",
    "Speak Spanish": "indoeuropean.romance.spanish",
    "Speak French, Haitian, or Cajun": "creole.french_based",
    "Speak other languages": "other",
}
EXTRA_NODES = []


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"vi2020: unmapped label {label!r}")
