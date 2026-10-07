"""Turkmenistan 2022 census, mother tongue by velayat (sources/tm_census.py) -> node.

Keyed by data/normalized/tm.csv's `source_category`, the ten "considered as their mother tongue"
columns of section 4's tables 4.9-4.29, as the English edition prints them.

CALLS.
  Every named column is a language other countries already draw; nothing new.
  Baloch: `indoeuropean.iranian.balochi` (tree.txt's Balochi). The Baloch of Mary's Murgab and
    Tejen oases speak Western (Rakhshani) Balochi; the tree has one Balochi leaf and the census
    one column, so no split.
  other languages: `other`. It is every language without a column, and the nationality rows say
    what is in it: Persians 7,190, Afghans 1,796, Karakalpaks 1,613, Lezgins 1,241, Turkmens 888,
    Kurds 856, Balochi 576, Koreans 556, Uighurs 420, Turkish 359 and 47 more nationalities
    (17,596 people, 0.25%). Persian, Pashto, Karakalpak, Lezgian, Kurdish and Korean are all in
    there, so the narrowest node holding it is the root `other`. Not guessed into Persian by the
    nationality: the census does not name the language.
"""
NAMES = {
    "Turkmen": "turkic.turkmen",
    "Russian": "indoeuropean.slavic.east.russian",
    "Ukrainian": "indoeuropean.slavic.east.ukrainian",
    "Uzbek": "turkic.uzbek",
    "Kazakh": "turkic.kazakh",
    "Tatar": "turkic.tatar",
    "Armenian": "indoeuropean.armenian.armenian",
    "Azerbaijani": "turkic.azerbaijani",
    "Baloch": "indoeuropean.iranian.balochi",
    "other languages": "other",
}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"tm2022: unmapped label {label!r}")
    return NAMES[label]
