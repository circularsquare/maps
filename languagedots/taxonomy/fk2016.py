"""Falkland Islands, 2016 census Table 5a: language other than English spoken at home, by location
(sources/fk_census.py, sources/fk.md). Everyone else is an English first-language speaker.

Calls:
  - "Castellano" is Spanish (the Spanish name for it), merged with "Spanish".
  - "Bisaya/Cebuano", "Filipino" and "Tagalog" each get their node (all three printed); the
    report's "Pilipino" is their sum.
  - "Chinese" (2) on the Sinitic group, variety not named; "Gaelic" (2) on the Celtic group: the
    table does not say Irish or Scottish. "French Arabic" (1) on the Arabic group.
  - "Sri Lankan" (1) and "Swiss" (1) are nationalities, not languages (Sinhala or Tamil, Swiss
    German, French or Italian): drawn on `other`.
"""
NAMES = {
    "English": "indoeuropean.germanic.english",
    "Afrikaans": "indoeuropean.germanic.continental.afrikaans",
    "Bisaya/Cebuano": "austronesian.philippine.cebuano",
    "Cantonese": "sinotibetan.sinitic.cantonese",
    "Castellano": "indoeuropean.romance.spanish",
    "Chinese": "sinotibetan.sinitic",
    "Czech": "indoeuropean.slavic.west.czech",
    "Danish": "indoeuropean.germanic.north.danish",
    "Dutch": "indoeuropean.germanic.continental.dutch",
    "Filipino": "austronesian.philippine.filipino",
    "French": "indoeuropean.romance.french",
    "French Arabic": "afroasiatic.arabic",
    "Gaelic": "indoeuropean.celtic",
    "Georgian": "kartvelian.georgian",
    "German": "indoeuropean.germanic.continental.german",
    "Italian": "indoeuropean.romance.italian",
    "Japanese": "japonic.japanese",
    "Lithuanian": "indoeuropean.baltic.lithuanian",
    "Nepalese": "indoeuropean.indoaryan.pahari.eastern.nepali",
    "Polish": "indoeuropean.slavic.west.polish",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Romanian": "indoeuropean.romance.romanian",
    "Russian": "indoeuropean.slavic.east.russian",
    "Setswana": "nigercongo.bantu.sotho_tswana.setswana",
    "Shona": "nigercongo.bantu.shona",
    "Spanish": "indoeuropean.romance.spanish",
    "Sri Lankan": "other",
    "Swiss": "other",
    "Tagalog": "austronesian.philippine.tagalog",
    "Thai": "kradai.thai",
    "Urdu": "indoeuropean.indoaryan.central.urdu",
    "Welsh": "indoeuropean.celtic.welsh",
}


def resolve(label):
    return NAMES[label]
