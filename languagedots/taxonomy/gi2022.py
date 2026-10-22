"""Gibraltar, 2022 census Q12 "What is your main language?" (sources/gi_census.py, sources/gi.md).
Population aged 14 and over; English or a language written in.

Calls:
  - "Arabic" (806; 576 of them Gibraltarian, 160 Moroccan nationals) on Moroccan Arabic (Darija):
    Gibraltar's Arabic speakers are overwhelmingly the Moroccan community and its descendants, so
    the census's lumped "Arabic" is split to the variety they speak (AGENT_BRIEF, CLAUDE.md:
    lean towards splitting a lumped answer by knowledge, recorded). Reversing it puts them on the
    Arabic group.
  - "Chinese" on the Sinitic group (variety not named). "Other" (780 after the named languages
    below are taken out) is the unnamed remainder, on `other`.
  - The eight small languages (Portuguese, Polish, French, German, Hebrew, Russian, Chinese,
    Filipino) are bars on Figure 11, read to about 10 people (sources/gi_census.py).
  - "British Sign Language" (4) on `signlanguage.bsl`. Llanito, the local speech mixing English and
    Spanish, was not an option on the form and is not a row.
"""
NAMES = {
    "English": "indoeuropean.germanic.english",
    "Spanish": "indoeuropean.romance.spanish",
    "Arabic": "afroasiatic.darija",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Polish": "indoeuropean.slavic.west.polish",
    "French": "indoeuropean.romance.french",
    "German": "indoeuropean.germanic.continental.german",
    "Hebrew": "afroasiatic.hebrew",
    "Russian": "indoeuropean.slavic.east.russian",
    "Chinese": "sinotibetan.sinitic",
    "Filipino": "austronesian.philippine.filipino",
    "Other": "other",
    "British Sign Language": "signlanguage.bsl",
}


def resolve(label):
    return NAMES[label]
