"""Guyana, 2012 census ethnic groups with MICS6 2019-20's language of the household head
(sources/gy_mics.py) -> node. The labels are gy_mics.py's. Replaces gy2012.py's flat 20%
Amerindian retention (2026-10-09); sources/gy.md §0.

  "Creole (English answer)"           Guyanese Creole (creo1235). Census groups read as language
                                      (as gy2012.py), less MICS's other answers; MICS's "English"
                                      is the answer for 99.7% of non-Amerindian-headed persons.
  "English (white Guyanese)"          English, census White (as bb, tt).
  "Indigenous language"               americas_other: MICS has one "indigenous language" answer,
                                      covering Arawakan, Cariban and Warao languages (Makushi,
                                      Wapishana, Akawaio, Patamona, Arecuna, Carib, Arawak,
                                      Warao), so the narrowest node holding all of them.
  "Other language (Amerindian head)"  americas_other too: "other" with an Amerindian head, nearly
                                      all in Upper Mazaruni clusters beside the indigenous
                                      answers (gy_mics.py AM_OTHER).
  "Spanish", "Portuguese"             as named.
  "Other language"                    `other` (non-Amerindian heads; 3 of 7 households Chinese).
"""
NAMES = {
    "Creole (English answer)": "creole.english_based.guyanese",
    "English (white Guyanese)": "indoeuropean.germanic.english",
    "Indigenous language": "americas_other",
    "Other language (Amerindian head)": "americas_other",
    "Spanish": "indoeuropean.romance.spanish",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Other language": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"gy2019: unmapped label {label!r}")
