"""Vanuatu, 2020 census, first language learnt to speak (Table 6.16) -> node. Keyed by the
categories sources/vu_census.py writes.

NAMED AS LANGUAGES: English, French, Bislama, each on the node other countries use (Bislama is
au.txt's, under English-based creoles).

NAMED AS A GROUP: "Indigenous (Vernacular)" is every indigenous language of Vanuatu in one answer,
82.6% of those asked. It sits on `austronesian.oceanic.vanuatu`, the narrowest node holding all of
them (tree.d/vu.txt says why that node and not a Glottolog subgroup), and is drawn washed out as
"language not named", which is what it is.

NOT DRAWN:
  * Not stated: 3 people.
  * Not asked (speaks no indigenous language): 29,448 people aged 3+, 10.9%, a quarter of the
    towns. The questionnaire put E10 (first language) only to people who said at E9 they can
    speak an indigenous language. These people are the census's own complement, and their first
    language is almost all Bislama, English or French, but the table does not say which. Drawing
    them on Bislama would be a proxy, which is Anita's to allow; countries/vu.py has it behind
    DRAW_NOT_ASKED, off.
  * Total aged 3+: the universe (Table 6.17), not an answer.
"""
NAMES = {
    "English": "indoeuropean.germanic.english",
    "French": "indoeuropean.romance.french",
    "Bislama": "creole.english_based.bislama",
    "Indigenous (Vernacular)": "austronesian.oceanic.vanuatu",
    "Not stated": None,
    "Not asked (speaks no indigenous language)": None,
    "Total aged 3+": None,
}
EXCLUDED = ("Not stated", "Not asked (speaks no indigenous language)", "Total aged 3+")
# what countries/vu.py draws the not-asked on if DRAW_NOT_ASKED is switched on
EXTRA_NODES = ["creole.english_based.bislama"]


def resolve(name):
    return NAMES[name]
