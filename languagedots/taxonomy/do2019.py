"""Dominican Republic, ENHOGAR-MICS6 2019 item HC1B, mother tongue of the household head -> node.
sources/do_enhogar.py; keyed by the survey's own Spanish labels.

CALLS (sources/do.md says more):
  Creol: Haitian Creole. In the Dominican Republic "creol" means Haitian Kreyol; the survey offers
    it beside French, so the two are told apart by the respondent.
  Ingles: English. Some of it is Samana English, the 19th-century African-American settlers'
    variety, and some the Anglophone Caribbean *cocolo* migration to the eastern sugar belt;
    the survey has no separate answer, so the answer given is drawn.
  Otro idioma: `other` (written-in answers are not in the open microdata).
"""
NAMES = {
    "Español": "indoeuropean.romance.spanish",
    "Creol": "creole.french_based.haitian",
    "Inglés": "indoeuropean.germanic.english",
    "Francés": "indoeuropean.romance.french",
    "Otro idioma": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"do2019: unmapped label {label!r}")
