"""Curaçao, Census 2023, language spoken most often at home ("eerste taal") -> node.
sources/cw_census.py; keyed by the Dutch labels of the first-results box (page 21), as
data/normalized/cw.csv carries them. The 2011 rows (geo_level national_2011, Table D-6) are a check
only and are keyed by their English labels; countries/cw.py does not draw them, but they resolve
so the record's comparison runs through the same mapping.

CALLS (sources/cw.md says more):
  Papiamentu: `creole.portuguese_based.papiamento`, pl2021's node.
  Spaans, Nederlands, Engels: the nodes every other country uses.
  "overig" (2.0%, about 2,950 people): `other`. In 2011 the same remainder was French Creole,
    Chinese, Portuguese, Hindi, Arabic and other (3,375 people), several families and no
    indigenous language, so `other` is the narrowest node holding it.
"""
NAMES = {
    "Papiamentu": "creole.portuguese_based.papiamento",
    "Spaans": "indoeuropean.romance.spanish",
    "Nederlands": "indoeuropean.germanic.continental.dutch",
    "Engels": "indoeuropean.germanic.english",
    "overig": "other",
}
# Census 2011, Table D-6 (check only)
NAMES_2011 = {
    "Papiamentu": "creole.portuguese_based.papiamento",
    "Spanish": "indoeuropean.romance.spanish",
    "Dutch": "indoeuropean.germanic.continental.dutch",
    "English": "indoeuropean.germanic.english",
    "Arabic": "afroasiatic.arabic",
    "French Creole": "creole.french_based.haitian",
    "Chinese": "sinotibetan.sinitic",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Hindi": "indoeuropean.indoaryan.central.hindi",
    "Other": "other",
}
SKIP = {"Not reported", "Total"}


def resolve(label):
    if label in SKIP:
        return None
    if label in NAMES:
        return NAMES[label]
    if label in NAMES_2011:
        return NAMES_2011[label]
    raise KeyError(f"cw2023: unmapped label {label!r}")
