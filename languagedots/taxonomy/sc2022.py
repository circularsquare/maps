"""Seychelles, Population and Housing Census 2022, Table B3.1a, first main language spoken at
home (aged 3+) -> node. Keyed by the column heads sources/sc_census.py writes.

  * Creole is Seychellois Creole (Kreol Seselwa), on au.txt's `seselwa` leaf.
  * English, French, Gujarati, Hindi, Tamil: their own leaves, as au2021 and mu2022 map them.
    The Outer Islands' 707 Gujarati speakers (no Creole speaker there at all) are drawn as
    printed; sources/sc.md says why.
  * Other (Specify) (1,454): the report prints no breakdown at any geography. Seychelles had no
    people before settlement in 1770, so there is no indigenous remainder to keep apart: `other`.
  * Do Not Know (115), Refusal (13), Missing (11,417): not drawn; countries/sc.py counts them in
    `gap`.
"""

NAMES = {
    "Creole": "creole.french_based.seselwa",
    "English": "indoeuropean.germanic.english",
    "French": "indoeuropean.romance.french",
    "Gujarati": "indoeuropean.indoaryan.gujarati.gujarati",
    "Hindi": "indoeuropean.indoaryan.central.hindi",
    "Tamil": "dravidian.southern.tamil",
    "Other (Specify)": "other",
}
SKIP = {"Do Not Know", "Refusal", "Missing"}


def resolve(label):
    if label in SKIP:
        return None
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"sc2022: unmapped label {label!r}")
