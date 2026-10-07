"""Antigua and Barbuda, 2011 census Q58 country of birth (sources/ag_census.py, sources/ag.md).
No language question: each birthplace on its home vernacular, as Barbados (taxonomy/bb2010.py).

  Antigua and Barbuda, Montserrat, St Kitts and Nevis: Antiguan and Barbudan Creole (Glottolog
    anti1245, which covers the Leeward Islands' creole).
  USVI: Virgin Islands Creole (virg1240), new here.
  USA, Canada, UK: English. Syria: Levantine Arabic (Antigua's Syrian-Lebanese community).
  Africa: `africa_other`; Other Caribbean / Latin or North American / Asian / European: `other`
    (pooled, unnamed).
"""
CR = "creole.english_based"
EN = "indoeuropean.germanic.english"
NAMES = {
    "Antigua and Barbuda": f"{CR}.antiguan",
    "Monsterrat": f"{CR}.antiguan",                       # sic in the Redatam output
    "St. Kitts and Nevis": f"{CR}.antiguan",
    "Guyana": f"{CR}.guyanese",
    "Jamaica": f"{CR}.jamaican",
    "St. Vincent and the Grenadines": f"{CR}.vincentian",
    "Trinidad and Tobago": f"{CR}.trinidadian",
    "USVI United States Virgin Islands": f"{CR}.virgin_islands",
    "Dominica": "creole.french_based.antillean",
    "St. Lucia": "creole.french_based.antillean",
    "Dominican Republic": "indoeuropean.romance.spanish",
    "USA": EN, "Canada": EN, "United Kingdom": EN,
    "Syria": "afroasiatic.levantine_arabic",
    "Africa": "africa_other",
    "Other Caribbean countries": "other",
    "Other Latin or North Amercian countries": "other",   # sic
    "Other Asian countries": "other",
    "Other European countries": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ag2011: unmapped label {label!r}")
