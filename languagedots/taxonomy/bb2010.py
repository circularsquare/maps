"""Barbados, 2010 census country of birth (Table 04.01) and ethnic origin -> node
(sources/bb_census.py, sources/bb.md). No language question: each birthplace on its country's
home vernacular, which for the English-speaking Caribbean is its English-based creole
(Glottolog: Guyanese Creole English creo1235, Vincentian vinc1243, Trinidadian trin1276,
Grenadian gren1247, Antigua and Barbuda (Leeward) creo anti1245, which also covers St Kitts).

  Barbados (not white): Bajan (Glottolog baja1265).
  Barbados (white): English; white Barbadians speak a variety of English, not Bajan creole.
  India: Gujarati. Barbados's Indian community came mostly from Gujarat (Surat and Bharuch).
  Suriname: Sranan Tongo, its lingua franca (COUNTRY_LANG's Ndyuka is a Maroon language).
  Other Asia / Other Latin America / Other Countries: pooled, unnamed -> `other`.
"""
EN = "indoeuropean.germanic.english"
CR = "creole.english_based"
NAMES = {
    "Barbados (not white)": f"{CR}.bajan",
    "Barbados (white)": EN,
    "Guyana": f"{CR}.guyanese",
    "St. Vincent and the Grenadines": f"{CR}.vincentian",
    "Trinidad and Tobago": f"{CR}.trinidadian",
    "Grenada": f"{CR}.grenadian",
    "Antigua and Barbuda": f"{CR}.antiguan",
    "St. Kitts and Nevis": f"{CR}.antiguan",
    "Jamaica": f"{CR}.jamaican",
    "Bahamas": f"{CR}.bahamian",
    "Suriname": f"{CR}.sranan",
    "St. Lucia": "creole.french_based.antillean",
    "Dominica": "creole.french_based.antillean",
    "Haiti": "creole.french_based.haitian",
    "Belize": EN,
    "Australia": EN, "Bermuda": EN, "Canada": EN, "U.K.": EN, "U.S.A": EN,
    "China": "sinotibetan.sinitic.mandarin",
    "Cuba": "indoeuropean.romance.spanish",
    "India": "indoeuropean.indoaryan.gujarati.gujarati",
    "Other Asia": "other",
    "Other Latin America": "other",
    "Other Countries": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"bb2010: unmapped label {label!r}")
