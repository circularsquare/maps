"""Cayman Islands, 2021 census country of birth by district (sources/ky_census.py, sources/ky.md).
No language question: each birthplace on its home vernacular, as Barbados (taxonomy/bb2010.py).

  Cayman Islands: English (Caymanian English; no creole is recognised for the Caymans).
  South Africa: English, not COUNTRY_LANG's Zulu: South Africans in the Caymans are mostly
    English-speaking professionals (financial services).
  Honduras: Spanish. Some Honduran-born residents come from the English-creole-speaking Bay
    Islands; no count separates them (sources/ky.md).
  India: Hindi (COUNTRY_LANG); no source gives Cayman Indians' origins.
  Other, and the rows a district table leaves out: `other` (unnamed).
"""
EN = "indoeuropean.germanic.english"
ES = "indoeuropean.romance.spanish"
CR = "creole.english_based"
NAMES = {
    "Cayman Islands": EN,
    "Jamaica": f"{CR}.jamaican",
    "United States of America": EN, "United Kingdom": EN, "Canada": EN, "Ireland": EN,
    "Australia": EN, "South Africa": EN,
    "Honduras": ES, "Nicaragua": ES, "Cuba": ES, "Costa Rica": ES, "Colombia": ES,
    "Barbados": f"{CR}.bajan",
    "Trinidad and Tobago": f"{CR}.trinidadian",
    "Guyana": f"{CR}.guyanese",
    "Philippines": "austronesian.philippine.tagalog",
    "India": "indoeuropean.indoaryan.central.hindi",
    "Other": "other",
    # the gap between a district table's rows and its printed Total, where the table leaves rows
    # out (East End, Cayman Brac, Little Cayman); the birthplace is unknown
    "Not listed in the district table": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ky2021: unmapped label {label!r}")
