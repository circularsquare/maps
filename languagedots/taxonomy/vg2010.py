"""British Virgin Islands, 2010 census grouped country of birth by island (sources/vg_census.py,
sources/vg.md). No language question: the BVI-born on Virgin Islands Creole (Glottolog virg1240,
covering both the US and British Virgin Islands), the foreign-born on their birth country's
language, as Barbados (bb2010.py). Pooled regions sit on `other` (Africa on africa_other)."""
CR = "creole.english_based."
EN = "indoeuropean.germanic.english"
ES = "indoeuropean.romance.spanish"
NAMES = {
    "Virgin Islands": CR + "virgin_islands",
    "United States Virgin Islands": CR + "virgin_islands",
    "Dominica": "creole.french_based.antillean",
    "St Lucia": "creole.french_based.antillean",
    "Grenada": CR + "grenadian",
    "Jamaica": CR + "jamaican",
    "St Kitts and Nevis": CR + "antiguan",      # anti1245, the Leeward creole (kn.md)
    "St Vincent and Grenadines": CR + "vincentian",
    "Trinidad and Tobago": CR + "trinidadian",
    "Guyana": CR + "guyanese",
    "Dominican Republic": ES,
    "Puerto Rico": ES,
    # as Barbados, Antigua, Bermuda and St Kitts: many are BVI islanders' children
    "United States of America": EN,
    "United Kingdom": EN,
    # pooled, unnamed: the narrowest node holding everything filed there
    "Other Caribbean": "other",       # English, French and Spanish creoles and languages
    "Overseas Territories": "other",  # British overseas territories, Anguilla to Bermuda
    "Europe": "other",
    "Latin America": "other",
    "Asia": "other",
    "Pacific": "other",
    "Middle East": "other",
    "Other Countries": "other",
    "Africa": "africa_other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"vg2010: unmapped label {label!r}")
