"""Anguilla 2001 census citizenship read as first language (sources/ai_census.py, sources/ai.md).
"""
CR = "creole.english_based."
NAMES = {
    "Anguillian": CR + "antiguan",          # the Leeward creole, Glottolog anti1245 (AI listed)
    "St. Kitts": CR + "antiguan",
    "USA": "indoeuropean.germanic.english",
    "UK": "indoeuropean.germanic.english",
    "Dominican Republic": "indoeuropean.romance.spanish",
    "Jamaica": CR + "jamaican",
    "Other Caribbean": "creole",             # unnamed mix of Caribbean creoles: the group node
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ai2001: unmapped label {label!r}")
