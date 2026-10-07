"""Dominica, 2011 census settlement counts (sources/dm_census.py, sources/dm.md). No language
question: Wesley and Marigot/Concord on Kokoy (Glottolog files it under Antiguan and Barbudan
Creole, anti1245), the Haitian-born on Haitian Creole, everyone else on Kweyol (Antillean
Creole)."""
NAMES = {
    "Kokoy": "creole.english_based.antiguan",
    "Haitian-born": "creole.french_based.haitian",
    "Everyone else": "creole.french_based.antillean",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"dm2011: unmapped label {label!r}")
