"""Turks and Caicos Islands, 2012 census country of citizenship (sources/tc_census.py,
sources/tc.md). No language question: TCI citizens on Turks and Caicos Creole (Glottolog turk1310,
node from tree.d/bs.txt), foreign nationals on their country's language, "Other" on `other`."""
CR = "creole.english_based."
NAMES = {
    "TCI": CR + "turks_caicos",
    "Haiti": "creole.french_based.haitian",
    "Domincan Republic": "indoeuropean.romance.spanish",   # sic, the sheet's spelling
    "Bahamas": CR + "bahamian",
    "USA": "indoeuropean.germanic.english",
    "Canada": "indoeuropean.germanic.english",
    "England": "indoeuropean.germanic.english",
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"tc2012: unmapped label {label!r}")
