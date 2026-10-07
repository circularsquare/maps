"""Marshall Islands 2021 census, Table 3.3 (sources/mh_census.py, sources/mh.md). Languages
spoken, several allowed; only Marshallese is named, so the people who do not speak it go on
`other` (language not named).
"""
NAMES = {
    "Marshallese": "austronesian.oceanic.marshallese",
    "Does not speak Marshallese": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"mh2021: unmapped label {label!r}")
