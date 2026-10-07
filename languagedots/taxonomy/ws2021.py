"""Samoa, 2021 census (sources/ws_census.py, sources/ws.md). No language question: Samoan
citizens drawn as Samoan; non-citizens, whose nationality is not published, are not drawn."""
NAMES = {"Samoan": "austronesian.oceanic.samoan"}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ws2021: unmapped label {label!r}")
