"""Cuba: no language question; one label, everyone on Spanish (sources/cu_geo.py, sources/cu.md)."""
NAMES = {"Español": "indoeuropean.romance.spanish"}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"cu2023: unmapped label {label!r}")
