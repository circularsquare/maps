"""San Marino 2024 register by citizenship (sources/sm_census.py, sources/sm.md): the script
writes node ids directly.
"""
NAMES = {
    "indoeuropean.romance.romagnol": "indoeuropean.romance.romagnol",
    "indoeuropean.romance.italian": "indoeuropean.romance.italian",
    "other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"sm2024: unmapped label {label!r}")
