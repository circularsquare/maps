"""Andorra, World Values Survey wave 7 (2018), Q272 language at home (sources/ad_wvs.py,
sources/ad.md). "Other European" and "Other" are unnamed answers, so both sit on `other` (no
European-remainder node exists, and the survey does not say which languages they were)."""
NAMES = {
    "Catalan; Valencian": "indoeuropean.romance.catalan",
    "Spanish; Castilian": "indoeuropean.romance.spanish",
    "French": "indoeuropean.romance.french",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Other European": "other",
    "Other": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ad2018: unmapped label {label!r}")
