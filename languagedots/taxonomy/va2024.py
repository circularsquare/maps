"""Vatican City 2024 population (sources/va_pop.py, sources/va.md): the script writes node ids
directly (Italian, and Switzerland's home mix for the Swiss Guard).
"""
NAMES = {n: n for n in ("indoeuropean.romance.italian",
                        "indoeuropean.germanic.continental.german",
                        "indoeuropean.romance.french")}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"va2024: unmapped label {label!r}")
