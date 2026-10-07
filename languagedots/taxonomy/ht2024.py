"""Haiti: no language question; one label, everyone on Haitian Creole (sources/ht_pop.py,
sources/ht.md). French is a learned second language (AGENT_BRIEF §2) and is not drawn.
"""
NAMES = {"Kreyòl ayisyen": "creole.french_based.haitian"}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ht2024: unmapped label {label!r}")
