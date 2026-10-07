"""Rwanda: no language question; one label, everyone on Kinyarwanda (sources/rw_pop.py,
sources/rw.md). English, French and Swahili are learned second languages (AGENT_BRIEF §2) and
are not drawn.
"""
NAMES = {"Ikinyarwanda": "nigercongo.bantu.kinyarwanda"}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"rw2022: unmapped label {label!r}")
