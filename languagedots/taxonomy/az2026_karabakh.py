"""Karabakh's resettled population, 2026 (sources/az_karabakh.py) -> node.

One label: the returnees are former internally displaced Azerbaijanis, drawn as Azerbaijani. No
source gives their languages; this is the ethnicity-only rule (AGENT_BRIEF §2) applied to a
published count, every row `modelled`.
"""
NAMES = {"Azerbaijani": "turkic.azerbaijani"}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"az2026_karabakh: unmapped label {label!r}")
    return NAMES[label]
