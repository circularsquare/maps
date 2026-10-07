"""Israeli settlers beyond the Green Line, CBS Social Survey 2021 native language
(sources/xs_social.py) -> node. Israel's mapping (il2021) restricted to the "Jews and others"
group, the only one drawn here: Arabic named by Jews and others sits on plain Arabic, "Another
Language" on `other`, exactly as on Israel's entry.
"""
from il2021 import NAMES as _IL

NAMES = {k: v for k, v in _IL.items() if k.endswith("(Jews and others)")}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"xs2021: unmapped label {label!r}")
