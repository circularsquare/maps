"""
Canonical-label normalization shared by make_colors.py and build_dots.py.

Goal: a unified label/colour space with the US map (../ancestry_colors.csv) so the
two maps can eventually share one tileset. StatCan labels are normalized by:
  1. dropping ", n.o.s." / ", n.i.e." qualifiers  ("French, n.o.s." -> "French")
  2. a small ALIAS table for spelling / parenthetical differences vs the US labels
     ("Salvadorean" -> "Salvadoran", "Cambodian (Khmer)" -> "Cambodian", ...)

canonical(raw_statcan_label) -> display/colour label used everywhere downstream.
"""

from __future__ import annotations

import re

# Applied AFTER stripping n.o.s./n.i.e. Maps a normalized StatCan label to the
# canonical (usually US) spelling so the two datasets share one colour entry.
ALIAS = {
    "Salvadorean": "Salvadoran",
    "Argentinian": "Argentinean",
    "Trinidadian/Tobagonian": "Trinidadian and Tobagonian",
    "Cambodian (Khmer)": "Cambodian",
    "Mayan": "Maya",
    "Slovenian": "Slovene",
    # parenthetical cleanups (Canada-only, just tidier display)
    "First Nations (North American Indian)": "First Nations",
}

_NOS = re.compile(r",?\s*n\.o\.s\.\s*$")
_NIE = re.compile(r",?\s*n\.i\.e\.\s*$")


def canonical(label: str) -> str:
    s = _NIE.sub("", _NOS.sub("", label.strip())).strip()
    return ALIAS.get(s, s)
