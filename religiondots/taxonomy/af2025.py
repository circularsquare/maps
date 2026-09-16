"""
Afghanistan, NSIA 1404 (2025-26) estimate base -> religiondots taxonomy.

Named for the population base: NSIA's *Estimated Population of Afghanistan 2025-26*. No source asks
Afghans their religion; `sources/af.py` and `sources/af.md` have the construction.

Settled population, 34,935,197 in 34 provinces, as drawn:

    100%  Muslim  -> islam

Every row is `modelled` (spec §7b). The 1,500,000 nomadic Kuchis are not drawn (no province).
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as Mauritania. NOBODY WAS ASKED: no census since 1979 "
        "(NSIA's 1404 introduction), and the Survey of the Afghan People has no religion or sect "
        "item. Every settled person is drawn on Islam: Pew Research Center's 2020 estimate is "
        "99.862% Muslim, and its 53,928 non-Muslims (35,179 of them `other religions`) are a cell "
        "nothing explains or places; the US State Department's 2023 report counts six Sikhs and "
        "Hindus and says no reliable estimate exists for Christians or Baha'is. NOT islam.sunni: "
        "Shia estimates run from 11% (World Religion Database 2022) to 29% (Gulf 2000), Pew's 2011 "
        "survey found 7% of Muslims calling themselves Shia, and nothing gives a share by province "
        "(sources/af.md §3). Shia Hazaras and Ismailis sit inside this node; a split is a §14 "
        "question before it is a data one.",
}

MAP = {
    "Muslim": "islam",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
