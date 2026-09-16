"""
Somalia (no source asks religion; `sources/so.py`) -> religiondots taxonomy.

Named for the population base, the 2026 humanitarian planning estimate (COD-PS). `sources/so.md` has
the construction: every person in each region is drawn on Islam, every row `modelled` (spec §7b),
because no census, survey or register in Somalia asks religion (`sources.md` §11aq) and the only
figures are compilers' national estimates.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as Mauritania. The State Department's 2023 report cites the "
        "federal Ministry of Endowments and Religious Affairs for 'more than 99 percent' Sunni, but "
        "nothing measured the split, so no branch is drawn. Somalis who are not Muslim (a Christian "
        "community of about 1,000, per the same report quoting World Atlas) are not placed anywhere "
        "below the nation, on spec §14: converts have been targeted by al-Shabaab (sources/so.md §2).",
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
