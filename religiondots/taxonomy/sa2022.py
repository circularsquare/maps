"""
Saudi Arabia, 2022 census base -> religiondots taxonomy.

Named for the population base (GASTAT's census, reference date 10 May 2022). No source asks anyone
in Saudi Arabia their religion; `sources/sa.py` and `sources/sa.md` have the construction.

Saudi citizens, 18,792,262 (the census's Saudis in each of the 13 regions), as drawn:

    100%  Saudi citizens  -> islam

Non-Saudis, 13,382,962, arrive already on nodes (`data/normalized/sa_foreign.csv`): each region's
non-Saudi men and women take the census's national nationality mix for their sex, through
`taxonomy/origin_religion.py`, with Muslim branches folded to `islam` in `sources/sa.py`. Every row
is `modelled` (spec §7b).
"""

EXCLUDED = {}

REVIEW = {
    "Saudi citizens":
        "-> islam, the bare family node, as Mauritania and the Maghreb. NOBODY WAS ASKED: the 2022 "
        "census has no religion item, the Arab Barometer's Saudi rows leave Q1012 empty (wave II) or "
        "do not exist (wave V), and the Ministry of Islamic Affairs publishes mosque counts. Citizens "
        "are legally Muslim, so every citizen is drawn on Islam and every non-Muslim dot is a "
        "non-Saudi. NOT SPLIT INTO SUNNI AND SHIA although the US State Department's 2023 report "
        "puts citizens at 85-90% Sunni and 10-12% Shia, with Shia 25-30% of the Eastern Province's "
        "population and Ismailis widely believed a large majority of Najran (Human Rights Watch "
        "2008): Saudi Shia have been bombed in their mosques (Qatif and Dammam, May 2015; Najran, "
        "October 2015) and are prosecuted out of proportion, which is spec §14's case. Filed as an "
        "ask with the evidence; drawing a split would change these rows to islam.sunni and "
        "islam.shia by region.",
}

MAP = {
    "Saudi citizens": "islam",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
