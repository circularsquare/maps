"""
Oman, end-2024 register base -> religiondots taxonomy.

Named for the population base (NCSI's register at the end of December 2024, *Statistical Year Book
2025*, Table 7-2). No source asks anyone in Oman their religion; `sources/om.py` and `sources/om.md`
have the construction.

Omanis, 2,984,793 (the register's Omanis in each of the 63 wilayat), as drawn:

    100%  Omani citizens  -> islam

Expatriates, 2,283,279, arrive already on nodes (`data/normalized/om_foreign.csv`): each
governorate's male workers, female workers and dependants take their own national nationality mix,
through `taxonomy/origin_religion.py`, with Muslim branches folded to `islam` in `sources/om.py`.
Every row is `modelled` (spec §7b).
"""

EXCLUDED = {}

REVIEW = {
    "Omani citizens":
        "-> islam, the bare family node, as Saudi Arabia and Mauritania. NOBODY WAS ASKED: the 2003 "
        "census form has no religion item, the 2003, 2010 and 2020 results print none, the register "
        "holds none, and Oman is not in the Arab Barometer. No source counts an Omani who is not "
        "Muslim, so every non-Muslim dot is an expatriate. NOT SPLIT INTO IBADI, SUNNI AND SHIA: the "
        "figures for Omanis are national only and disagree (Peterson 2004 about 45% Ibadi, 50% "
        "Sunni, under 5% Shia; AEI 2013 three-quarters Ibadi; an uncited 21% Ibadi), and below the "
        "nation there is only description (Dhofar entirely Sunni, the interior Ibadi, Shia in "
        "Muscat and on the Batinah coast). A Shia mosque in Wadi al-Kabir, Muscat, was attacked in "
        "July 2024 (spec §14). Filed as an ask; a split would change these rows to islam.ibadi, "
        "islam.sunni and islam.shia by wilaya.",
}

MAP = {
    "Omani citizens": "islam",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
