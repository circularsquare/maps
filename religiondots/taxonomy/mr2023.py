"""
Mauritania, RGPH 2023 base -> religiondots taxonomy.

Named for the population base (RGPH 2023, counted 25 December 2023 to 8 January 2024). No source
asks Mauritanians their religion; `sources/mr.py` and `sources/mr.md` have the construction.

Mauritanian nationals, 4,801,598 (the census total less its foreign residents, wilaya by wilaya),
as drawn:

    100%  Muslim  -> islam

Foreign residents, 125,933, arrive already on nodes (`data/normalized/mr_foreign.csv`): the
census's national nationality groups through `taxonomy/origin_religion.py`, with the Mbera camp's
refugees placed in Hodh Chargui and Muslim branches folded to `islam` in `sources/mr.py`. Every row
is `modelled` (spec §7b).
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as Morocco, Algeria, Tunisia and Libya. NOBODY WAS ASKED: "
        "no Mauritanian census has published religion (the 2013 form has no item; none of ANSADE's "
        "sixteen thematic reports on 2023 has a table on it) and no survey that reaches Mauritania "
        "puts the question (Arab Barometer VII-VIII leave Q1012 empty, Afrobarometer skips it, DHS "
        "2019-21 has no item). Every national is drawn on Islam: the CIA World Factbook's entry reads "
        "`Muslim (official) 100%`, and Pew Research Center's 2020 estimate for everyone living in the "
        "country (99.185% Muslim, 10,754 Christians) has as many Christians as the foreigner layer "
        "gives on its own (sources/mr.py prints the comparison), so Pew's non-Muslims are the "
        "foreigners. Mauritanians who are not Muslim, whom no source has counted, are not drawn; "
        "sources/mr.md §4 has the alternative (Pew's residual on nationals) and why it was not "
        "taken. Maliki Sunni in practice, but no source names a school, so not islam.sunni.",
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
