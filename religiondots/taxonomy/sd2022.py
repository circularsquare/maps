"""
Afrobarometer and Arab Barometer answers (Sudan, 2013-2022, grouped in `sources/sd.py`) -> religiondots
taxonomy.

Named for the population base (the Central Bureau of Statistics' projection for 2022, COD-PS).
`sources/sd.py` and `sources/sd.md` have the construction: one national non-Muslim share and mix
in every state, every row `modelled` (spec §7b), because no Sudanese census has asked religion
since 1956 and the 2008 census had the question deleted.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as Libya, Algeria, Morocco and Tunisia. The Arab "
        "Barometer's follow-up records a label, not a school: sources.md §11af found 81.8% of "
        "Sudanese Muslims answer 'Just a Muslim'. The Afrobarometer's brotherhood answers "
        "(Tijaniya 6, Qadiriya 1, Mouridiya 1) and its Ismaili (2) and Shia (2) answers are "
        "folded in, too few to draw.",
    "Christian":
        "-> christianity, the bare family node. The Afrobarometer names churches (Presbyterian 3, "
        "Seventh-day Adventist 2, Anglican, Lutheran, Methodist, Orthodox, Jehovah's Witness, "
        "Church of Christ, Mormon, an independent church, one each) and the Arab Barometer's "
        "follow-up records Catholic and Orthodox, but most say only 'Christian', and no church "
        "reaches ten answers, so none is drawn.",
    "No religion":
        "-> unaffiliated. The Afrobarometer's None, Atheist and Agnostic, and each Arab Barometer "
        "card's one box (V `Atheist`, VII `No religion`). Six of wave V's seven atheists named a "
        "Muslim branch on the follow-up and are dropped, and so are round 8's 31 `None` answers "
        "in Darfur (sources/sd.md §3).",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "No religion": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a grouped survey answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
