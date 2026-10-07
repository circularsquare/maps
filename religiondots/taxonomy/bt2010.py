"""
Bhutan, GNH 2010 level -> religiondots taxonomy.

Named for the level's vintage: the Centre for Bhutan Studies' estimate from its 2010 Gross National
Happiness survey, "Eighty-one per cent of Bhutanese are Buddhists, 18% are Hindus, and 1.2% are
Christians" (*An Extensive Analysis of GNH Index*, 2012, p.153). The census asked religion in 2005
and never published it; `sources/bt.py` and `sources/bt.md` have the construction.

Bhutanese, 681,720 (the 2017 census, dzongkhag by dzongkhag), as drawn:

    80.80%  Buddhist   -> buddhism
    18.00%  Hindu      -> hinduism      placed by the 2015 survey's Nepali mother tongue
     1.20%  Christian  -> christianity

Non-Bhutanese, 45,425, arrive already on nodes (`data/normalized/bt_foreign.csv`): UN DESA's 2020
origins through `taxonomy/origin_religion.py`, Muslim branches folded to `islam`. Every row is
`modelled` (spec §7b).
"""

EXCLUDED = {}

REVIEW = {
    "Buddhist":
        "-> buddhism, the bare family node. Bhutan's Buddhism is Drukpa Kagyu and Nyingma, both "
        "Vajrayana, but the survey's answer is `Buddhism` and spec §2.6 and ask 025 keep a country "
        "on plain buddhism where no source names a school.",
    "Hindu":
        "-> hinduism. Placed by mother tongue, not measured by dzongkhag: sources/bt.py, ask 046.",
    "Christian":
        "-> christianity, unspecified. The survey's answer is `Christianity` with no church named; "
        "Bhutan's churches are unregistered house fellowships, mostly Protestant and Pentecostal by "
        "every report, but no source counts them, so no branch is guessed. Drawn at one national "
        "share: nothing places them.",
}

MAP = {
    "Buddhist": "buddhism",
    "Hindu": "hinduism",
    "Christian": "christianity",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted by dzongkhag.


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
