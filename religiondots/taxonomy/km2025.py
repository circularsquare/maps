"""Afrobarometer round 10, Comoros (2025; *Résumé des résultats*, sample description p.5, weighted)
-> religiondots taxonomy.

Named for the survey year. `sources/km.py` and `sources/km.md` have the construction: the
survey's weighted national shares, the same mix on each of the three islands' 2017 census counts,
every row `modelled` (spec §7b), since nothing published places a religion below the nation.

    99.6%  Musulmans   -> islam
     0.3%  Chrétiens   -> christianity
     0.1%  Autre       -> unaffiliated
     0.0%  Refus       (dropped in sources/km.py, not a category here)
"""

EXCLUDED = {}

REVIEW = {
    "Musulmans":
        "-> islam, no branch. Q97 splits it into `Musulman seulement` 96.2%, `Sunnite seulement` "
        "1.0% and `Ismaélite` 2.4% (29 of 1,200 unweighted, 3.7% in towns). Nothing places any of "
        "the three, so no sect is drawn (the Oman, Saudi and Tajikistan rulings); the Ismaili "
        "share is not checked against any other source.",
    "Chrétiens":
        "-> christianity, the bare branch. 0.3% weighted, 3 respondents unweighted, all coded "
        "`Mormon/saints des derniers jours` in the codebook of 25 August 2025 (the revised summary "
        "also shows a trace of `Chrétien seulement`). Three answers cannot name a church, so "
        "nothing goes to a Latter-day Saint node. DHS 2012 also finds 0.3% Catholic or Protestant "
        "among women and men 15-49.",
    "Autre":
        "-> unaffiliated. The sample description's `Autre` (0.1%) is Q97's `Aucune` (0.1%), the "
        "only answer outside Christianity, Islam and refusal (one respondent unweighted); "
        "sources/km.py asserts the two agree. The card offered traditional religion, agnostic and "
        "atheist as separate codes and nobody chose them.",
}

MAP = {
    "Musulmans": "islam",
    "Chrétiens": "christianity",
    "Autre": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, a national survey share on a census
# count.


def resolve(category):
    """religiondots branch for a summary category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
