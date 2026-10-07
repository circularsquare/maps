"""
DHS EDSGE-I 2011 (Equatorial Guinea, Cuadro 3.1, printed nationally) -> religiondots taxonomy.

Named for the survey year. `sources/gq.py` and `sources/gq.md` have the construction: the survey's
weighted national shares of women and men 15-49, combined at the 2015 census's sex split, the same
mix in every province on the 2015 census count, every row `modelled` (spec §7b), because the 2015
census asked religion and published no table of it.

    89.77%  Cristiano, católico      -> christianity.catholic.latin
     5.10%  Cristiano, protestante   -> christianity.protestant
     3.81%  Musulmana                -> islam
     0.79%  Animista                 -> indigenous.african
     0.30%  Otro                     -> other.gq (NEW)
     0.23%  Sin religión             -> unaffiliated
"""

EXCLUDED = {}

REVIEW = {
    "Cristiano, católico":
        "The survey's one Christian box (94.87%) split 88:5 at the government estimate for 2015 "
        "that the US State Department's 2023 religious freedom report quotes (88% Catholic, 5% "
        "Protestant, 2% Muslim, 5% other). The estimate names no method; it is dated to the census "
        "year and may come from the census's own question, which offered Catholic and Protestant "
        "boxes, but nothing says so. Drawn as Bulgaria's national denomination split was (Anita, "
        "2026-09-08): the survey sets the Christian total and the estimate only divides it, and its "
        "own Christian total (93%) is within 2 points of the survey's, a gap `sources/gq.py` "
        "asserts. The fallback is the bare `christianity` node for all 94.87%.",
    "Cristiano, protestante":
        "-> christianity.protestant, the family node. The State Department report names the "
        "Reformed Church of Equatorial Guinea beside the Catholic Church as the two bodies the "
        "government favours, but no figure splits Protestants further.",
    "Musulmana":
        "-> islam. The State Department report says most Muslims are Sunni and are expatriates from "
        "other West African countries, with no figure for either, so no branch is drawn and they "
        "are not placed with the foreign residents (sources/gq.md §3).",
    "Animista":
        "-> indigenous.african, the card's own word for traditional religion. The report adds that "
        "many Christians practise some traditional rites too; a one-answer question records whichever "
        "a person names.",
    "Otro":
        "-> other.gq, a per-source residual (spec §3.11). The report prints no specified text; the "
        "State Department lists Baha'is and Jews among the other 5% of the government estimate.",
}

MAP = {
    "Cristiano, católico": "christianity.catholic.latin",
    "Cristiano, protestante": "christianity.protestant",
    "Musulmana": "islam",
    "Animista": "indigenous.african",
    "Sin religión": "unaffiliated",
    "Otro": "other.gq",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, a national survey share on a count.


def resolve(category):
    """religiondots branch for a DHS answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
