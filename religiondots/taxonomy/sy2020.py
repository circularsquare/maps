"""
Pew Research Center's 2020 composition for Syria (the World Religion Database, through Pew), grouped in
`sources/sy.py` -> religiondots taxonomy.

Named for the year of the mix. `sources/sy.py` and `sources/sy.md` have the construction: one national
mix in every governorate on the CBS end-2011 estimate, every row `modelled` (spec §7b), because no
Syrian census has asked religion since 1960.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node. Pew's row is the World Religion Database's, which files "
        "Syria's Druze, Alawites, Ismailis and Twelver Shia with the Sunni majority: its "
        "`Other_religions` is 559 people (0.003%), where Pew's Lebanon row carries the Druze at "
        "4.3%. The Druze would be `druze`, their own family, if anything gave their number and "
        "place; Suwayda is the obvious placement and is ask 050's, under spec §14. No sect split "
        "on Oman and Saudi Arabia's ruling (2026-09-15): nothing places one.",
    "Christian":
        "-> christianity, the bare family node. Syria's Christians are mostly Greek Orthodox "
        "(Antiochian), Syriac Orthodox, Armenian Apostolic and Melkite Catholic, but Pew gives one "
        "number and nothing splits it.",
    "No religion":
        "-> unaffiliated. Pew's `Religiously_unaffiliated`, the World Religion Database's "
        "agnostics and atheists, 1.98%; no survey of Syria checks it.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "No religion": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a grouped compiler family, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
