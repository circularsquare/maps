"""
CSO Census 2022 religion classification -> religiondots taxonomy.

**Branch-level mapping, like the others.** No leaves created (spec §2.4).

CSO publishes 5 categories at Small Area and 24 at county, so 20 of these arrive already
allocated (§3.9) with `tier=derived`; countries.py turns that into `may_ring=False` (§3.10).
The 4 measured ones are Roman Catholic, No religion, Not stated and the `Other religion`
total that the rest were split out of.

Ireland's list is a WRITE-IN list, not a classification: `Protestant`, `Evangelical` and
`Born Again Christian` are three separate rows that all mean "Christian, no body named", and
`Lapsed (Roman) Catholic` is a category because people wrote it. That is why nearly all of
this maps to branches rather than to anything specific.
"""

EXCLUDED = {
    "Not stated":
        "345,165 people, 6.7% of Ireland. Not a religion and not 'no religion', which is "
        "its own category at 755,455. Excluded from the dots, as Czechia's 'Neuvedeno' and "
        "Brazil's 'Não sabe' are. The Irish question is NOT voluntary in the Czech sense — "
        "sources/ie.md §6 — so 6.7% is a non-response rate rather than an opt-out.",
}

REVIEW = {
    "Orthodox (Greek, Coptic, Russian)":
        "-> christianity.orthodox. CSO's label puts EASTERN Orthodoxy (Greek, Russian) and "
        "ORIENTAL Orthodoxy (Coptic) in one row, and those are separate communions and have "
        "been since 451 — branches.py calls conflating them the commonest error in religion "
        "taxonomies. One row, so it cannot be split here. Filed Eastern because the Greek "
        "and Russian share is much the larger; the Coptic minority is misfiled and this note "
        "is the record of that. Same call StatCan forced for Canada.",
    "Lapsed (Roman) Catholic":
        "-> christianity.catholic.latin. 3,279 people who wrote 'lapsed Catholic' as their "
        "religion. CSO files it under `Other religion` rather than `No religion`, and the "
        "answer does name Catholicism, so it goes with the Catholics — but it is arguable "
        "either way and someone may reasonably move it to unaffiliated.",
    "Born Again Christian":
        "-> christianity.protestant. An evangelical self-description, not a body and not "
        "quite `nondenominational` either.",
    "Evangelical":
        "-> christianity.protestant. Names no body; sits beside `Protestant` in CSO's own "
        "list, which is why both land on the same node.",
}

MAP = {
    # ---------------------------------------------------------------- Catholic
    "Roman Catholic": "christianity.catholic.latin",
    "Lapsed (Roman) Catholic": "christianity.catholic.latin",

    # ---------------------------------------------------------------- other Christian
    "Church of Ireland, England, Anglican, Episcopalian": "christianity.anglican",
    "Orthodox (Greek, Coptic, Russian)": "christianity.orthodox",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Methodist, Wesleyan": "christianity.methodist",
    "Lutheran": "christianity.lutheran",
    "Baptist": "christianity.baptist",
    "Apostolic or Pentecostal": "christianity.pentecostal",
    "Jehovah's Witness": "christianity.witnesses",
    "Protestant": "christianity.protestant",
    "Evangelical": "christianity.protestant",
    "Born Again Christian": "christianity.protestant",
    "Christian (Not Specified)": "christianity",

    # ---------------------------------------------------------------- other families
    "Islam": "islam",
    "Hindu": "hinduism",
    "Buddhist": "buddhism",
    "Spiritualist": "spiritualism",
    "Pagan, Pantheist": "paganism",

    # ---------------------------------------------------------------- no religion
    "No religion": "unaffiliated",
    "Agnostic": "secular",
    "Atheist": "secular",

    # ---------------------------------------------------------------- residual
    "Other stated religion (nec)": "other.ie",
}


def resolve(category):
    """religiondots branch for a CSO category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
