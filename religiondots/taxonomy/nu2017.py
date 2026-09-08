"""Niue 2017 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Eight categories, **1,591 people — the smallest country on this map**, and an exact
partition. **One new node**, the national church.

    61.7%  Ekalesia Niue         -> christianity.reformed.congregational.niue   <- new
     8.7%  Latter Day Saints     -> christianity.latterday
     8.4%  Roman Catholic        -> christianity.catholic
     8.2%  Other                 -> other.nu
     5.1%  Not Stated            -> not on the tree
     3.7%  None                  -> unaffiliated
     2.7%  Jehovah's Witnesses   -> christianity.witnesses
     1.4%  Seventh Day Adventist -> christianity.adventist

**AT 1 DOT = 1,000 PEOPLE NIUE IS ONE DOT.** Everything below Ekalesia Niue's 981 rounds to
nothing, so the country draws as a single Congregational dot with a ring for whatever else
survives spec §4.3. That is the honest picture of a country of 1,591 at this dot value and
not a defect; the categories are recorded so the 1:10,000 view and the about panel can say
what is there.

Ekalesia Niue is LMS-descended, planted from 1846 through Samoan and Rarotongan teachers, and
is the sibling of the Cook Islands Christian Church and Ekalesia Kelisiano Tuvalu.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not Stated":
        "81 people, 5.1%, who did not answer. Kept apart from `None` (59), which the same "
        "table publishes separately, so nothing here merges an answer with a non-answer.",
}

REVIEW = {
    "Latter Day Saints":
        "-> christianity.latterday. **8.7% is the highest Latter-day Saint share of any "
        "country drawn here**, ahead of Tonga's and Samoa's if either is ever added, and it "
        "is second only to Ekalesia Niue in Niue itself.",
}

MAP = {
    "Ekalesia Niue": "christianity.reformed.congregational.niue",
    "Latter Day Saints": "christianity.latterday",
    "Roman Catholic": "christianity.catholic",
    "Jehovah's Witnesses": "christianity.witnesses",
    "Seventh Day Adventist": "christianity.adventist",
    "None": "unaffiliated",
    "Other": "other.nu",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
