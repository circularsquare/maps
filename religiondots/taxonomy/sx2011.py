"""Sint Maarten 2011 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Seventeen categories, 33,609 people, national. The UNSD row equals the census report's
Table B-10 to the person (read 2026-09-14; the report prints `Christianity` where UNSD prints
`Christian`, and `Not reported` where UNSD prints `Unknown`). No new nodes beyond `other.sx`.

    33.1%  Roman Catholic   -> christianity.catholic
    14.7%  Pentecostal      -> christianity.pentecostal
    10.0%  Methodist        -> christianity.methodist
     7.9%  No Religion      -> unaffiliated
     6.6%  Adventist        -> christianity.adventist
     5.2%  Hindu            -> hinduism
     4.7%  Baptist          -> christianity.baptist
     4.1%  Christian        -> christianity
     3.1%  Anglican         -> christianity.anglican
     2.8%  Protestant       -> christianity.protestant
     2.4%  Unknown          -> EXCLUDED
     1.7%  Jehovah Witness  -> christianity.witnesses
     1.4%  Evangelical      -> christianity.evangelical
     1.1%  Islam/Judaism    -> other.sx
     0.8%  Other            -> other.sx
     0.3%  Buddhism/ Sikh   -> other.sx
     0.2%  Rastafarian      -> rastafari
"""

EXCLUDED = {
    "Unknown":
        "796 people, 2.4%; the census report's `Not reported`. `No Religion` is its own row.",
}

REVIEW = {
    "Islam/Judaism":
        "-> other.sx. **377 people, and the census itself prints them as one row** (Table "
        "B-10, `Islam / Judaism`), so no node on the tree holds them: splitting would invent "
        "a figure, and filing under either religion would misplace the other.",
    "Buddhism/ Sikh":
        "-> other.sx, for the same reason: one printed row, `Buddhism / Sikh`, 88 people.",
    "Other":
        "-> other.sx. 270 people; with the two combined rows the node holds 735, 2.2%.",
    "Christian":
        "-> christianity, the root. The report's `Christianity`: people who named no church.",
    "Protestant":
        "-> christianity.protestant, the node that holds the answer rather than a church. The "
        "census lists Methodist, Anglican, Baptist, Pentecostal and Adventist separately.",
}

MAP = {
    "Roman Catholic": "christianity.catholic",
    "Pentecostal": "christianity.pentecostal",
    "Methodist": "christianity.methodist",
    "No Religion": "unaffiliated",
    "Adventist": "christianity.adventist",
    "Hindu": "hinduism",
    "Baptist": "christianity.baptist",
    "Christian": "christianity",
    "Anglican": "christianity.anglican",
    "Protestant": "christianity.protestant",
    "Jehovah Witness": "christianity.witnesses",
    "Evangelical": "christianity.evangelical",
    "Islam/Judaism": "other.sx",
    "Other": "other.sx",
    "Buddhism/ Sikh": "other.sx",
    "Rastafarian": "rastafari",
}


def _key(cat):
    return " ".join(str(cat).split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
