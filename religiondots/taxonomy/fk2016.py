"""Falkland Islands Census 2016, Table 6 (religion, sex and location) -> religiondots taxonomy.

Eight categories, 3,198 people on three locations (Stanley, Camp, Mount Pleasant Complex).
No new nodes beyond `other.fk`.

    57.1%  Christian           -> christianity
    35.4%  No Religion         -> unaffiliated
     6.0%  Not Specified       -> EXCLUDED
     0.6%  Other               -> other.fk
     0.4%  Jehovah's Witness   -> christianity.witnesses
     0.3%  Buddhist            -> buddhism
     0.2%  Muslim              -> islam
     0.1%  Baha'i              -> bahai

**CHRISTIANS ARE ONE CELL.** The report prints no denomination, so 1,825 people go to the
Christianity root and nothing finer. The 2006 round in UNSD's table is the same shape
(Christian 1,985), and the 2021 census asked no religion at all (sources.md §11ap).
"""

EXCLUDED = {
    "Not Specified":
        "192 people, 6.0%. Separate from `No Religion` (1,131), so not read as irreligion.",
}

REVIEW = {
    "Christian":
        "-> christianity, the root, because the census prints one undivided cell. The "
        "islands' churches are Anglican (Christ Church Cathedral), Catholic (St Mary's) and "
        "the United Free Church, but the census does not count them, so no share is drawn "
        "for any of them.",
    "Jehovah's Witness":
        "-> christianity.witnesses. Printed apart from `Christian` by the census itself, so "
        "kept apart; 13 people.",
}

MAP = {
    "Baha'i": "bahai",
    "Buddhist": "buddhism",
    "Christian": "christianity",
    "Jehovah's Witness": "christianity.witnesses",
    "Muslim": "islam",
    "No Religion": "unaffiliated",
    "Other": "other.fk",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
