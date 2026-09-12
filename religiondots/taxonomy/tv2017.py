"""Tuvalu 2017 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Eleven categories, 10,507 people, an exact partition. **One new node**, the state church.

    85.9%  Ekalesia Kelisiano Tuvalu -> christianity.reformed.congregational.ekt    <- new
     2.8%  Brethren Assembly         -> christianity.plymouth
     2.6%  Other                     -> other.tv
     2.5%  Seventh Day Adventist     -> christianity.adventist
     1.5%  Baha'i                    -> bahai
     1.5%  Assembly of God           -> christianity.pentecostal.trinitarian
     1.5%  Jehovah Witness           -> christianity.witnesses
     0.9%  Latter Day Saints         -> christianity.latterday
     0.5%  Catholic                  -> christianity.catholic
     0.2%  None                      -> unaffiliated
     0.1%  Refused to answer         -> not on the tree

**85.9% IS THE LARGEST SHARE ANY SINGLE CHURCH HOLDS IN ANY COUNTRY ON THIS MAP.** Ekalesia
Kelisiano Tuvalu is the established church under Tuvalu's constitution, LMS-descended by way
of Samoan missionaries from 1861. Its siblings here are the Cook Islands Christian Church and
Ekalesia Niue.

**AND THE TAIL IS UNUSUALLY SPECIFIC FOR ITS SIZE.** A country of 10,507 names the Brethren,
the Adventists, the Baha'is, the Assembly of God, the Witnesses, the Latter-day Saints and 53
Catholics separately. At 1 dot = 1,000 people almost none of that draws, which is a fact
about the dot value rather than about the source.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Refused to answer":
        "14 people. A refusal is not a religion, and Tuvalu asks it apart from `None` (26), "
        "so the two are not merged here either. Between them they are 0.4% of the country.",
}

REVIEW = {
    "Brethren Assembly":
        "-> christianity.plymouth, with lc2022.py's, gd2021.py's and bb2010.py's `Brethren`. "
        "The Brethren Assembly in Tuvalu is an Open Brethren body and the second-largest "
        "church in the country at 296 people.",
    "None":
        "-> unaffiliated. 26 people, and Tuvalu is the least irreligious country drawn "
        "anywhere here at 0.25%.",
}

MAP = {
    "Ekalesia Kelisiano Tuvalu": "christianity.reformed.congregational.ekt",
    "Brethren Assembly": "christianity.plymouth",
    "Seventh Day Adventist": "christianity.adventist",
    "Baha'i": "bahai",
    "Assembly of God": "christianity.pentecostal.trinitarian",
    "Jehovah Witness": "christianity.witnesses",
    "Latter Day Saints": "christianity.latterday",
    "Catholic": "christianity.catholic",
    "None": "unaffiliated",
    "Other": "other.tv",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
