"""Anguilla 2001 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Eighteen categories, 11,430 people, national, exact. No new nodes beyond `other.ai`.

    29.0%  Anglican               -> christianity.anglican
    23.9%  Methodist              -> christianity.methodist
     7.7%  Pentecostal            -> christianity.pentecostal
     7.6%  Seventh Day Adventist  -> christianity.adventist
     7.6%  Church of God          -> christianity.holiness
     7.3%  Baptist                -> christianity.baptist
     5.7%  Roman Catholic         -> christianity.catholic
     4.0%  No Religion            -> unaffiliated
     3.5%  Other Religions        -> other.ai
     0.7%  Rastafarian            -> rastafari
     0.7%  Jehovah Witness        -> christianity.witnesses
     0.5%  Evangelical            -> christianity.evangelical
     0.4%  Hindu                  -> hinduism
     0.3%  Brethren               -> christianity.plymouth
     0.3%  Not Specified          -> EXCLUDED
     0.3%  Muslim                 -> islam
     0.2%  Presbyterian           -> christianity.reformed.presbyterian
     0.1%  Jewish                 -> judaism
"""

EXCLUDED = {
    "Not Specified":
        "39 people, 0.3%. `No Religion` is its own row at 456.",
}

REVIEW = {
    "Church of God":
        "-> christianity.holiness, with every bare `Church of God` in the eastern Caribbean "
        "here (ag2001, bm2010, dm2001, ms2001). 869 people.",
    "Brethren":
        "-> christianity.plymouth, the reading this map takes in eight other countries "
        "(nr2021.py lists them). 39 people.",
    "Other Religions":
        "-> other.ai. 400 people, 3.5%.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Methodist": "christianity.methodist",
    "Pentecostal": "christianity.pentecostal",
    "Seventh Day Adventist": "christianity.adventist",
    "Church of God": "christianity.holiness",
    "Baptist": "christianity.baptist",
    "Roman Catholic": "christianity.catholic",
    "No Religion": "unaffiliated",
    "Other Religions": "other.ai",
    "Rastafarian": "rastafari",
    "Jehovah Witness": "christianity.witnesses",
    "Evangelical": "christianity.evangelical",
    "Hindu": "hinduism",
    "Brethren": "christianity.plymouth",
    "Muslim": "islam",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Jewish": "judaism",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
