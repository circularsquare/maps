"""Saint Kitts and Nevis 2011, *Population by Religious Belief* (Department of Statistics)
-> religiondots taxonomy.

Twenty-one categories, 47,195 people, national. Not in UNSD table 28. No new nodes beyond
`other.kn`.

    16.6%  Anglican               -> christianity.anglican
    15.8%  Methodist              -> christianity.methodist
    10.8%  Pentecostal            -> christianity.pentecostal
     8.8%  None                   -> unaffiliated
     7.4%  Church of God          -> christianity.holiness
     5.9%  Roman Catholic         -> christianity.catholic
     5.4%  Baptist                -> christianity.baptist
     5.4%  Seventh Day Adventist  -> christianity.adventist
     5.3%  Wesleyan Holiness      -> christianity.holiness
     4.8%  Moravian               -> christianity.moravian
     4.3%  Other                  -> other.kn
     2.1%  Evangelical            -> christianity.evangelical
     1.8%  Hindu                  -> hinduism
     1.7%  Brethren               -> christianity.plymouth
     1.4%  Jehovah Witness        -> christianity.witnesses
     1.3%  Rastafarian            -> rastafari
     0.5%  Muslim                 -> islam
     0.3%  Presbyterian           -> christianity.reformed.presbyterian
     0.1%  Salvation Army         -> christianity.holiness.salvation-army
     0.1%  Not stated             -> EXCLUDED
     0.1%  Bahai                  -> bahai

**`None` IS A LABEL, NOT A MISSING VALUE.** pandas reads the string `None` as NaN by default,
so every reader of `data/normalized/kn.csv` has to pass `keep_default_na=False`.
"""

EXCLUDED = {
    "Not stated":
        "56 people, 0.1%. `None` is its own row at 4,141, so a non-answer is not read as "
        "irreligion.",
}

REVIEW = {
    "Church of God":
        "-> christianity.holiness, with every bare `Church of God` in the eastern Caribbean "
        "here (ag2001, bm2010, dm2001, ms2001, ky2021). 3,495 people. The name alone cannot "
        "separate the Anderson (Holiness) and Cleveland (Pentecostal) families.",
    "Wesleyan Holiness":
        "-> christianity.holiness, as ky2021.py maps the same church. The tree now has a "
        "`christianity.holiness.wesleyan` child (from the US build); moving this row there "
        "should move Cayman's with it, so both stay at the parent until someone does both.",
    "Brethren":
        "-> christianity.plymouth, the reading this map already takes in Antigua, Barbados, "
        "Bermuda, Jamaica, St Lucia, Tuvalu and Nauru. 801 people.",
    "Other":
        "-> other.kn. 2,047 people, 4.3%. Hindu, Muslim, Rastafari and Baha'i are named "
        "separately, so this is not where the non-Christian population went.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Baptist": "christianity.baptist",
    "Bahai": "bahai",
    "Brethren": "christianity.plymouth",
    "Church of God": "christianity.holiness",
    "Evangelical": "christianity.evangelical",
    "Hindu": "hinduism",
    "Jehovah Witness": "christianity.witnesses",
    "Methodist": "christianity.methodist",
    "Moravian": "christianity.moravian",
    "Muslim": "islam",
    "Pentecostal": "christianity.pentecostal",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Rastafarian": "rastafari",
    "Roman Catholic": "christianity.catholic",
    "Salvation Army": "christianity.holiness.salvation-army",
    "Seventh Day Adventist": "christianity.adventist",
    "Wesleyan Holiness": "christianity.holiness",
    "None": "unaffiliated",
    "Other": "other.kn",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
