"""Dominica 2001 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Fourteen categories, 68,635 people, an exact partition. No new nodes beyond `other.dm`.

    61.4%  Roman Catholic             -> christianity.catholic
     7.1%  Other Evangelical Churches -> christianity.evangelical
     6.1%  No Religion                -> unaffiliated
     6.0%  Seventh Day Adventist      -> christianity.adventist
     5.6%  Pentecostal                -> christianity.pentecostal
     4.1%  Baptist                    -> christianity.baptist
     3.7%  Methodist                  -> christianity.methodist
     1.3%  Rastafarian                -> rastafari
     1.2%  Church of God              -> christianity.holiness
     1.2%  Jehovah Witness            -> christianity.witnesses
     1.0%  Not Specified              -> not on the tree
     0.6%  Anglican                   -> christianity.anglican
     0.4%  Other                      -> other.dm
     0.2%  Islam                      -> islam

**DOMINICA IS THE ODD ONE OUT IN THE EASTERN CARIBBEAN AND THE REASON IS FRENCH.** 61.4%
Roman Catholic against Antigua's 10.4% and Montserrat's 11.6%, and 0.6% Anglican against
Antigua's 25.7%. The island changed hands repeatedly and the French missionary period left a
Catholic majority that British rule never displaced; St Lucia, next door and drawn from its
own census, has the same shape for the same reason.

**AND THE RASTAFARI SHARE IS THE HIGHEST IN THIS TIER**, 879 people or 1.3%, ahead of
Antigua's 1.3% by a hair and Bermuda's 0.3%. Dominica's Rastafari community is long
established and the census counts it separately rather than folding it into `Other`.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not Specified":
        "720 people, 1.0%, who did not answer. `No Religion` is its own category at 4,165.",
}

REVIEW = {
    "Other Evangelical Churches":
        "-> christianity.evangelical rather than `other.dm`, because the category names a "
        "family and not a residual: these are 4,882 people known to be in evangelical "
        "Protestant congregations whose individual bodies the census does not list. "
        "`other.dm` is kept for the 252 the census itself calls `Other`.",
    "Church of God":
        "-> christianity.holiness, with the rest of the Caribbean here.",
    "Islam":
        "-> islam. Printed with a trailing space in the DYB, normalised by `_key`. 139 "
        "people.",
}

MAP = {
    "Roman Catholic": "christianity.catholic",
    "Other Evangelical Churches": "christianity.evangelical",
    "Seventh Day Adventist": "christianity.adventist",
    "Pentecostal": "christianity.pentecostal",
    "Baptist": "christianity.baptist",
    "Methodist": "christianity.methodist",
    "Rastafarian": "rastafari",
    "Church of God": "christianity.holiness",
    "Jehovah Witness": "christianity.witnesses",
    "Anglican": "christianity.anglican",
    "Islam": "islam",
    "No Religion": "unaffiliated",
    "Other": "other.dm",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
