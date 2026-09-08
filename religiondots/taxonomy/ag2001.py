"""Antigua and Barbuda 2001 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Twenty-one categories, 76,886 people. No new nodes beyond `other.ag`.

    25.7%  Anglican              -> christianity.anglican
    12.3%  Seventh Day Adventist -> christianity.adventist
    10.6%  Pentecostal           -> christianity.pentecostal
    10.5%  Moravian              -> christianity.moravian
    10.4%  Roman Catholic        -> christianity.catholic
     7.9%  Methodist             -> christianity.methodist
     4.9%  Baptist               -> christianity.baptist
     4.5%  Church of God         -> christianity.holiness
     4.1%  No Religion           -> unaffiliated
     2.6%  Evangelical           -> christianity.evangelical
     1.7%  Not Declared          -> not on the tree
     1.6%  Jehovah Witness       -> christianity.witnesses
     1.3%  Rastafarian           -> rastafari
     0.6%  Salvation Army        -> christianity.holiness.salvation-army
     0.3%  Islam                 -> islam
     0.3%  Presbyterian          -> christianity.reformed.presbyterian
     0.3%  Brethren              -> christianity.plymouth
     0.2%  Hindu                 -> hinduism
     0.2%  Other                 -> other.ag
     0.1%  Spiritualist          -> spiritualism
     0.1%  Baha'i                -> bahai

**THE MORAVIANS AT 10.5% ARE THE THING TO SEE HERE.** 8,057 people, the fourth-largest church
in the country, and the largest Moravian share of any country on this map. The mission dates
from 1756 and was aimed at the enslaved population, which is why Antigua, Barbados and the
Danish Virgin Islands have Moravian communities and most of the rest of the Caribbean does
not.

**THE PARTITION IS OFF BY THREE PEOPLE AND THAT IS THE YEARBOOK'S.** The 21 categories sum to
76,889 against a stated total of 76,886. It is 0.004%, it is in the published table rather
than in this code, and `sources/micro.py` allows exactly this much slack for `ag` and no other
country so that a real break would still fail the build.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not Declared":
        "1,329 people, 1.7%, who declined. `No Religion` is its own category at 3,145, so "
        "a refusal is not being read as irreligion.",
}

REVIEW = {
    "Spiritualist":
        "-> spiritualism, the root, and **this is the least certain call in the file**. In "
        "an eastern Caribbean census the word can mean the Spiritual Baptist (Shouter) "
        "tradition, which gd2021.py files on `afrodiasporic.spiritualbaptist` from a "
        "category that says `SPIRITUAL BAPTIST` in full. Antigua's says only "
        "`Spiritualist`, so the literal reading is taken. 66 people, which draws no dot; "
        "if Antigua's own census report is ever read directly this is the row to check.",
    "Evangelical":
        "-> christianity.evangelical, the 'unspecified' node, with vc2012.py's "
        "`Evangelical Christian` and gd2021.py's `EVANGELICAL`.",
    "Church of God":
        "-> christianity.holiness, with the rest of the Caribbean here.",
    "Islam":
        "-> islam. The DYB prints this category with a trailing space; `_key` normalises "
        "whitespace, which is the house convention, so the key here is the tidied form. "
        "228 people.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Seventh Day Adventist": "christianity.adventist",
    "Pentecostal": "christianity.pentecostal",
    "Moravian": "christianity.moravian",
    "Roman Catholic": "christianity.catholic",
    "Methodist": "christianity.methodist",
    "Baptist": "christianity.baptist",
    "Church of God": "christianity.holiness",
    "Evangelical": "christianity.evangelical",
    "Jehovah Witness": "christianity.witnesses",
    "Rastafarian": "rastafari",
    "Salvation Army": "christianity.holiness.salvation-army",
    "Islam": "islam",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Brethren": "christianity.plymouth",
    "Hindu": "hinduism",
    "Spiritualist": "spiritualism",
    "Baha'i": "bahai",
    "No Religion": "unaffiliated",
    "Other": "other.ag",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
