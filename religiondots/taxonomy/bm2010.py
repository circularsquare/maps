"""Bermuda 2010 census religion (UNSD DYB table 28) -> religiondots taxonomy.

**Twenty-three categories, the deepest list in the microstate tier**, 64,237 people, an exact
partition. No new nodes beyond `other.bm`.

    17.9%  None                               -> unaffiliated
    15.8%  Anglican                           -> christianity.anglican
    14.5%  Roman Catholic                     -> christianity.catholic
     8.6%  African Methodist Episcopal Church -> christianity.methodist.african
     8.3%  Non-denominational                 -> christianity.nondenominational
     6.9%  Other Religions                    -> other.bm
     6.7%  Seventh Day Adventist              -> christianity.adventist
     3.5%  Pentecostal                        -> christianity.pentecostal
     2.7%  Methodist                          -> christianity.methodist
     2.4%  Other                              -> other.bm
     2.2%  Not Stated                         -> not on the tree
     2.0%  Presbyterian                       -> christianity.reformed.presbyterian
     1.6%  Church of God                      -> christianity.holiness
     1.3%  Jehovah Witness                    -> christianity.witnesses
     1.2%  Baptist                            -> christianity.baptist
     1.1%  Salvation Army                     -> christianity.holiness.salvation-army
     1.0%  Brethren                           -> christianity.plymouth
     1.0%  Muslim                             -> islam
     0.4%  Ethiopian Orthodox                 -> christianity.oriental.ethiopian
     0.4%  Lutheran                           -> christianity.lutheran
     0.3%  Rastafarian                        -> rastafari
     0.2%  Jewish                             -> judaism
     0.2%  Baha'i                             -> bahai

**THE AFRICAN METHODIST EPISCOPAL CHURCH AT 8.6% IS THE HIGHEST AME SHARE ANYWHERE ON THIS
MAP**, including the United States, where ASARB counts it in absolute terms but nowhere near
this proportion. It arrived in Bermuda in 1870 and is the island's historically Black
Methodist tradition, distinct from the `Methodist` row below it.

**AND THE IRRELIGIOUS PLURALITY IS UNUSUAL FOR THE ATLANTIC CARIBBEAN.** 17.9% `None` is the
largest single answer in the country, well above Montserrat's 2.6%, Antigua's 4.1% and
Dominica's 6.1%. Bermuda is not really Caribbean in this respect and reads more like a
North Atlantic offshore financial centre, which is what it is.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not Stated":
        "1,407 people, 2.2%, who did not answer. `None` is its own category at 11,466, so "
        "no non-answer is being read as irreligion.",
}

REVIEW = {
    "Other Religions": "-> other.bm. See `Other` below; the two are merged on one node.",
    "Other":
        "-> other.bm as well. **The Demographic Yearbook prints both `Other Religions` "
        "(4,399) and `Other` (1,527) and does not say how they differ.** Both are residuals "
        "and both land on the same node, 5,926 people between them. Whatever the "
        "distinction was, it is not the obvious one: the Muslims, Jews, Baha'is, Rastafari "
        "and Ethiopian Orthodox are all named separately in the same table.",
    "Non-denominational":
        "-> christianity.nondenominational. 5,309 people, 8.3%, which is a large share for "
        "a category that names no body; it is the fifth-largest answer in the country and "
        "is what a reader should have in mind before treating Bermuda's Anglican plurality "
        "as the whole Protestant story.",
    "African Methodist Episcopal Church":
        "-> christianity.methodist.african, the branch node rather than a named US "
        "connexion, because the census names the tradition and not a General Conference.",
    "Church of God":
        "-> christianity.holiness, with the rest of the Caribbean here (bb2010.py, "
        "gd2021.py, jm2011.py, ms2001.py).",
    "Ethiopian Orthodox":
        "-> christianity.oriental.ethiopian. 253 people, and the only Oriental Orthodox "
        "count in the tier. In Bermuda it sits beside 212 Rastafari, which is not a "
        "coincidence: the two are historically linked through Haile Selassie.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Roman Catholic": "christianity.catholic",
    "African Methodist Episcopal Church": "christianity.methodist.african",
    "Non-denominational": "christianity.nondenominational",
    "Seventh Day Adventist": "christianity.adventist",
    "Pentecostal": "christianity.pentecostal",
    "Methodist": "christianity.methodist",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Church of God": "christianity.holiness",
    "Jehovah Witness": "christianity.witnesses",
    "Baptist": "christianity.baptist",
    "Salvation Army": "christianity.holiness.salvation-army",
    "Brethren": "christianity.plymouth",
    "Muslim": "islam",
    "Ethiopian Orthodox": "christianity.oriental.ethiopian",
    "Lutheran": "christianity.lutheran",
    "Rastafarian": "rastafari",
    "Jewish": "judaism",
    "Baha'i": "bahai",
    "None": "unaffiliated",
    "Other Religions": "other.bm",
    "Other": "other.bm",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
