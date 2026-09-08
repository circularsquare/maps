"""Montserrat 2001 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Eleven categories, 4,303 people, an exact partition. No new nodes: the whole vocabulary was
already decided by the other Caribbean countries here.

    21.8%  Anglican              -> christianity.anglican
    17.0%  Methodist             -> christianity.methodist
    14.2%  Pentecostal           -> christianity.pentecostal
    11.6%  Roman Catholic        -> christianity.catholic
    10.8%  Unknown               -> not on the tree
    10.6%  Seventh Day Adventist -> christianity.adventist
     5.8%  Other Religions       -> other.ms
     3.7%  Church of God         -> christianity.holiness
     2.6%  No Religion           -> unaffiliated
     1.4%  Rastafarian           -> rastafari
     0.7%  Hindu                 -> hinduism

**THE POPULATION IS THE POINT.** This census was taken six years after the Soufriere Hills
eruption began, which buried Plymouth and drove roughly two thirds of Montserrat's people
off the island. 4,303 is what was left. The religious composition is an ordinary Leeward
Islands one; what is unusual is that it describes so few people, and that the Kontur
placement layer puts them all in the northern third of the island, which is where the
exclusion zone allows anyone to live.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Unknown":
        "465 people, **10.8%**, the largest non-answer share in this tier. A coverage "
        "residual rather than a refusal, and the census keeps it apart from `No Religion` "
        "(110) and `Other Religions` (251), so nothing is merged here.",
}

REVIEW = {
    "Church of God":
        "-> christianity.holiness, which is bb2010.py's, gd2021.py's and jm2011.py's call "
        "for the same words. The name covers both Holiness and Pentecostal bodies across "
        "the Caribbean and the parent is where the region's other censuses put it; "
        "consistency across neighbours matters more here than resolving 158 people.",
    "Other Religions":
        "-> other.ms. The census names the Hindus and Rastafari separately, so this is a "
        "genuine tail rather than a stand-in for the non-Christian population.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Methodist": "christianity.methodist",
    "Pentecostal": "christianity.pentecostal",
    "Roman Catholic": "christianity.catholic",
    "Seventh Day Adventist": "christianity.adventist",
    "Church of God": "christianity.holiness",
    "Rastafarian": "rastafari",
    "Hindu": "hinduism",
    "No Religion": "unaffiliated",
    "Other Religions": "other.ms",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
