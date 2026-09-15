"""British Virgin Islands 2010 census, Table 77 (religion by island) -> religiondots taxonomy.

Twenty-one categories plus `Not Stated`, 28,054 people on seven census islands. No new nodes
beyond `other.vg`.

    17.6%  Methodist                    -> christianity.methodist
    10.4%  Church of God                -> christianity.holiness
     9.5%  Anglican                     -> christianity.anglican
     9.0%  Seventh Day Adventist        -> christianity.adventist
     8.9%  Roman Catholic               -> christianity.catholic
     8.2%  Pentecostal                  -> christianity.pentecostal
     7.9%  None/No Religion             -> unaffiliated
     7.4%  Baptist                      -> christianity.baptist
     6.9%  New Testament Church of God  -> christianity.pentecostal
     4.1%  Other affiliation            -> other.vg
     2.5%  Jehovah Witness              -> christianity.witnesses
     2.4%  Not Stated                   -> EXCLUDED
     1.9%  Hindu                        -> hinduism
     0.9%  Muslim/Islam                 -> islam
     0.7%  Evangelical                  -> christianity.evangelical
     0.6%  Rastafarian                  -> rastafari
     0.3%  Moravian                     -> christianity.moravian
     0.3%  Mormon                       -> christianity.latterday
     0.2%  Presbyterian                 -> christianity.reformed.presbyterian
     0.2%  Budhaism                     -> buddhism
    <0.1%  Judaism                      -> judaism
    <0.1%  Bahai                        -> bahai

The labels are the report's own spelling (`Budhaism`, `Muslim/Islam`), because
`sources/terr.py` transcribes Table 77 rather than UNSD's national row. **The UNSD row has
Muslim at 255 where the report has 266**, and that 11 is exactly why the oracle marks this
country `NOT a partition`; the report's table closes to the person on every island.
"""

EXCLUDED = {
    "Not Stated":
        "683 people, 2.4%. `None/No Religion` is its own row at 2,230, so a non-answer is "
        "not being read as irreligion.",
}

REVIEW = {
    "Church of God":
        "-> christianity.holiness, with every other bare `Church of God` in the eastern "
        "Caribbean here (ag2001, bm2010, dm2001, ms2001, ky2021). **2,913 people, the "
        "second-largest body.** The Cleveland, Tennessee family is printed separately as "
        "`New Testament Church of God` (1,924), which is the reason this cell is not simply "
        "folded into Pentecostalism: the census already split the Pentecostal Church of God "
        "out. What stays unknown is whether the bare cell also holds Church of God of "
        "Prophecy congregations, which are Cleveland-line too.",
    "New Testament Church of God":
        "-> christianity.pentecostal, as jm2011.py maps the same body: the Caribbean name of "
        "the Church of God (Cleveland, Tennessee).",
    "Other affiliation":
        "-> other.vg. 1,153 people, 4.1%, and 1,072 of them on Tortola. The form already "
        "names Hindu, Muslim, Buddhist, Jewish, Baha'i and Rastafari, so the non-Christian "
        "population is not pooled here.",
    "Mormon":
        "-> christianity.latterday, the family node, as ee2021.py maps `Mormon`.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Church of God": "christianity.holiness",
    "Evangelical": "christianity.evangelical",
    "Methodist": "christianity.methodist",
    "Moravian": "christianity.moravian",
    "New Testament Church of God": "christianity.pentecostal",
    "Pentecostal": "christianity.pentecostal",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Roman Catholic": "christianity.catholic",
    "Seventh Day Adventist": "christianity.adventist",
    "Jehovah Witness": "christianity.witnesses",
    "Baptist": "christianity.baptist",
    "Bahai": "bahai",
    "Hindu": "hinduism",
    "Judaism": "judaism",
    "Mormon": "christianity.latterday",
    "Muslim/Islam": "islam",
    "Rastafarian": "rastafari",
    "Budhaism": "buddhism",
    "Other affiliation": "other.vg",
    "None/No Religion": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
