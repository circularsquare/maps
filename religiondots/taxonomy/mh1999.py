"""Marshall Islands 1999 census religion (UNSD DYB table 28) -> religiondots taxonomy.

**Four categories — the shallowest table drawn anywhere on this map** — 50,848 people, an
exact partition. No new nodes beyond `other.mh`.

    54.8%  Protestant      -> christianity.protestant
    25.8%  Assembly of God -> christianity.pentecostal.trinitarian
    11.1%  Other           -> other.mh
     8.4%  Roman Catholic  -> christianity.catholic

**READ THIS COUNTRY AS A FOUR-WAY SPLIT AND NOTHING FINER.** Every religious body in the
Marshall Islands other than the Assembly of God and the Catholic church is inside one of two
cells. The 54.8% `Protestant` is overwhelmingly the United Church of Christ, the Congregational
church the American Board planted in 1857 and the direct sibling of the national churches this
map draws separately for Tuvalu, Niue and the Cook Islands. **It cannot be filed there**,
because the census says `Protestant` and naming the body would assert something the source
does not.

**AND `Other` AT 11.1% IS THE LEAST SATISFYING NODE IN THE TIER.** It certainly contains
Bukot nan Jesus, the indigenous Marshallese church, and the country's Baha'i community, and
probably its Latter-day Saints; the source names none of them.

**1999 IS THE OLDEST CENSUS ON THIS MAP.** The Marshall Islands ran censuses in 2011 and 2021
and neither forwarded a religion tabulation to UNSD, and `rmi-data.sprep.org` is a 403
(queue.md). So this is drawn because §3.9c's variety floor was retired for the microstate
tier, not because it is a good table; `how=` says the year on the country's own panel.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Protestant":
        "-> christianity.protestant, the 'named no body' node, for 27,849 people, 54.8% of "
        "the country. **The body is known and is deliberately not asserted**: it is the "
        "United Church of Christ in the Marshall Islands, an American Board Congregational "
        "mission of 1857, and on the evidence of every neighbouring country it is most of "
        "this cell. Filing it on `christianity.reformed.congregational` would put a name on "
        "a census category that carries none, which is the same call mm2014.py makes for "
        "Myanmar's undivided `Christian`. If a 2011 or 2021 Marshallese table is ever "
        "found, this is the row it would change.",
    "Assembly of God":
        "-> christianity.pentecostal.trinitarian, following bs2022.py. **25.8% is the "
        "highest Assemblies of God share of any country on this map**, and it is a real "
        "feature of Micronesia rather than a coding artefact.",
    "Other":
        "-> other.mh, holding 5,632 people the source does not describe at all. See the "
        "module docstring; this node is a limit of the census, not a tail.",
}

MAP = {
    "Protestant": "christianity.protestant",
    "Assembly of God": "christianity.pentecostal.trinitarian",
    "Roman Catholic": "christianity.catholic",
    "Other": "other.mh",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
