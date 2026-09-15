"""Caribbean Netherlands, CBS table 82868NED (Omnibus survey 2021, persons 15+) -> religiondots.

Twelve published answers plus the withheld remainder, per island (Bonaire, Sint Eustatius,
Saba). The labels are CBS's Dutch column titles. `sources/bq.py` turns the shares into counts
on each island's population of 1 January 2022. No new nodes beyond `other.bq`.

    Geen godsdienst     -> unaffiliated
    Rooms Katholiek     -> christianity.catholic
    Pinkstergemeente    -> christianity.pentecostal
    Protestant          -> christianity.protestant
    Adventist           -> christianity.adventist
    Methodist           -> christianity.methodist
    Evangelisch         -> christianity.evangelical
    Anglicaans          -> christianity.anglican
    Islam               -> islam
    Jehova              -> christianity.witnesses
    Hindoeïsme          -> hinduism
    Anders              -> other.bq
    Niet gepubliceerd   -> EXCLUDED

The island shares as published for 2021 (`.` = withheld):

    Bonaire         Catholic 60.3, none 15.2, evangelical 6.0, other 5.8, Pentecostal 4.9,
                    Protestant 3.3, Adventist 2.0, Witnesses 1.4, Anglican 0.4
    Sint Eustatius  Methodist 24.8, Catholic 23.3, none 22.1, Adventist 18.9, Pentecostal 3.2,
                    evangelical 2.2, other 1.9, Anglican 1.7, Witnesses 1.2
    Saba            Catholic 50.1, none 20.4, Anglican 8.9, Pentecostal 6.7, other 6.1,
                    Adventist 3.6, evangelical 2.4
"""

EXCLUDED = {
    "Niet gepubliceerd":
        "Not a CBS label: the share of each island that the published cells do not account "
        "for, 100 minus their sum. It is mostly cells CBS withheld as too unreliable to print "
        "(Methodists on Bonaire and Saba, Protestants on Sint Eustatius and Saba) plus "
        "rounding to one decimal. 0.7% of Bonaire, 0.7% of Sint Eustatius, 1.8% of Saba. It "
        "cannot be assigned to any one of the withheld religions.",
}

REVIEW = {
    "Protestant":
        "-> christianity.protestant, literally, as cw2023.py. The label does not say which "
        "Protestant church; on Bonaire it is most likely the Protestant congregation of the "
        "Dutch colonial church.",
    "Methodist":
        "-> christianity.methodist. **24.8% of Sint Eustatius**, the largest answer there. "
        "Statia and Saba were settled from the English-speaking Leewards, which is why their "
        "shape is Sint Maarten's and Anguilla's and not Bonaire's.",
    "Anders":
        "-> other.bq, CBS's `other`. 5.8% of Bonaire, 1.9% of Sint Eustatius, 6.1% of Saba.",
    "Jehova":
        "-> christianity.witnesses; CBS's short form for Jehovah's Witnesses.",
}

MAP = {
    "Geen godsdienst": "unaffiliated",
    "Rooms Katholiek": "christianity.catholic",
    "Pinkstergemeente": "christianity.pentecostal",
    "Protestant": "christianity.protestant",
    "Adventist": "christianity.adventist",
    "Methodist": "christianity.methodist",
    "Evangelisch": "christianity.evangelical",
    "Anglicaans": "christianity.anglican",
    "Islam": "islam",
    "Jehova": "christianity.witnesses",
    "Hindoeïsme": "hinduism",
    "Anders": "other.bq",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
