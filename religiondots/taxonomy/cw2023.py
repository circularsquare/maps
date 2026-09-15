"""Curaçao Census 2023, Table D-5 (population by religion, age group and sex) -> religiondots.

Fourteen categories, 155,826 people, national, exact. No new nodes beyond `other.cw`.

    68.2%  Roman Catholic            -> christianity.catholic
     8.7%  I don't have a religion   -> unaffiliated
     5.5%  Pentecostal Church        -> christianity.pentecostal
     3.5%  Other                     -> other.cw
     3.0%  Adventist                 -> christianity.adventist
     2.6%  Protestant                -> christianity.protestant
     2.6%  Not reported              -> EXCLUDED
     2.0%  Jehovah's Witness         -> christianity.witnesses
     1.9%  Evangelical               -> christianity.evangelical
     0.8%  Hinduism                  -> hinduism
     0.4%  Islam                     -> islam
     0.4%  Methodist                 -> christianity.methodist
     0.2%  Judaism                   -> judaism
     0.1%  Anglican                  -> christianity.anglican

**HINDUISM 1,211 IN 2023 AGAINST 3,058 IN 2011 IS THE 2011 TABLE'S PROBLEM, NOT THIS ONE'S.**
§11ap asked for a look. The 2011 D-5 very probably has its `Hinduidm` and `Jehova's Witness`
rows swapped: 2011's "Hindu" row (3,058, 62% female, 13% aged 65+) has the size and shape of
2023's Witnesses (3,184, 62% female, 31% aged 65+), and 2011's "Witness" row (1,222, 54% male)
has the size and shape of 2023's Hindus (1,211, 49% male). Read that way, both groups grow
slightly between the censuses instead of one falling by 60% while the other rises by 160%. It
does not touch this build, which draws 2023 only; `sources/terr.md` has the figures.
"""

EXCLUDED = {
    "Not reported":
        "4,088 people, 2.6%. `I don't have a religion` is its own row at 13,634.",
}

REVIEW = {
    "Protestant":
        "-> christianity.protestant, literally. On Curaçao the word mostly means the United "
        "Protestant Congregation (Verenigde Protestantse Gemeente), the 1825 union of the "
        "Dutch Reformed and Lutheran congregations, which no single Reformed node describes. "
        "Aruba's form said `Protestant, reformed` and is mapped to the Reformed family "
        "(aw2010.py); this label does not say so. 4,096 people.",
    "Other":
        "-> other.cw. 5,408 people, 3.5%. The 2011 table printed Mormonism (86), Baptism (23) "
        "and an unspecified `Christian*` (1,380) separately; 2023 does not, so they are "
        "presumably in here.",
    "Evangelical":
        "-> christianity.evangelical, the node for the answer itself.",
}

MAP = {
    "Adventist": "christianity.adventist",
    "Anglican": "christianity.anglican",
    "Evangelical": "christianity.evangelical",
    "Hinduism": "hinduism",
    "Islam": "islam",
    "Jehovah's Witness": "christianity.witnesses",
    "Judaism": "judaism",
    "Methodist": "christianity.methodist",
    "Pentecostal Church": "christianity.pentecostal",
    "Protestant": "christianity.protestant",
    "Roman Catholic": "christianity.catholic",
    "I don't have a religion": "unaffiliated",
    "Other": "other.cw",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
