"""
SLiCA Greenland 2003-2006 (`sources/gl.py`) -> religiondots taxonomy.

The survey asked one thing: "Do you consider yourself to be a Christian?". 98% of the
Greenland-born said yes (SLiCA Results Tables, Table 162). The church roll splits the yes (spec
3.1 allows a roll to split a self-identified category, never to add to it):

    96.38%  Church of Greenland (Lutheran)              -> christianity.lutheran
     1.62%  Christian, other or church not established  -> christianity
     2.00%  Not Christian                               -> unknown

Every row is `modelled`: one national mix laid on each municipality's Greenland-born residents.
"""

EXCLUDED = {}

REVIEW = {
    "Church of Greenland (Lutheran)":
        "-> christianity.lutheran, Denmark's node (taxonomy/dk2024.py), because the Church of "
        "Greenland is the Greenlandic diocese of the Church of Denmark and its members are "
        "counted in Statistics Greenland's BEXKIRK as Church of Denmark members. The magnitude "
        "is the roll's (96.38% of the Greenland-born on 1 January 2026), not the survey's: "
        "SLiCA only asked whether people are Christian. The roll is below the 98% who said "
        "yes, so it fits inside it. The roll is 2026 and the survey 2003-2006; the roll among "
        "the Greenland-born fell from 97.5% in 2012 to 96.4% in 2026, so self-identification "
        "has very likely drifted down too since SLiCA, which this map cannot show.",
    "Christian, other or church not established":
        "-> christianity, the family node. The 1.62 points between SLiCA's 98% Christian and "
        "the 2026 roll: Greenland-born people who call themselves Christian and are not on the "
        "Church of Denmark's roll (Pentecostals, Catholics, Jehovah's Witnesses and others), "
        "plus whatever part of the gap is the twenty years between the two sources. Neither "
        "source can say which, so the family node and not a church.",
    "Not Christian":
        "-> unknown, not unaffiliated. SLiCA's no is everyone who does not consider themselves "
        "Christian: no religion, Inuit belief held apart from Christianity, Baha'i and other "
        "religions are all in it, and the question cannot tell them apart. branches.py's test "
        "for `unknown` is that the source counted these people and what they practise is not "
        "determinable from it, which is this case. 2% of the Greenland-born, about 1,000 "
        "people, one dot.",
}

MAP = {
    "Church of Greenland (Lutheran)": "christianity.lutheran",
    "Christian, other or church not established": "christianity",
    "Not Christian": "unknown",
}

# No COLUMNS dict (spec 7a-i-1): every row is `modelled`, as in cu2016.py and eg2022.py.


def resolve(category):
    """religiondots branch for a gl.csv category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
