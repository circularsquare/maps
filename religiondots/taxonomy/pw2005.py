"""Palau 2005 census religion (UNSD DYB table 28) -> religiondots taxonomy.

Nine categories, 19,907 people, an exact partition. **One new ROOT, `modekngei`.**

    49.4%  Catholic                -> christianity.catholic
    23.2%  Protestant              -> christianity.protestant
     8.7%  Modekngei               -> modekngei                     <- new
     8.1%  Other                   -> other.pw
     5.3%  Seventh Day Adventist   -> christianity.adventist
     2.5%  Other Protestant        -> christianity.protestant
     1.1%  Jehovah Witness         -> christianity.witnesses
     1.1%  None or Refused         -> unaffiliated
     0.7%  Mormon                  -> christianity.latterday

**MODEKNGEI IS WHY PALAU IS WORTH DRAWING.** It is Palau's own religion, founded around 1915
by Temedad, fusing Palauan spirit belief with Christian elements, and suppressed under the
Japanese administration. At 8.7% it is a larger share of its country than any other
indigenous religion on this map. taxonomy/branches.py says why it is a root rather than a
child of `indigenous`.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "None or Refused":
        "-> unaffiliated, and the merge is the publisher's rather than a choice made here. "
        "**The 1995 census of the same country asked them apart and the split settles it**: "
        "`None` 1,577 against `Refused to answer` 7, so 99.6% of a merged cell of this kind "
        "in Palau is people with no religion. 222 people in 2005, which draws no dot at "
        "1:1,000 either way; the evidence is recorded because the reasoning, not the "
        "number, is what would matter if a finer census arrives.",
    "Protestant":
        "-> christianity.protestant, the 'named no body' node. Palau's Protestants are "
        "largely the Evangelical Church of Palau, a United Church of Christ descendant of "
        "the American Board mission, but the census names no body and filing them on a "
        "specific one would assert it.",
    "Other Protestant":
        "-> christianity.protestant as well, joining the row above rather than going to "
        "`other.pw`. The category says Protestant, so the family is known and only the body "
        "is not, which is exactly what that node is for.",
    "Other":
        "-> other.pw. Large for this tier at 8.1%, and the reason is visible in the "
        "census's own history: the 1995 round named twelve categories including Baha'i, "
        "Assembly of God and Church of Christ, and 2005 names nine. Those three are inside "
        "this cell.",
}

MAP = {
    "Catholic": "christianity.catholic",
    "Protestant": "christianity.protestant",
    "Other Protestant": "christianity.protestant",
    "Modekngei": "modekngei",
    "Seventh Day Adventist": "christianity.adventist",
    "Jehovah Witness": "christianity.witnesses",
    "Mormon": "christianity.latterday",
    "None or Refused": "unaffiliated",
    "Other": "other.pw",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
