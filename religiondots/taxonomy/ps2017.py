"""Palestine PCBS census 2017 religion (Preliminary Results, Table 3) -> religiondots taxonomy.

Four answers and a total at governorate, sixteen units, 4,665,426 Palestinians counted
(sources/ps.py). The form's Religion item has three codes, 1 Muslim, 2 Christian, 3 Other, asked
of Palestinians only, and no code for no answer.

    98.93%  Islam        -> islam
     1.00%  Christian    -> christianity
     0.03%  Other        -> other.ps  (a NEW node, the per-country residual)
     0.03%  Not Stated   EXCLUDED
            Total        EXCLUDED (the universe row)

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not Stated":
        "1,509 people, 0.03% of Palestinians counted. The form has no code for no answer, so "
        "these are blank or unreadable answers. Leans nowhere that can be shown: 442 are in "
        "Ramallah & Al-Bireh and 299 in Jerusalem, the two governorates with the most "
        "Christians after Bethlehem, and 263 in Nablus.",
    "Total":
        "Table 3's own total, the universe: Palestinians counted, 4,665,426. Kept in the "
        "normalised file so tools/gap_share.py can confirm the residual from both ends.",
}

REVIEW = {
    "Christian":
        "-> christianity, the parent, because the form has one Christian box. 46,850 people, "
        "1.00%. The churches are Greek Orthodox, Latin Catholic, Melkite Greek Catholic, "
        "Lutheran, Anglican, Armenian, Syriac and others, and nothing PCBS publishes separates "
        "them. **Bethlehem holds 49.4% of them** (23,165, 10.9% of the governorate), then "
        "Ramallah & Al-Bireh 10,255 (3.3%) and Jerusalem 8,558 (2.2%), Jenin 2,699 (Zababdeh "
        "and Jenin town are in it) and Gaza 1,082. The whole Gaza Strip has 1,138.",
    "Islam":
        "-> islam, no branch, because the form gives none. 4,615,683 people, 98.93%.",
    "Other":
        "-> other.ps, a NEW per-country residual node, following other.cg and other.sl. 1,384 "
        "people, 0.03%. The census does not say what it holds. **Nablus has 361 of them**, and "
        "the Samaritan community of Kiryat Luza on Mount Gerizim is in Nablus governorate, so "
        "they may be much of that 361; nothing in the book says so and it is not asserted. Jerusalem has 594. "
        "With no box for no religion on the form, a Palestinian with none could only answer "
        "Other or leave it blank, so some of the cell may be that. Not sent to `unaffiliated` "
        "or to any named body.",
}

MAP = {
    "Islam": "islam",
    "Christian": "christianity",
    "Other": "other.ps",
}

# spec §7a-i-1: every row is measured at the node it is drawn on.
COLUMNS = {v: v for v in MAP.values()}


def _key(cat):
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}
_FOLDED_EXCLUDED = {_key(k) for k in EXCLUDED}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = str(cat)
    if c in EXCLUDED or _key(c) in _FOLDED_EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))
