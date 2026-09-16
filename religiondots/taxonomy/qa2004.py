"""Qatar Census 2004 religion (Planning Council, population Table 6) -> religiondots taxonomy.

Three cells on the ten municipalities of 2004, 744,029 people (everyone present on 16 March 2004,
labour gatherings included), about 74,000 a municipality. sources/qa.md is the write-up.

The Qatari form's religion column offers three answers: 1 Muslim, 2 Christian, 3 Other. There is
no box for no religion or for not stated, and Table 6's total is Table 1's whole population. The
religion of non-Qataris is a weighted estimate from the census's sample, calibrated to the counted
population by municipality and sex (sources/qa.py). No table read here gives the non-Qatari share.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, no branch. 576,391 people, 77.47%. The form asks only Muslim, so a Sunni or "
        "Shia node would be an inference; nothing published splits them. Qataris are counted in "
        "full and every non-Qatari's answer comes from the weighted sample. Lowest in Mesaieed "
        "(38.51%) and Al Khor (56.10%), highest in Umm Salal (87.80%).",
    "Christian":
        "-> christianity, the bare branch; no church is asked. 63,212 people, 8.50%. 64% men. "
        "Jeryan Al Batna is 33.98% Christian, 2,222 of its 2,269 Christians men, and Table 2 puts "
        "4,214 of that municipality's 5,633 men in zone 82, Rawdat Rashed, which reads as a large "
        "male workforce rather than a settled community; no table says which employer or camp. "
        "Drawn as printed.",
    "Other":
        "-> other.qa, a new node. 104,426 people, 14.04%, 89% of them men. **The form's code 3 "
        "Other, with no box for no religion**, so it holds Qatar's Hindus, Buddhists and anyone "
        "else neither Muslim nor Christian, and anyone with no religion could only be recorded "
        "here. Not a lumped no-religion box in the sense of the draft procedure (its label is "
        "Other and it is the only box for the Hindus and Buddhists of a mostly South and "
        "South-East Asian workforce), so `unknown` would hide what the category mostly is. It is "
        "46.47% of Mesaieed and 32.83% of Al Khor, and 96% men in Al Rayyan, whose zone 57, the "
        "Industrial Area, held 62,555 men and 57 women.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "Other": "other.qa",
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
