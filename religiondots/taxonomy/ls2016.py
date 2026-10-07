"""Afrobarometer Lesotho religion -> religiondots taxonomy.

Eleven categories at the 10 districts of the 2016 census, from rounds 4 to 9 pooled.
`sources/ls.py`'s docstring has the construction; `sources/ls.md` is the record.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Lesotho Evangelical Church":
        "-> christianity.reformed, the bare branch, as `mg2018.py` files Madagascar's FJKM. The "
        "Lesotho Evangelical Church in Southern Africa grew from the Paris Evangelical Missionary "
        "Society (1833) and is a member of the World Communion of Reformed Churches; "
        "`christianity.reformed.continental` would also fit, but its description is the Dutch and "
        "German line. The survey's `Evangelical` (rounds 4-9) and `Calvinist` (round 9, and 11 "
        "answers in round 5) are grouped into it: round 9 splits the church's members across the "
        "two boxes in every district, and together they hold 17.0-22.8% by round. Some answers of "
        "`Evangelical` may be other evangelical churches; the card has no way to say. 19.6% as "
        "drawn, against the DHS's 17.3% (2014) and 15.3% (2023-24) for women and men 15-49. "
        "**Its geography is not drawn**: it fails the split-half (+0.267 against a null of "
        "+0.412), so every district has the national proportion of what the placed churches leave.",
    "Roman Catholic":
        "-> christianity.catholic, the bare branch. 41.2% as drawn (+0.758); the DHS has 39.3% "
        "(2014) and 35.8% (2023-24) for women and men 15-49, and the survey's adults run a few "
        "points older. Thaba-Tseka 58.3%, Butha-Buthe 25.3%.",
    "Anglican":
        "-> christianity.anglican. 8.8% as drawn (+0.812), against the DHS's 7.4% (2014) and 6.3% "
        "(2023-24). It falls by round, 11.5% in 2008 to 6.0% in 2022, so the pooled level is 2.4 "
        "points over rounds 8-9, inside the 3.5-point bar. Qacha's Nek 14.9%, Leribe 13.7%.",
    "Methodist":
        "-> christianity.methodist. 1.95% as drawn (+0.770): the DHS 2023-24 has 1.3%. Quthing, "
        "Qacha's Nek, Butha-Buthe and Mokhotlong 4-5%, Thaba-Tseka 0.5%.",
    "Pentecostal":
        "-> christianity.pentecostal. 6.6% as drawn (+0.461, p=0.02). It rises by round, 4.3% to "
        "10.4%, and the pooled level is 2.4 points under rounds 8-9. Nothing witnesses it: the DHS's "
        "`Pentecostal` (23.1% in 2014, 15.4% in 2023-24) is wider than the survey's box and "
        "evidently holds the Zionist and apostolic churches too, which the DHS card does not name.",
    "Zionist and independent churches":
        "-> christianity.africaninstituted, the bare node, as Eswatini's `Zionists` and `Apostles` "
        "and South Africa's AIC cell. The survey's `Zionist Christian Church` (rounds 5-9), "
        "`Independent` (all rounds; round 4's only box for them) and `Apostolic church` (round 8 "
        "only). `Zionist Christian Church` is read as the Zionist movement, not only the Moria ZCC, "
        "since Basotho Zionist churches are many. 11.0% as drawn (+0.745); Butha-Buthe 26.3%, "
        "Quthing 16.5%, Maseru 6.2%.",
    "Other Christian":
        "-> christianity, bare. `Christian only` (0.5-4.3% by round), Baptist, Church of Christ, "
        "Jehovah's Witness, Seventh-day Adventist, Presbyterian, Lutheran, Dutch Reformed, "
        "Orthodox, Coptic, Mormon and Quaker. 7.0% as drawn, in the tail (fails the split-half).",
    "None":
        "-> unaffiliated. Step 2 of the draft \"no religion\" procedure: every round's card "
        "offers `Traditional/ethnic religion` beside `None`; `Atheist` and `Agnostic` (8 answers) "
        "are folded in. 2.6% as drawn, rising 1.3% to 3.3% by round; the DHS has 2.3% (2014) and "
        "3.7% (2023-24), most of it men. In the tail.",
    "Traditional/ethnic religion":
        "-> indigenous.african. 0.46% as drawn, in the tail, after round 6's 62 answers in "
        "Butha-Buthe, Leribe and Berea are dropped: 10-25% of those districts in that one round "
        "against 1.7% or less there in every other, a fieldwork block (sources/ls.py).",
    "Muslim":
        "-> islam, with no branch. 9 answers in six rounds (`Muslim only`, one `Shia`, one "
        "`Ismaeli`); 0.13% as drawn, in the tail. The DHS 2023-24 has 0.3%.",
    "Other":
        "-> other.ls. 0.58% as drawn, in the tail; one Bahá'í answer folded in.",
}

MAP = {
    "Roman Catholic": "christianity.catholic",
    "Lesotho Evangelical Church": "christianity.reformed",
    "Anglican": "christianity.anglican",
    "Methodist": "christianity.methodist",
    "Pentecostal": "christianity.pentecostal",
    "Zionist and independent churches": "christianity.africaninstituted",
    "Other Christian": "christianity",
    "None": "unaffiliated",
    "Traditional/ethnic religion": "indigenous.african",
    "Muslim": "islam",
    "Other": "other.ls",
}

# spec §7a-i-1: every row is drawn at the node its own category names; nothing is inferred downward.
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
