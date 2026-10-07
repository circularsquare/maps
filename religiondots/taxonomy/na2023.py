"""Afrobarometer Namibia religion -> religiondots taxonomy.

Ten categories at the 14 regions of the 2023 census, from rounds 4 to 9 pooled, with Lutheran and
Anglican taken from rounds 5 and 6 inside the pool they trade with. `sources/na.py`'s docstring has
the construction; `sources/na.md` is the record.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Lutheran":
        "-> christianity.lutheran, bare. Three Lutheran churches answer to this box: the "
        "Evangelical Lutheran Church in Namibia (ELCIN) in the north, the Evangelical Lutheran "
        "Church in the Republic of Namibia (ELCRN) in the centre and south, and the small German "
        "church; the survey does not tell them apart. 40.5% as drawn. **Taken from rounds 5 and 6 "
        "only**, as a share of the `Christian, other` pool, because those two rounds put Lutherans "
        "at 42.8% and 41.9% where the 2013 DHS has ELCIN alone at 43.9%, and the other four rounds "
        "at 20-24%, with the difference coded `Evangelical`, `Christian only` or `Anglican` "
        "depending on the round. Within rounds 5 and 6 the split-half is +0.725 (p=0.007). The "
        "drawn level is 3.4 points under ELCIN alone, because the pool's level is all six rounds'.",
    "Anglican":
        "-> christianity.anglican. 6.6% as drawn, from rounds 5 and 6 as Lutheran is (+0.578, "
        "p=0.02): Ohangwena 30.6%, the Anglican diocese's Oukwanyama heartland, and under 10% "
        "everywhere else. Nothing outside the survey names Anglicans alone (the DHS prints "
        "`Protestant/Anglican` at 18.6%, an upper bound). Across all six rounds it swings 6.8-15.5%, "
        "the swing being round 8's 30-34% in Omusati and Oshikoto, which no other round reproduces.",
    "Roman Catholic":
        "-> christianity.catholic, the bare branch, as `mg2018.py` and `zm2022.py` file a card's "
        "Catholic. 22.6% as drawn, all six rounds (+0.757); the DHS has 21.6% in 2013 and 22.4% in "
        "2006-07. Kavango 47.2%, //Kharas 33.1%, Omaheke 32.2%.",
    "Seventh Day Adventist":
        "-> christianity.adventist.sda. 2.9% as drawn, all six rounds (+0.573), Zambezi 41.0%. "
        "**Drawn short**: the 2013 DHS has 4.5%. Its level holds across rounds (1.9-3.9%), so the "
        "gap is the survey's against the DHS, not the fieldwork's; most of it is likely Zambezi.",
    "Pentecostal":
        "-> christianity.pentecostal. 3.5% as drawn, all six rounds (+0.670), holding 2.4-5.1% by "
        "round; Omaheke 11.3%. No outside source names Pentecostals, so this is Cameroon's case: "
        "a church drawn because the probing that moves Lutherans does not move it. Rounds 8 and 9 "
        "add the gloss `Born Again and/or Saved` to the box.",
    "Other Christian":
        "-> christianity, bare. What the pool leaves once Lutheran and Anglican are taken out: "
        "`Christian only`, Evangelical, Methodist, Jehovah's Witness, Baptist, Church of Christ, "
        "Dutch Reformed, Orthodox, Zionist and nine smaller answers, and in rounds 5-6 the Lutherans "
        "and Anglicans still answering `Christian only`. 18.4% as drawn. The Oruuano (the Herero "
        "Protestant Unity Church) and the African Methodist Episcopal Church have no box on the card "
        "and are presumably here or in `Other`.",
    "None":
        "-> unaffiliated. Step 2 of the draft \"no religion\" procedure: every round's card offers "
        "`Traditional/ethnic religion` beside `None` and `Atheist`. 2.7% as drawn. Placed with "
        "traditional religion as one box (split-half +0.555) and split at the six rounds' pooled "
        "ratio, 68.1% none, because the two trade places: rounds 4-6 put none at 33.5% of the pair "
        "and rounds 7-9 at 94.5%, with Kunene going from 43% traditional to 16-18% none. "
        "Madagascar used its late rounds' ratio because two DHS surveys witnessed it; Namibia's "
        "DHS has no traditional box (no religion 1.3%), so nothing witnesses either end, and the "
        "pool weights every round alike as the other levels here do.",
    "Traditional/ethnic religion":
        "-> indigenous.african, as `tz2022.py` and `mg2018.py`. 1.25% as drawn: 31.9% of the "
        "none-or-traditional pair in every region (see `None`). Kunene 8.0%, where Himba "
        "communities keep the ancestral fire; Omaheke 2.7%.",
    "Muslim":
        "-> islam, with no branch. Five answers in six rounds (`Muslim only`, `Sunni only`); fails "
        "the split-half and is drawn at its national share, 0.06%, in every region.",
    "Other":
        "-> other.na. 1.5% as drawn, at the national share in every region: it passes the split-half "
        "(+0.446) but 105 of its 108 answers are in rounds 7-9, so a placed share would measure "
        "which rounds are in (Cameroon's `Other`). `Bahai` (1 answer) is folded in.",
}

MAP = {
    "Lutheran": "christianity.lutheran",
    "Roman Catholic": "christianity.catholic",
    "Anglican": "christianity.anglican",
    "Seventh Day Adventist": "christianity.adventist.sda",
    "Pentecostal": "christianity.pentecostal",
    "Other Christian": "christianity",
    "None": "unaffiliated",
    "Traditional/ethnic religion": "indigenous.african",
    "Muslim": "islam",
    "Other": "other.na",
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
