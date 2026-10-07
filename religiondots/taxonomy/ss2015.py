"""South Sudan, World Bank and NBS High Frequency South Sudan Survey, wave 1 (2015), religion of the
household head -> religiondots taxonomy.

Module C item C.9, a multi-select on one card. Six of the ten former states (the Equatorias, Lakes,
Northern and Western Bahr el Ghazal); Jonglei, Unity, Upper Nile and Warrap are not drawn.
sources/ss.py builds the table and sources/ss.md is the record.

As drawn over the six states (people in households whose head gave the answer; a head who ticked
two boxes is split between them):

    88.77%  Christianity                  -> christianity  (the family root)
     6.74%  Traditional African Religion  -> indigenous.african
     2.48%  Not recorded                  -> EXCLUDED
     1.62%  Islam                         -> islam
     0.33%  Atheism                       -> secular
     0.03%  Buddhism                      -> buddhism
     0.01%  Judaism                       -> judaism   (national share in every state)
     0.00%  Agnostic                      -> secular   (national share in every state)

Hinduism and Other are on the card and no wave-1 head gave either as an answer.
"""

EXCLUDED = {
    "Not recorded":
        "58 of 3,550 wave-1 heads with no answer, 47 of them in rural Eastern Equatoria; 2.48% of "
        "the six states as weighted, 12.8% of Eastern Equatoria. It passes the split-half, so it "
        "is at each state's own share. Not drawn; in `gap` in countries/ss.py.",
}

REVIEW = {
    "Christianity":
        "-> christianity, the family root, drawn as Christianity unspecified. The card offers one "
        "Christian box. IRI's 2013 national poll split it (Catholic 27%, Protestant 13%, Episcopal "
        "Church of Sudan 9%, unspecified Christian 38%), nationally only, so no branch is drawn. "
        "Western Equatoria 99.6%, Central Equatoria 98.6%; Northern Bahr el Ghazal and Eastern "
        "Equatoria 75%.",
    "Traditional African Religion":
        "-> indigenous.african. 253 head-answers, 6.74% as drawn: Northern Bahr el Ghazal 23.4% "
        "(24.8% of its rural sample), Eastern Equatoria 10.6%, Lakes 6.1%, none sampled in either "
        "of the western Equatorias or Central Equatoria. A box offered beside Christianity is a "
        "floor for practice (sources.md §11b). It agrees with IRI 2013's 7% nationally and is a "
        "fifth of Pew 2020's 32.8% `other religions`, which no survey that asks reproduces.",
    "Islam":
        "-> islam, no branch, because the card gives none. 140 head-answers: Western Bahr el Ghazal "
        "10.5% (19.7% of its towns, Wau and Raja), elsewhere 0 to 1.3%. Wave 2's towns put Western "
        "Bahr el Ghazal at 14.0%.",
    "Atheism":
        "-> secular, as every other card's Atheist box. 8 heads, all rural, in Eastern Equatoria and "
        "Western Bahr el Ghazal; passes the split-half (p 0.005) and the vetoes, so drawn where "
        "found. In a rural survey with a traditional box beside it this may hold people with no "
        "church rather than declared atheists; nothing in the file says which.",
    "Buddhism":
        "-> buddhism. 3 heads, all in Central Equatoria (Juba's foreign residents, most likely). "
        "Passes the split-half at p 0.027 and both vetoes, so it is drawn in Central Equatoria "
        "only, at 0.13% of the state. Kept as measured rather than folded into another node.",
    "Judaism":
        "-> judaism. 2 heads (one beside Christianity). Fails the split-half; drawn at its "
        "national 0.01% in every state, about 800 people over the six. Kept as the card's answer, "
        "not reassigned.",
    "Agnostic":
        "-> secular. One head, as a second answer beside Christianity; national share everywhere.",
}

MAP = {
    "Christianity": "christianity",
    "Traditional African Religion": "indigenous.african",
    "Islam": "islam",
    "Atheism": "secular",
    "Agnostic": "secular",
    "Buddhism": "buddhism",
    "Judaism": "judaism",
}

# spec §7a-i-1: every row is drawn at the node its column names.
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
