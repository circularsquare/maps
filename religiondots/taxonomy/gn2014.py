"""Guinea RGPH 2014 religion (INS, État et structure de la population, Tableau 5.10) ->
religiondots taxonomy.

Five cells at région administrative, 10,503,134 people in ordinary households, eight units,
about 1.3 million each. sources/gn.md is the write-up. The five are the census form's own
five boxes (P11: 0 sans religion, 1 musulmane, 2 chrétienne, 3 animiste, 4 autre religion),
so there is no subtotal to exclude and no denomination to lose: the form never asked one.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Chrétienne":
        "-> christianity, the bare branch. 711,350 people, 6.77%. The 2014 form has one "
        "Christian box and no follow-up, so this is not a census that split Christianity "
        "and lost the split. **62.5% of these people are in N'Zérékoré**, where the "
        "Catholic, Protestant (the CMA-founded Église Protestante Évangélique) and "
        "Pentecostal churches of the Kpelle, Kissi, Toma and Mano peoples are; the rest is "
        "mostly Conakry (4.8% of the capital) and Faranah (9.7%). NO LONGER DRAWN since "
        "2026-10-16: gn_split.csv replaces it with three national shares (below) and this "
        "cell is what they roll back to (COLUMNS).",
    "Catholique":
        "-> christianity.catholic, DERIVED. 462,376, 65% of Christians in every région. The "
        "Afrobarometer's named Christians, rounds 5-9, are 64.8% Catholic (n=414, 57-75% by "
        "round) and Pew 2010 gives 68%. N'Zérékoré's own named answers sit in the same range, "
        "so the flat ratio costs little where most of the dots are.",
    "Protestante ou évangélique":
        "-> christianity.protestant, DERIVED. 213,406, 30%: mostly the Église Protestante "
        "Évangélique, which the CMA founded; the survey's `Evangelical` box is 24.3% of named "
        "Christians and Pew 2010's Protestant 31%. On `.protestant` with Mali and Burkina Faso "
        "rather than `.evangelical` with Côte d'Ivoire; that node question is Anita's and is "
        "parked (unspecified_splits/westafrica.md §5).",
    "Autre chrétienne":
        "-> christianity.other, DERIVED. 35,568, 5%: Orthodox, Witnesses, New Apostolic and "
        "the rest of the survey's small boxes.",
    "Musulmane":
        "-> islam, no branch. 9,358,718 people, 89.10%. Overwhelmingly Sunni, but the "
        "census asks neither school nor order, so `islam.sunni` would be an inference.",
    "Animiste":
        "-> indigenous.african, the node Ghana added. 167,406 people, 1.59%, and **98.8% of "
        "them in one région**: N'Zérékoré is 10.4% animist and no other région prints more "
        "than 0.1%. **Read it as a floor**, per §11b's continental rule: the box is an "
        "alternative to Musulmane and Chrétienne, so a Kissi or Toma Christian who also "
        "keeps the forest initiation societies is counted once, as Christian.",
    "Sans religion":
        "-> unaffiliated. 252,787 people, 2.41%, and **88.1% of them in N'Zérékoré**, where "
        "it is 14.2% against 0.1% to 0.8% everywhere else. A cell that large in the one "
        "région where traditional religion is also large is not secularisation, and some "
        "part of it is very probably traditional practice answered as no religion; the "
        "census cannot say how much, so it is drawn as printed (§14.4). Nothing goes to "
        "`secular`, which needs a separately counted atheist answer (ke2019.py, lr2022.py).",
    "Autres religions":
        "-> other.gn. 12,873 people, 0.12%. The report does not name what it holds. 72.8% of "
        "it is in N'Zérékoré.",
}

MAP = {
    "Sans religion": "unaffiliated",
    "Musulmane": "islam",
    "Chrétienne": "christianity",
    "Animiste": "indigenous.african",
    "Autres religions": "other.gn",
    # 2026-10-16, sources/wa_christian.py: gn_split.csv replaces `Chrétienne`. `derived`.
    "Catholique": "christianity.catholic",
    "Protestante ou évangélique": "christianity.protestant",
    "Autre chrétienne": "christianity.other",
}

# spec §7a-i-1: measured rows at their own node; the three derived Christian rows roll back
# to the `Chrétienne` cell the census counted per région.
COLUMNS = {v: v for v in MAP.values()}
COLUMNS["Chrétienne"] = "christianity"


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
