"""Burundi RGPH 2008 religion (Tableau 1.13), on an Afrobarometer pattern -> religiondots taxonomy.

Nine census rows plus one excluded column at 17 provinces. Every row is the census's national count
by urban and rural; `sources/bi.py` fits it to the 2008 province populations with the pattern of
Afrobarometer rounds 5 and 6, and `sources/bi.md` is the record. Shares are of the 7,964,078 people
in ordinary households, the religion table's universe.

    62.05%  Catholique          -> christianity.catholic
    21.62%  Protestante         -> christianity.protestant
     6.17%  Aucune religion     -> unaffiliated
     3.28%  Autre religion      -> other.bi
     2.52%  Musulmane           -> islam
     2.33%  Adventiste          -> christianity.adventist
     1.68%  ND                  -> EXCLUDED
     0.32%  Témoin de Jéhovah   -> christianity.witnesses
     0.03%  Traditionnelle      -> indigenous.african

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "ND":
        "133,956 people, 1.68% of ordinary households: `non déclaré`, the census's own non-response "
        "row (spec §3.5), 2.8% of urban and 1.6% of rural Burundi.",
    "Ménages collectifs":
        "89,496 people in collective households (Tableau 1.2: barracks, boarding schools, prisons, "
        "convents, camps), outside the universe of the religion table. Built by `sources/bi.py` at "
        "each province's exact Tableau 1.4 count, so every province totals its census population; "
        "not a religion anybody gave.",
}

REVIEW = {
    "Catholique":
        "-> christianity.catholic, the parent rather than .latin, as rw2022.py. 4,941,833, 62.05%. "
        "Placed by the survey's `Roman Catholic` (+0.639 R5 against R6, null +0.424): Gitega and "
        "Muramvya about 81%, Bururi and Makamba 37-39%.",
    "Protestante":
        "-> christianity.protestant, which holds an answer and not a church, as rw2022.py. 1,722,039, "
        "21.62%. The census lifts Adventists and Witnesses out of it. Placed by the survey's "
        "Pentecostal, Anglican, Methodist, Baptist, Evangelical, Lutheran, Presbyterian, Church of "
        "Christ and Dutch Reformed answers together (+0.659), which read 1.34x the census nationally: "
        "Bururi and Makamba 47-49%, Gitega 6.7%. Pentecostals are 438 of those 701 answers; no "
        "church is drawn on its own, the census naming none.",
    "Musulmane":
        "-> islam, no branch (`Sunni only` 12 of 93 Muslim answers). 200,509, 2.52%, and 54.7% of "
        "them urban. Placed by the survey (+0.467, p = 0.036) with the census's urban split as a "
        "third margin (`sources/bi.py` docstring): Bujumbura Mairie 14.8% of those with a stated "
        "religion. The survey found no Muslim in Muramvya, Mwaro or Rutana (80, 80 and 96 "
        "respondents; about 2 expected in each at the census rate), so they are drawn at zero.",
    "Adventiste":
        "-> christianity.adventist, the parent, as rw2022.py. 185,361, 2.33%. Fails the split-half "
        "(+0.356 against +0.415): the survey's Adventists are 15.7% of Cibitoke's respondents and "
        "under 5% everywhere else, which does not repeat between rounds. Drawn at the census's urban "
        "and rural shares in every province.",
    "Témoin de Jéhovah":
        "-> christianity.witnesses. 25,454, 0.32%. Passes the split-half (+0.472) on 18 answers, "
        "under the 1% floor for placing a survey category; census urban and rural shares.",
    "Aucune religion":
        "-> unaffiliated, step 2 of the no-religion procedure: the 2008 form (P14) offers "
        "`Traditionnelle` separately. 491,098, 6.17%, 94.2% rural and 55% men. The Afrobarometer's "
        "None and Atheist took 0.85% of adults in 2012-2014, a seventh of the census share; the two "
        "count different things (the census counts children by the household's answer, the survey "
        "asks adults what they call themselves) and nothing reconciles them. Not placed by the "
        "survey (the rank test passes, the chi-square does not); census urban and rural shares.",
    "Autre religion":
        "-> other.bi, a node for Burundi like `other.rw` and `other.tg`. 261,081, 3.28%, 94.3% "
        "rural. The census's other rows name Catholics, Protestants, Muslims, Adventists, Witnesses, "
        "traditional religion and no religion, so this holds whatever else Burundians answered, "
        "possibly independent and revival churches that do not call themselves Protestant; ISTEEBU "
        "publishes no list. The survey's Other, Orthodox, Coptic and Independent answers (25) fail "
        "the split-half and are urban where this row is rural, so they are not its pattern.",
    "Traditionnelle":
        "-> indigenous.african. 2,747 people, 0.03%: read it as the count of people who named "
        "nothing else, as rw2022.py reads Rwanda's 0.02%. Kiranga's cult and ancestor practice sit "
        "alongside church membership. No survey box; census urban and rural shares.",
}

MAP = {
    "Catholique": "christianity.catholic",
    "Protestante": "christianity.protestant",
    "Musulmane": "islam",
    "Adventiste": "christianity.adventist",
    "Témoin de Jéhovah": "christianity.witnesses",
    "Aucune religion": "unaffiliated",
    "Autre religion": "other.bi",
    "Traditionnelle": "indigenous.african",
}

# spec §7a-i-1: every row is counted at the node it is drawn on; nothing is inferred downward.
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
