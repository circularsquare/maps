"""Republic of the Congo RGPH 2007 religion (CNSEE, Le RGPH-2007 en quelques chiffres,
Tableau 11) -> religiondots taxonomy.

Nine cells at département, 3,697,490 residents (the whole census population), twelve units,
308,124 people each. sources/cg.md is the write-up. The nine are the census form's own nine
codes (P13: CA 1, PR 2, SA 3, KI 4, MU 5, ER 6, AN 7, AU 8, SR 9), in the table's column order,
so there is no subtotal to exclude.

    33.05%  Catholique          -> christianity.catholic
    22.29%  Eglises de réveil   -> christianity.pentecostal
    19.85%  Protestante         -> christianity.protestant
    11.35%  Sans religion       -> unaffiliated
     7.44%  Autres              -> other.cg  (a NEW node, the per-country residual)
     2.20%  Salutiste           -> christianity.holiness.salvation-army
     1.62%  Musulmane           -> islam
     1.46%  Kimbanguiste        -> christianity.africaninstituted.kimbanguist
     0.73%  Animiste            -> indigenous.african

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Eglises de réveil":
        "-> christianity.pentecostal, the bare family node. 824,133 people, 22.29%, the "
        "second answer in the country. *Églises de réveil* is the name used on both banks of "
        "the Congo river for the Pentecostal and charismatic churches that multiplied from "
        "the 1970s, as distinct from the mission Protestant churches, which the form counts "
        "under `Protestante`. The alternatives were `christianity.evangelical` (Kenya's "
        "answer-node for `Evangelical Churches`, which sits beside `Protestant` in the same "
        "way) and `christianity.pentecostal.charismatic`. The family node was taken because "
        "the cell is one census answer covering both classical Pentecostal bodies and newer "
        "independent ministries, which the form does not separate, and the charismatic child "
        "would claim the second. **Its geography is the north**: Sangha 38.8%, Likouala "
        "38.7%, Cuvette 37.8%, Cuvette-Ouest 33.6%, against 11.6% to 12.7% in Niari, "
        "Lékoumou, Bouenza and Pool, where the mission Protestant and Catholic churches were "
        "established first. Brazzaville holds 42.0% of these people (25.2% of the city).",
    "Protestante":
        "-> christianity.protestant, the answer-node. 733,978 people, 19.85%. The form's "
        "`PR` is the mission Protestant answer and names no body. The largest church it "
        "covers is the Église Évangélique du Congo, which came out of the Swedish Mission "
        "Covenant field in the south; the cell's geography is that field: Lékoumou 37.1%, "
        "Niari 33.0%, Bouenza 30.3%, against 7.1% in Cuvette and 9.7% in Plateaux. Not sent "
        "to `christianity.reformed` or any named family because the census names none and "
        "the cell certainly includes other mission churches too (cf2003.py's reasoning).",
    "Catholique":
        "-> christianity.catholic, the parent, consistent with every African census cell that "
        "says only Catholic (cf2003.py, gh2021.py, gn2014.py). 1,222,190 people, 33.05%, and "
        "the largest answer. Brazzaville 42.8% and Pool 42.3%; Plateaux 9.4%.",
    "Salutiste":
        "-> christianity.holiness.salvation-army. 81,334 people, 2.20%, **the largest "
        "Salvation Army count on this map**. The census gives it its own code, so this is a "
        "reading, not an inference. Lékoumou 7.3%, Pool 4.4%, Niari 3.6%; 0.2% to 0.3% in "
        "the four northern départements. Brazzaville holds 39.0% of them.",
    "Kimbanguiste":
        "-> christianity.africaninstituted.kimbanguist, the node Angola added. 54,153 people, "
        "1.46%. Pool 2.7%, Brazzaville 1.9%; 0.1% to 0.3% in Lékoumou, Cuvette and "
        "Cuvette-Ouest. Brazzaville holds 47.9%. The church's centre, Nkamba, is across the "
        "river in the DRC, and Pool is the Kongo-speaking département facing it.",
    "Musulmane":
        "-> islam, no branch, because the form gives none. 59,871 people, 1.62%. Brazzaville "
        "and Pointe-Noire hold 78.9% of them; Tableau 10 of the same brochure counts 20,430 "
        "Malian, 4,985 Senegalese and 5,544 Beninese residents, so a large part of the cell "
        "is West African traders and their families. The highest shares are Likouala 2.8% "
        "and Sangha 2.6%, the northern river and forest towns.",
    "Animiste":
        "-> indigenous.african. 27,009 people, 0.73%, and **read it as a floor** per §11b's "
        "continental rule: the box is an alternative to the Christian ones, so someone who "
        "keeps ancestral practice alongside a church is counted once, as the church. Kouilou "
        "1.7% and Plateaux 1.6% are the highest; no département passes 2%.",
    "Sans religion":
        "-> unaffiliated. 419,826 people, 11.35%. **Plateaux is 33.9%**, Cuvette 27.9%, "
        "Sangha 25.7% and Cuvette-Ouest 25.3%, against 6.0% in Pool and 6.4% in Brazzaville. "
        "A rural share that high in the départements where the Animiste box stays near 1% is "
        "unlikely to be mostly secularisation, and some part of it is very probably "
        "traditional practice answered as no religion; the census cannot say how much, so it "
        "is drawn as printed (§14.4), as gn2014.py does for N'Zérékoré. Nothing goes to "
        "`secular`, which needs a separately counted atheist answer.",
    "Autres":
        "-> other.cg, a NEW per-country residual node, following other.gn and other.cf. "
        "274,996 people, 7.44%. The brochure does not say what it holds. Its geography is "
        "sharp: **Kouilou 29.3%** (the largest answer there), Plateaux 16.9%, Niari 11.1%, "
        "against 3.5% in Brazzaville. Congo's own prophetic movements have no code on the "
        "form, among them the Matsouanists and the Mission of the prophet Zéphirin Lassy, "
        "and they may well be much of it, but nothing published separates them, so the cell "
        "is not sent to `christianity.africaninstituted` or anywhere else.",
}

MAP = {
    "Catholique": "christianity.catholic",
    "Protestante": "christianity.protestant",
    "Salutiste": "christianity.holiness.salvation-army",
    "Kimbanguiste": "christianity.africaninstituted.kimbanguist",
    "Musulmane": "islam",
    "Eglises de réveil": "christianity.pentecostal",
    "Animiste": "indigenous.african",
    "Autres": "other.cg",
    "Sans religion": "unaffiliated",
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
