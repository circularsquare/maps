"""Mali RGPH5 2022 religion (INSTAT, Tableau 2.03 of *Caractéristiques culturelles de la
population* for five groups, Tableau 6.13 of *État et structure de la population* for the
Christian split) -> religiondots taxonomy.

Seven answers at région, 21,347,586 residents of ordinary households, 20 units of about
1.07 million each. sources/ml.md is the write-up. The seven are the census's own codes (P10 of
the household form prints 1 Musulman, 2 Catholique, 3 Protestant, 4 Autre religion chrétienne,
5 Animiste, 6 Sans religion; Tableau 1.01 of the cultural volume adds 7 Autre religion). There is
no code for no answer; annex A01 counts 48,746 `Non Déclaré` (0.23%) nationally, and every
published share spreads them over the seven answers in proportion, so the counts here do too.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the région's residents of ordinary households (Tableau 2.9), not a category.",
}

REVIEW = {
    "Musulman":
        "-> islam, no branch. 20,541,904 in annex A01, 96.45% with non-response spread in "
        "(Tableau 2.01). Above 98% in 14 of the 20 régions; lowest in San 70.10%, Koutiala "
        "91.57% and Bandiagara 93.96% (Tableau 2.03). Overwhelmingly Sunni (Maliki) with Tijani "
        "and Qadiri orders and Wahhabi-leaning communities, but the form asks none of it.",
    "Catholique":
        "-> christianity.catholic. 290,952 in A01 (1.36%). San 9.6%, Bandiagara 4.1%, Koutiala "
        "1.8%, Bamako 1.5% (Tableau 6.13, one decimal). Those one-decimal cells, weighted by "
        "population, come to 1.5% more Catholics than A01 counts, so the rake in sources/ml.py "
        "takes every région's Catholics down by that factor.",
    "Protestant":
        "-> christianity.protestant, as bf2006.py and td2009.py map the same French box. 175,691 "
        "in A01 (0.82%). San 7.2%, Koutiala 2.4%, Bandiagara 1.9%. One code for every Protestant, "
        "evangelical and Pentecostal church; nothing in the census splits them.",
    "Autre religion chrétienne":
        "-> christianity.other, following bj2013.py and ci2021.py for the same French box, though "
        "the node's description is for bodies with no branch rather than a residual, and nothing "
        "says what Mali's box holds (Orthodox, Jehovah's Witnesses and Latter-day Saints are the "
        "usual guesses; none is named). 17,107 in A01 (0.08%). Tableau 6.13 prints it at 0.1 or "
        "0.2 in nine régions (San and Gao 0.2) and 0.0 in eleven, so the région pattern is mostly "
        "rounding, and raking the nine printed cells to A01's count multiplies them by about "
        "1.26. It is 17 dots.",
    "Animiste":
        "-> indigenous.african. 138,800 in A01 (0.65%). San 8.33% and Koutiala 3.14%, then "
        "Sikasso 0.93% and Koulikoro 0.74%; no other région reaches 0.2% (Tableau 2.03). The "
        "box is an alternative to the others, so a Muslim or Christian who also keeps a "
        "traditional practice is counted once: read it as a floor (§11b's continental rule). "
        "The 2009 census printed 2.0% nationally (Tableau 6.14 of the structure volume). "
        "Bandiagara, the Dogon plateau, is 0.04%.",
    "Sans religion":
        "-> unaffiliated. 105,984 in A01 (0.50%). San 3.65%, Koulikoro 1.44%, Sikasso 1.04% "
        "(Tableau 2.03). The form offers `Animiste` as its own code beside it (code 5 against "
        "code 6), so this is the `separate` case, not the Mozambique or Laos box that also took "
        "traditional religion (spec §6.3a-ii). It is highest where animism is also high, so some "
        "of it is probably traditional practice answered as none; the census cannot say how "
        "much. Nothing goes to `secular`.",
    "Autre religion":
        "-> other.ml, a NEW per-country residual node, following other.bf and other.td. 28,403 in "
        "A01 (0.13%). San 0.93%, Koulikoro 0.36%, Koutiala 0.35% (Tableau 2.03). The report does "
        "not say what it holds. The paper household form prints only six codes, and Tableau 1.01 "
        "of the cultural volume lists this as the seventh, so it is probably the tablet form's "
        "extra code; how the paper forms' other answers were coded is not stated.",
}

MAP = {
    "Musulman": "islam",
    "Catholique": "christianity.catholic",
    "Protestant": "christianity.protestant",
    "Autre religion chrétienne": "christianity.other",
    "Animiste": "indigenous.african",
    "Sans religion": "unaffiliated",
    "Autre religion": "other.ml",
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
