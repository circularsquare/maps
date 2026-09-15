"""Chad RGPH2 2009 religion (INSEED, État et structures de la population, Tableaux 5.06 and
5.07) -> religiondots taxonomy.

Six cells at région, 10,941,682 censused people, 22 units, about 500,000 each. sources/td.md
is the write-up. The six are the census form's own six boxes (B12: ANI 1 animiste, CAT 2
catholique, MUS 3 musulman, PRO 4 protestant, AUT 5 autres, SAN 6 sans), so there is no
subtotal to exclude and no denomination to lose beyond the Catholic/Protestant split the form
already makes.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Musulmane":
        "-> islam, no branch. 6,392,040 people, 58.42%. Overwhelmingly Sunni and largely "
        "Tijani, but the form asks neither school nor order, so `islam.sunni` would be an "
        "inference. 91% of them live outside the seven southern régions; N'Djaména is 70.7% "
        "Muslim.",
    "Catholique":
        "-> christianity.catholic. 2,026,152 people, 18.52%, 90.3% of them in the seven "
        "southern régions (Logone Oriental 48.1%, Mandoul 47.9%, Logone Occidental 47.8%).",
    "Protestante":
        "-> christianity.protestant, as cg2007.py and cf2003.py map the same French box. "
        "1,761,516 people, 16.10%. The form has one Protestant box and the report says the "
        "1993-2009 growth came from new revival churches (`églises du réveil (pentecôtistes, "
        "etc.)`, p131), so the box holds the Pentecostal churches together with the evangelical "
        "bodies of the Entente des Églises et Missions Évangéliques au Tchad. Not guessed "
        "apart.",
    "Animiste":
        "-> indigenous.african. 438,831 people, 4.01%, and **94.4% of them in the south**: "
        "Mayo Kebbi Est 32.0%, Mayo Kebbi Ouest 12.5%, Moyen Chari 6.4%, against 0.1% to 1.3% "
        "in every northern région. Read it as a floor, per §11b's continental rule: the box is "
        "an alternative to the others, so a Christian who also keeps a traditional practice is "
        "counted once, as Christian. The share fell from 7.5% in 1993 (Tableau 5.08).",
    "Sans religion":
        "-> unaffiliated. 266,486 people, 2.44%, 94.0% in the south, and **15.2% of Mayo Kebbi "
        "Ouest**, 7.4% of Mandoul and 7.1% of Moyen Chari. The form had a separate animist box "
        "(ANI), so this is not the Mozambique or Laos case (spec §6.3a-ii), where the only "
        "no-religion answer also took traditional religion. A cell that large in the régions "
        "where animism is also large is still probably partly traditional practice answered "
        "as none; the census cannot say how much, so it is drawn as printed, as gn2014.py does "
        "for N'Zérékoré. Nothing goes to `secular`.",
    "Autres religions":
        "-> other.td, a NEW per-country residual node, following other.gn and other.cg. 56,657 "
        "people, 0.52%. The report does not say what it holds. 91.4% of it is in the south, "
        "3.6% of Mandoul and 1.4% of Mayo Kebbi Ouest.",
}

MAP = {
    "Animiste": "indigenous.african",
    "Catholique": "christianity.catholic",
    "Musulmane": "islam",
    "Protestante": "christianity.protestant",
    "Autres religions": "other.td",
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
