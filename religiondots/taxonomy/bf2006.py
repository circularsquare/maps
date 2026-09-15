"""Burkina Faso RGPH 2006 religion (INSD, Thème 2: État et structure de la population, Tableau
A5.6) -> religiondots taxonomy.

Six answers at province, 14,017,262 residents, 45 units of about 310,000 each. sources/bf.md is
the write-up. The six are the household form's own six codes (P15: 1 Animiste, 2 Musulman,
3 Catholique, 4 Protestant, 5 Autre, 6 Sans religion), so there is no subtotal to exclude and
no denomination to lose beyond the Catholic/Protestant split the form already makes. The form
has no code for no answer and the report prints none.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the province's own total of residents, not a category.",
}

REVIEW = {
    "Animiste":
        "-> indigenous.african. 2,150,309 people, 15.34%. Poni 74.8%, Noumbiel 72.5%, "
        "Bougouriba 61.8% and Tapoa 57.2%, against 98% Muslim Oudalan's 1.1%; Tapoa holds the "
        "largest count, 9.1% of the national one. Read it as a floor, per §11b's continental "
        "rule: the box is an alternative to the others, so a Christian or Muslim who also keeps "
        "a traditional practice is counted once. The 2019 census's Tableau I.22 (Volume des "
        "tableaux statistiques, PDF p42) puts it at 9.0% nationally and the Sud-Ouest région at "
        "48.1%, against 64.9% in 2006 (Tableau 5.3).",
    "Musulman":
        "-> islam, no branch. 8,485,149 people, 60.53%. Oudalan 98.0%, Loroum 97.0%, Séno 96.8%, "
        "Soum 96.3%. Mostly Sunni, with Tijani and other orders, but the form asks neither school "
        "nor order, so a branch would be an inference.",
    "Catholique":
        "-> christianity.catholic. 2,664,236 people, 19.01%. Sanguié 44.4%, Ioba 40.3%, "
        "Boulkiemdé 37.6%, Kadiogo (Ouagadougou) 36.2%; Kadiogo holds 23.5% of the national count.",
    "Protestant":
        "-> christianity.protestant, as td2009.py and cg2007.py map the same French box. 585,154 "
        "people, 4.17%. Gnagna 18.7%, Nahouri 12.0%, Tapoa 9.8%. One box for every Protestant "
        "church, Pentecostal and evangelical bodies included; nothing in the census splits them, "
        "so they are not guessed apart.",
    "Sans religion":
        "-> unaffiliated. 52,929 people, 0.38%. Kompienga 3.5%, Sanguié 2.0%, Tapoa 1.9%. The "
        "form offers `Animiste` as its own code beside it (P15, code 1 against code 6), so this "
        "is the `separate` case and not the Mozambique or Laos box that also took traditional "
        "religion (spec §6.3a-ii). It is highest where animism is also high, so some of it is "
        "probably traditional practice answered as none; the census cannot say how much. Nothing "
        "goes to `secular`.",
    "Autre":
        "-> other.bf, a NEW per-country residual node, following other.td and other.cg. 79,485 "
        "people, 0.57%; Kadiogo 1.0% and 21.2% of the count. The report does not say what it "
        "holds, and part of it is probably blank answers: the form has no non-response code, "
        "Tableau 1.3 prints 0.00% religion non-response, and Tableau A5.7 counts 13,919 `Autre` "
        "among the 74,487 people whose age was not recorded (18.7%, against 0.57% overall). "
        "The 2019 census put `Autre` at 0.2%. Kept on a religion node rather than `unknown` "
        "because the code is a religion answer and how much of it is blank is not measured "
        "anywhere; the alternative, `unknown`, would move 79 dots and relabel whatever real "
        "other religions it holds.",
}

MAP = {
    "Animiste": "indigenous.african",
    "Musulman": "islam",
    "Catholique": "christianity.catholic",
    "Protestant": "christianity.protestant",
    "Autre": "other.bf",
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
