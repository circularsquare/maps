"""DR Congo, INS Enquête 1-2-3 2005 and 2012 (U.S. Census Bureau tabulation of household heads)
-> religiondots taxonomy.

Nine columns at province, 31,755 household heads, each province's shares laid on COD-PS 2024.
sources/cd.md is the write-up; sources/cd.py builds the table. The eight answers are the survey's
own codes (2012 household roster, M27 `Religion pratiquée`: 1 Catholique, 2 Protestante,
3 Kimbanguiste, 4 Musulmane, 5 Autre Chrétiens, 6 Animiste, 7 Autre religion, 8 Sans religion),
under the labels USCB's data dictionary gives as each column's original field name.

National shares as drawn (people in households whose head gave the answer):

    35.29%  Catholique        -> christianity.catholic
    27.85%  Protestant        -> christianity.protestant
    22.28%  Autre chrétien    -> christianity                          (the family root)
     4.78%  Sans religion     -> unaffiliated
     4.32%  Autre réligion    -> other.cd  (a NEW node, the per-country residual)
     3.18%  Kimbanguiste      -> christianity.africaninstituted.kimbanguist
     1.69%  Musulman          -> islam
     0.61%  Animiste          -> indigenous.african  (at its national share everywhere)
            Manquant          -> EXCLUDED (20 heads, 0.06%)

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Manquant":
        "20 of 31,755 heads with no religion recorded, 0.06%. Not drawn; `gap` in countries/cd.py.",
}

REVIEW = {
    "Autre chrétien":
        "-> christianity, the family root, drawn as Christianity unspecified. 7,329 heads, 22.28% "
        "as drawn, the third answer in the country. The card offers Catholic, Protestant and "
        "Kimbanguist and nothing else Christian, so this box holds every other church: in the DRC "
        "that is above all the églises de réveil, the Pentecostal and charismatic revival churches "
        "(cg2007.py maps Congo-Brazzaville's explicit `Eglises de réveil` code to "
        "christianity.pentecostal), and also the Salvation Army, Jehovah's Witnesses, Adventists, "
        "Orthodox and the Branhamist Message churches, none of which has a code. Not sent to "
        "christianity.pentecostal because the survey does not say so, and not to "
        "christianity.other, which is for bodies with no branch rather than a residual "
        "(spec §3.2; au2021.py's `Other Christian`, be2024.py and Puerto Rico's `Otros` go to the "
        "root the same way). **Its geography is Kasaï and Kinshasa**: Kasaï-Oriental 55.0%, Kasaï "
        "44.5%, Lomami 43.8%, Sankuru 41.4%, Kasaï-Central 39.4%, Kinshasa 36.9%; Sud-Kivu 3.3%, "
        "Ituri 3.6%, Haut-Uele 4.1%. EDS-RDC III (2023-24) offers `Église de réveil` as its own "
        "code and prints `Église non dénominationelle` at 39.1% of women 15-49 nationally, which "
        "says what most of this box has since become.",
    "Protestant":
        "-> christianity.protestant, the answer-node. 8,757 heads, 27.85%. In the DRC the answer "
        "means the member communities of the Église du Christ au Congo, the single legal Protestant "
        "body since 1970, which is itself a federation of some sixty mission-descended "
        "communities (Baptist, Methodist, Presbyterian, Mennonite, Disciples and others), so no "
        "one family is named. Haut-Lomami 50.6%, Maniema 43.2%, Sud-Ubangi 41.3%, Nord-Kivu 40.4%; "
        "Kasaï-Oriental 11.6%.",
    "Catholique":
        "-> christianity.catholic, the parent, as every African census or survey cell that says "
        "only Catholic (cg2007.py, cf2003.py). 11,114 heads, 35.29%, the largest answer. Ituri "
        "73.7%, Haut-Uele 67.0%, Bas-Uele 50.1%; Kasaï-Oriental 13.3%, Sankuru 13.8%. EDS-RDC III "
        "puts Catholics at 23.7% of women and 25.8% of men 15-49 in 2023-24.",
    "Kimbanguiste":
        "-> christianity.africaninstituted.kimbanguist. 974 heads, 3.18%. Sankuru 12.7% and "
        "Kongo-Central 10.5%, where the church began (Nkamba, 1921); Lualaba 6.5%; 0.2% to 0.3% in "
        "Bas-Uele, Ituri and Nord-Kivu. The Sankuru figure rests on 695 heads, the fewest of any "
        "province but one, and it passes the split-half with the rest of the answer.",
    "Musulman":
        "-> islam, no branch, because the card gives none. 563 heads, 1.69%. **Maniema 16.4%**, "
        "the Swahili-speaking river towns of Kindu and Kasongo that the nineteenth-century "
        "Zanzibari trade settled; Sud-Kivu 4.9%, Kasaï 4.6%; none sampled in Kwango or "
        "Haut-Lomami. The largest single district is Kasongo, 17% of the answer, under the "
        "one-half veto.",
    "Animiste":
        "-> indigenous.african, **at its national share in every province**. 167 heads, 0.61% as "
        "drawn. It fails the split-half on districts (median +0.264 against a null 95th "
        "percentile of +0.278, p 0.059), so the survey is not allowed to say where it is. Read it "
        "as a floor per §11b's continental rule: the box is an alternative to the Christian ones.",
    "Sans religion":
        "-> unaffiliated. 1,470 heads, 4.78%. Maï-Ndombe 11.5%, Tanganyika 9.1%, Kasaï-Oriental "
        "7.9%; Sud-Ubangi 0.6%, Nord-Ubangi 0.7%. As in Congo-Brazzaville, some of a high rural "
        "figure is probably traditional practice answered as no religion; nothing measures how "
        "much, so it is drawn as the heads gave it (§14.4). Nothing goes to `secular`.",
    "Autre réligion":
        "-> other.cd, a NEW per-country residual node, following other.cg. 1,361 heads, 4.32%. "
        "The survey does not say what it holds. Tshopo 13.0%, Tshuapa 12.8%, Sankuru 12.1%, "
        "Kasaï-Oriental 10.7%; Bas-Uele 1.0%. The DRC's own prophetic and neo-traditional "
        "movements (Bundu dia Kongo, Mpeve ya Longo, Kitawala) and the Bahá'ís have no code, and "
        "may be much of it; nothing published separates them, so it is not sent to "
        "christianity.africaninstituted or indigenous.african.",
}

MAP = {
    "Catholique": "christianity.catholic",
    "Protestant": "christianity.protestant",
    "Kimbanguiste": "christianity.africaninstituted.kimbanguist",
    "Musulman": "islam",
    "Autre chrétien": "christianity",
    "Animiste": "indigenous.african",
    "Autre réligion": "other.cd",
    "Sans religion": "unaffiliated",
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
