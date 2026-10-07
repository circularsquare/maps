"""Djibouti RGPH-3 2024 religion (INSTAD, Tome 4: Caractéristiques socioculturelles de la
population, Tableau n°42) -> religiondots taxonomy.

Four cells on the 6 régions, 1,003,800 residents of ordinary and nomadic households, from 35,648
(Obock) to 728,010 (Djibouti-Ville). sources/dj.md is the write-up. The question (P12) had eight
codes (musulmane, catholique, protestante, orthodoxe, animiste, athée, sans religion, autres
religions); the tables keep four, and which code went where is not printed. There are no missing
values (Tome 4 Tableau 2), so nothing is excluded.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Islam":
        "-> islam, no branch. 998,273 people, 99.45%. One Muslim code; the volume calls Djibouti's "
        "Islam Sunni of the Shafi'i school (p.124), but that is the report's description and not "
        "an answer anyone gave, so no branch is drawn, as for Somalia and Mauritania. Obock is "
        "99.94% Muslim and Djibouti-Ville the lowest at 99.33%.",
    "Christianisme":
        "-> christianity, the bare branch. 4,455 people, 0.44%. The form had Catholic, Protestant "
        "and Orthodox codes, and the tables merge them. The report says the Christians are mainly "
        "Catholic and Orthodox, especially the Ethiopian Orthodox Church (p.124), with no figure, "
        "so no split is drawn. Djibouti-Ville is 0.53% Christian (3,860) and Ali-Sabieh 0.48% "
        "(350); every other région is 0.06-0.16%. By nationality (Tableau 43): Djiboutians 1,934, "
        "Ethiopians 1,807, other foreigners 475, Eritreans 132, stateless 64.",
    "Sans religion":
        "-> unaffiliated. 868 people, 0.09%, 841 of them in Djibouti-Ville. The form offered "
        "animist as its own code, so this is the `separate` case (spec §6.3a-ii), not a box that "
        "also took traditional religion. Atheist was a code of its own and is presumably merged "
        "in here; nothing goes to `secular`, because the tables do not separate it.",
    "Autre religion":
        "-> other.dj, a NEW per-country residual node, following other.ne and other.gm. 204 "
        "people, 0.02%. The report (p.124) says it covers animism and some Asian religions and "
        "regrets that the write-ins were not detailed (its footnote 5). Animism is not drawn on "
        "`indigenous.african` because the cell is not only animists and nothing says how many "
        "are. 182 are in Djibouti-Ville; 92 are foreigners from outside the four neighbouring "
        "nationalities.",
}

MAP = {
    "Islam": "islam",
    "Christianisme": "christianity",
    "Sans religion": "unaffiliated",
    "Autre religion": "other.dj",
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
