"""Statistics Lithuania 2021 census religion classification -> religiondots taxonomy.

Sixteen categories plus the universe total, at municipality. **The deepest census religion
question in Europe for a country this size** — and unusually, almost every category lands on
a node that already existed, because Lithuania names bodies rather than families.

    74.19%  Romos katalikų                    -> christianity.catholic.latin
    13.67%  Nenurodyta                        -> EXCLUDED (not stated)
     6.11%  Nė vienai                         -> unaffiliated
     3.75%  Stačiatikių (ortodoksų)           -> christianity.orthodox.canonical
     0.65%  Sentikių                          -> christianity.orthodox.oldbeliever
     0.56%  Evangelikų liuteronų              -> christianity.lutheran
     0.55%  Kitų                              -> other.lt
     0.20%  Evangelikų reformatų              -> christianity.reformed
     0.11%  Sekmininkų                        -> christianity.pentecostal
     0.08%  Musulmonų sunitų                  -> islam.sunni
     0.04%  Baptistų ir „laisvųjų bažnyčių“   -> christianity.baptist
     0.03%  Judėjų                            -> judaism
     0.03%  Graikų apeigų katalikų (unitų)    -> christianity.catholic.eastern
     0.03%  Septintos dienos adventistų       -> christianity.adventist
     0.01%  Naujosios apaštalų Bažnyčios      -> christianity.other
     0.01%  Karaimų                           -> judaism.karaite   (a NEW node)

**Lithuania is the only source on this map that separates Roman from Greek Catholics AND
Orthodox from Old Believers AND names the Karaims** — three splits that every other census
either collapses or never asks. That is what makes 255 people worth a taxonomy node.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Iš viso pagal religiją":
        "the unit's own population total, not a category.",
    "Nenurodyta":
        "384,094 people, 13.67%, and it is not irreligion — `Nė vienai` (belongs to no "
        "religion) is a separate answer taken by 171,810. Lithuania's religion question is "
        "voluntary, so this is a refusal plus item non-response (spec §3.5). It has "
        "**risen at every census** — 5.35% in 2001, 10.11% in 2011, 13.67% in 2021 — while "
        "the explicit no-religion answer barely moved (9.51%, 6.13%, 6.11%). Almost all of "
        "the fall in Roman Catholic identification since 2001 has gone here rather than to "
        "`no religion`, which is a fact about the question as much as about the country.",
}

REVIEW = {
    "Karaimų":
        "-> judaism.karaite, A NODE ADDED FOR THIS SOURCE, and the most arguable call in "
        "the file. 255 people: the Karaims of Trakai, Turkic-speaking, brought from Crimea "
        "by Vytautas in 1397, and one of the smallest recognised religious communities in "
        "Europe. Karaite Judaism is by any scholarly account a Jewish movement — it holds "
        "the written Torah and rejects the rabbinic oral law, and it separated in the "
        "eighth or ninth century. **But many Lithuanian Karaims describe themselves as a "
        "distinct people and not as Jews**, and the census keeps `Karaimų` and `Judėjų` "
        "apart, which is the source making the same distinction. Filing them under Judaism "
        "is therefore a claim about the RELIGION and not about the community, and it is the "
        "reason the node is a child rather than a root: a root would assert they are "
        "unrelated, and `judaism` alone would assert they are the same answer as `Judėjų`. "
        "The middle option says what the source says. Overturnable, and this is where.",
    "Romos katalikų":
        "-> christianity.catholic.latin and NOT the parent `christianity.catholic`, which "
        "is where most sources' bare `Catholic` goes. Lithuania publishes `Graikų apeigų "
        "katalikų (unitų)` as its own answer, so `Romos katalikų` means the Latin rite "
        "specifically rather than Catholics in general. When a source names both rites, "
        "file both at the rite.",
    "Stačiatikių (ortodoksų)":
        "-> christianity.orthodox.canonical, hr2021.py's and rs2022.py's call. Lithuania's "
        "Orthodox are the Vilnius and Lithuania diocese of the Moscow Patriarchate, "
        "canonical throughout; the Ecumenical Patriarchate's Lithuanian exarchate was "
        "created in 2024, after this census. Old Believers are counted separately below, so "
        "this category does not carry them.",
    "Sentikių":
        "-> christianity.orthodox.oldbeliever. 18,196 people, and **this is the largest Old "
        "Believer population any source on this map counts directly** — Russia's comes from "
        "a 56,900-person survey (ru2012.py) and nobody else asks at all. Lithuania's are "
        "the Pomorian (bespopovtsy) communities of the north-east, refugees from the Nikonian "
        "reforms of the 1650s, and their number has fallen by a third since 2001 (27,073).",
    "Baptistų ir „laisvųjų bažnyčių “":
        "-> christianity.baptist, and the alternative was christianity.protestant. The cell "
        "is a COMPOUND — Baptists *and* free churches — so neither node is exactly right: "
        "`christianity.baptist` overstates what is known about part of the 1,092, and "
        "`christianity.protestant` throws away the body the source names first. §12 says "
        "map to what the source names and record the disagreement, so the named body wins. "
        "The trailing space in the label is Statistics Lithuania's own and is kept verbatim "
        "(§2.4) — the key must match what sources/lt.py writes.",
    "Naujosios apaštalų Bažnyčios":
        "-> christianity.other, which is au2021.py's call for the same body. The New "
        "Apostolic Church is a 19th-century offshoot of the Catholic Apostolic (Irvingite) "
        "movement and belongs to no Protestant family the tree holds. 412 people.",
    "Musulmonų sunitų":
        "-> islam.sunni. The Lipka Tatars, settled around Alytus and Vilnius since the "
        "fourteenth century, and one of the oldest continuous Muslim communities in Europe. "
        "2,165 people. The census names the branch, so this does not go to the `islam` "
        "parent — Russia is the only other source that asks (ru2012.py).",
    "Kitų":
        "-> other.lt. 15,353 people, 0.55%, which is large for a residual on a form with "
        "sixteen answers. Romuva — the Baltic-faith revival, finally state-recognised in "
        "2025 — is inside it and has no cell of its own, which is the one thing this census "
        "does not count that it plausibly could. Per source, per spec §3.11.",
}

MAP = {
    "Romos katalikų": "christianity.catholic.latin",
    "Graikų apeigų katalikų (unitų)": "christianity.catholic.eastern",
    "Stačiatikių (ortodoksų)": "christianity.orthodox.canonical",
    "Sentikių": "christianity.orthodox.oldbeliever",
    "Evangelikų liuteronų": "christianity.lutheran",
    "Evangelikų reformatų": "christianity.reformed",
    "Baptistų ir „laisvųjų bažnyčių “": "christianity.baptist",
    "Sekmininkų": "christianity.pentecostal",
    "Septintos dienos adventistų": "christianity.adventist",
    "Naujosios apaštalų Bažnyčios": "christianity.other",
    "Musulmonų sunitų": "islam.sunni",
    "Judėjų": "judaism",
    "Karaimų": "judaism.karaite",
    "Kitų": "other.lt",
    "Nė vienai": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


# The source's own labels carry a trailing space inside the quotes on one category and a
# non-breaking space would be indistinguishable by eye, so both EXCLUDED and MAP are keyed
# through _key and the tables above are rewritten to match rather than trusted to.
EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
