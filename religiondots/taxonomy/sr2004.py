"""ABS Suriname Census 7 (2004) religion classification -> religiondots taxonomy.

Six categories on 62 ressorten. **The shallowest religion question this map draws in the
Americas, on its finest geography.** Every other Caribbean source here names between ten and
nineteen bodies; this one names three religions, a combined traditional/other cell, and no
religion.

    40.73%  Christianity                 -> christianity   (the PARENT)   REVIEW
    19.93%  Hinduism                     -> hinduism
    15.67%  Don't know/No answer         -> EXCLUDED
    13.45%  Islam                        -> islam
     5.79%  Traditional Religion +Others -> other.sr   (a NEW node)       REVIEW
     4.42%  No religion                  -> unaffiliated

**WHAT THE COUNTRY IS FOR, AND WHY THE THIN QUESTION STILL EARNS IT.** Suriname is
**13.45% Muslim — by far the highest share in the Americas** — and 19.93% Hindu, and it is
counted at 7,900 people per unit, which is fine enough to show that those two are not spread
but sorted. Hinduism reaches **65% in Jarikaba** and 60% in Westelijke Polders; Islam reaches
**48% in Nieuw Amsterdam** and 47% in Lelydorp. With Guyana (§9r) and Trinidad (§9ak) this
completes the Indo-Caribbean geography.

**THE VINTAGE IS 2004 AND THE REASON IS IN `sources/sr.py`**: Census 8 (2012) publishes
religion nationally only, its district presentations reach 3 of 10 districts, and Census 9
has published nothing. This is the only whole-country sub-national religion table Suriname
has.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the ressort's own population total, not a category. Carried in sr.csv because "
        "sources/sr.py checks the six categories against it.",
    "Don't know/No answer":
        "77,204 people, **15.67% — the largest non-answer on this map**, ahead of "
        "Trinidad's 11.10%. ABS pools 'don't know' with 'no answer' in one cell, so there "
        "is no refusal/blank pair to keep apart the way Croatia and Bosnia have. Spec "
        "§3.5: marked, not filled, and never redistributed — which matters more here than "
        "anywhere, because a sixth of the country is behind it and the drawn shares are "
        "therefore shares of everybody rather than of the people who answered. "
        "**Its geography is sharp and it has TWO peaks that are nothing like each other**, "
        "which is worth stating rather than explaining away. By district: **Sipaliwini "
        "22.9%** — the roadless interior, reached by river and air — and **Paramaribo "
        "21.1%**, the capital, against **Wanica 2.6%** in the suburban belt between them. "
        "The single worst unit is `Welgelegen (Par'bo)` at 33%, with Albina and Tapanahony "
        "at 29%. A remote-enumeration explanation covers Sipaliwini and not Paramaribo; a "
        "refusal explanation covers Paramaribo and not Sipaliwini. ABS publishes no "
        "analysis of it and none is invented here.",
}

REVIEW = {
    "Christianity":
        "-> `christianity`, the PARENT node, undivided. This is the single biggest loss in "
        "the country and it is worth being explicit about. 200,744 people, 40.73%, in one "
        "cell — **and Suriname's Christianity is genuinely plural**: the census of 2012 "
        "names Rooms Katholiek, E.B.G. (the Moravian Evangelische Broedergemeente, which "
        "has been in Suriname since 1735 and is one of the oldest Protestant missions in "
        "the Americas), Volle Evangelie and Luthers as separate answers. **All of that is "
        "flattened to one colour here.** "
        "**It is NOT subdivided using the 2012 national split**, which would be the "
        "obvious move and is forbidden: applying a national denominational mix to 62 "
        "ressorten would invent the entire spatial structure of the result while the only "
        "published number is a national one. That is §14 rule 1 and the same refusal §11r "
        "made for Saudi Arabia. A future build on Census 9, or on a 2012 district table if "
        "ABS ever publishes one, fixes this properly.",
    "Traditional Religion +Others":
        "-> other.sr, a NEW per-source residual (§3.11), and **not** "
        "`indigenous.african`, which is where Ethiopia's and Ghana's `Traditional` cells "
        "go. 28,549 people, 5.79%. "
        "The label itself is the reason: it is `Traditional Religion` **+Others**, a named "
        "tradition pooled with a residual, so mapping the whole cell to a traditional-"
        "religion node would assert that everyone in it practises one — which the 2012 "
        "census shows is false. Census 8's equivalent district label reads **`Andere "
        "godsdienst Jodendom Winti Jehova's Getuigen`**: Winti pooled with Judaism and the "
        "Jehovah's Witnesses in a single cell. So this cell certainly contains Winti and "
        "indigenous Amerindian religion, and certainly contains things that are neither. "
        "It cannot be split and is not (§14.4). See branches.py for what is known to be "
        "inside it.",
    "Islam":
        "-> islam, undivided, and the number is the point: **13.45%, the highest Muslim "
        "share of any country this map draws in the Americas** and higher than several it "
        "draws in Europe. Suriname's Muslims are mostly Javanese (from the Dutch East "
        "Indies indenture) and Indo-Surinamese, and 2012 does split `Islam; Soenniet` from "
        "`Overig Islam (Incl. Ahmadyah)` — the Ahmadiyya presence in Suriname is "
        "substantial and is invisible at this vintage. One cell here, so `islam`.",
    "Hinduism":
        "-> hinduism, undivided. 98,240 people. 2012 splits `Hindoe; Sanatan` from "
        "`Overig Hindoe (Incl. Aryah)` — the Sanatan Dharm / Arya Samaj division that "
        "`mu2022.py` was able to draw in Mauritius (§9af) — and 2004 does not, so the "
        "parent is the honest grain.",
}

MAP = {
    "Christianity": "christianity",
    "Hinduism": "hinduism",
    "Islam": "islam",
    "Traditional Religion +Others": "other.sr",
    "No religion": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
