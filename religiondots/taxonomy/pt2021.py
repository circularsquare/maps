"""INE Censos 2021 religion classification -> religiondots taxonomy.

Eleven categories plus the universe total, at freguesia. A short list on a very fine
geography — §3.9's trade taken to the geography end, the way Sri Lanka took it (§9j).

    80.20%  Católica                  -> christianity.catholic
    14.09%  Sem religião              -> unaffiliated
     2.13%  Protestante/Evangélica    -> christianity.protestant
     1.04%  Outra cristã              -> christianity.other
     0.72%  Testemunhas do Jeová      -> christianity.witnesses
     0.69%  Ortodoxa                  -> christianity.orthodox.canonical
     0.42%  Muçulmana                 -> islam
     0.28%  Outra não cristã          -> other.pt   (a NEW node)
     0.22%  Hindu                     -> hinduism
     0.19%  Budista                   -> buddhism
     0.03%  Judaica                   -> judaism

Percentages are of the 8,781,900 people who ANSWERED, not of the 15+ population — INE
removed the 229,978 who declined instead of publishing them (sources/pt.py). Nothing in
this file adjusts for that; the qualifier belongs in `note_public`, not in the mapping.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own universe total, not a category. It is the number who answered, "
        "and the 11 categories sum to it exactly on all 3,439 units.",
}

REVIEW = {
    "Católica":
        "-> christianity.catholic, the parent and not `christianity.catholic.latin`. INE "
        "publishes one Catholic cell and names no rite, so lt2021.py's rule applies in "
        "reverse: file at the rite only when the source names both. 7,043,016 people, "
        "80.2% of the answers, and **the largest Catholic share of any country on this map "
        "that asks the question directly** — higher than Poland's, Croatia's or Ireland's.",
    "Ortodoxa":
        "-> christianity.orthodox.canonical, which is rs2022.py's and hr2021.py's call for "
        "a bare `Orthodox` cell. Portugal's 60,381 Orthodox are an immigration of the "
        "2000s — Ukrainian, Romanian and Moldovan — served by parishes of the Ecumenical "
        "Patriarchate's Metropolis of Spain and Portugal and of the Moscow Patriarchate, "
        "both canonical. The small Igreja Ortodoxa Portuguesa, which is not, is inside "
        "this cell and cannot be separated from it. **Their geography is the Algarve and "
        "not Lisbon** — 3.2% of the Algarve's answers against 1.0% of Grande Lisboa's, "
        "peaking at 8.5% in Almancil — which is the tourism labour market, not a diaspora "
        "settling near a cathedral.",
    "Muçulmana":
        "-> islam, the parent, with no branch. The census does not split it and Portugal "
        "is the country where that costs most: its Muslim population is **substantially "
        "Nizari Ismaili**, descendants of the Mozambican community who arrived after 1975, "
        "and the Ismaili Imamat's seat has been in Lisbon since 2018 — alongside Sunni "
        "communities from Guinea-Bissau, Bangladesh, Nepal and Morocco. One cell of 36,480 "
        "carries all of it. Nothing in the table lets that be split, and splitting it from "
        "outside would be inventing a magnitude (§14.4).",
    "Testemunhas do Jeová":
        "-> christianity.witnesses. Worth noting that Portugal gives them a cell at all: "
        "63,609 people, more than the Orthodox, and most censuses on this map fold them "
        "into a residual. Their geography is the flattest of any category here — no "
        "freguesia exceeds 4.5% and the peaks are scattered rural, which is what a body "
        "that grows by door-to-door work rather than by migration looks like.",
    "Outra cristã":
        "-> christianity.other. 90,948 people. With Catholic, Orthodox, Protestant/"
        "Evangelical and Jehovah's Witnesses all named above it, this cell is Portugal's "
        "Latter-day Saints and Adventists plus the tail; none is separable. Its peaks are "
        "the western Algarve (Lagos 2.6%, Vila do Bispo 2.2%), which is where the "
        "no-religion share also peaks, so it is reading as part of the same foreign-"
        "resident population rather than as a Portuguese denominational geography.",
    "Protestante/Evangélica":
        "-> christianity.protestant. One cell for the whole Reformation and everything "
        "after it, which is the coarsest thing about this source: Lusitanian Anglicans, "
        "the historic Presbyterian and Methodist churches, and a large Pentecostal "
        "population all land together. **Its sharpest geography is a handful of Alentejo "
        "border villages** — Póvoa de São Miguel 12.9%, Sobral da Adiça 9.7% — an order of "
        "magnitude above the national 2.13% and nowhere near the cities where Brazilian "
        "and African Pentecostal churches are. The census cannot say what that is and "
        "neither can this file; it is recorded because a category with two unrelated "
        "geographies is one the map will make someone ask about.",
    "Outra não cristã":
        "-> other.pt, a per-source residual (§3.11). See branches.py for what is in it and "
        "why its geography is the interesting part.",
    "Sem religião":
        "-> unaffiliated and NOT `secular`. INE offers one no-religion answer and does not "
        "distinguish atheist, agnostic or simply unaffiliated, so the tree's coarser node "
        "is the honest one — rs2022.py can split them because Serbia asks separately.",
}

MAP = {
    "Católica": "christianity.catholic",
    "Ortodoxa": "christianity.orthodox.canonical",
    "Protestante/Evangélica": "christianity.protestant",
    "Testemunhas do Jeová": "christianity.witnesses",
    "Outra cristã": "christianity.other",
    "Budista": "buddhism",
    "Hindu": "hinduism",
    "Judaica": "judaism",
    "Muçulmana": "islam",
    "Outra não cristã": "other.pt",
    "Sem religião": "unaffiliated",
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
