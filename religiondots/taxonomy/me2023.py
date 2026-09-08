"""MONSTAT Popis 2023 religion classification -> religiondots taxonomy.

Twelve categories plus the universe total and a suppression residual, aggregated from
settlements to 23 municipalities. Montenegro is the smallest country on this map by
population after Liechtenstein's absence, and the reason to draw it is that its religious
boundary is the sharpest in the Balkans over the shortest distance.

    69.32%  Pravoslavna              -> christianity.orthodox   (the PARENT, see REVIEW)
    17.91%  Islamska                 -> islam
     4.69%  zaštićen podatak         -> EXCLUDED (disclosure control, §3.8)
     3.07%  Katolička                -> christianity.catholic
     2.15%  Ateista                  -> secular
     1.92%  Ne želi da se izjasni    -> EXCLUDED (explicit refusal, §3.5)
     0.37%  Ostale hrišćanske        -> christianity   (the root)
     0.34%  Agnostik                 -> secular
     0.09%  Ostalo                   -> unknown
     0.06%  Protestanti              -> christianity.protestant
     0.05%  Ostale vjere             -> other.me   (a NEW node)
     0.03%  Jehovini svjedoci        -> christianity.witnesses
     0.02%  Budisti                  -> buddhism

**MONTENEGRO HAS NO `unaffiliated` CELL AND THAT IS A PROPERTY OF THE FORM, NOT THE
COUNTRY.** There is no "no religion" answer anywhere in this table — the irreligious options
are `Ateista` and `Agnostik`, which are *positions* and map to `secular`, and beside them sits
a large `Ne želi da se izjasni`. So Montenegro is unlit for `unaffiliated` (coverage.py, §6.12)
and that is right: nobody was offered the box. Its 2.49% secular is therefore not comparable
with Czechia's or Estonia's no-religion share, which answers a different question.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Ukupno":
        "the unit's own population total, not a category.",
    "Ne želi da se izjasni":
        "11,924 people, 1.92%, and it is not irreligion — `Ateista` and `Agnostik` are "
        "separate answers taken by 15,508 between them. An explicit refusal, so spec §3.5 "
        "applies and these people are not drawn. Its geography is the north and the coast "
        "rather than the capital, which is the opposite of the Czech and Kosovar pattern "
        "and is worth a look if this country is ever revisited.",
    "zaštićen podatak":
        "29,225 people, 4.69% — NOT a category MONSTAT published but the residual "
        "`sources/me.py` computes: each unit's own total less the cells it printed, i.e. "
        "the mass hidden behind `z`, *zaštićen podatak*, under Montenegro's Law on Official "
        "Statistics. **Spec §3.8 removes suppressed cells rather than placing them**, and "
        "§6.3a's `unknown` names suppression as the one thing it will not take. It is "
        "carried in me.csv so the partition reconciles and so countries.py can state it "
        "with `gap=`, and it is drawn nowhere. What it costs is reported per category by "
        "`me.py`'s check() rather than as one headline number, which is Lithuania's lesson "
        "(§9q): 4.69% reads as nothing, and the truth is that Islam loses 10.6% of itself "
        "against Orthodoxy's 2.7%.",
}

REVIEW = {
    "Pravoslavna":
        "-> christianity.orthodox, the PARENT, and deliberately not "
        "christianity.orthodox.canonical, which is where hr2021.py and rs2022.py file "
        "Croatia's and Serbia's bare `Pravoslavci`. **This is mk2021.py's call, for "
        "Montenegro's version of the same problem, and here it is not history but current "
        "events.** The category names no jurisdiction. Two churches claim these 431,526 "
        "people: the **Serbian Orthodox Church**, canonical, with the great majority and "
        "the monasteries; and the **Montenegrin Orthodox Church**, self-declared "
        "autocephalous in 1993 and recognised by no canonical body, which is "
        "`christianity.orthodox.other`'s exact definition. The split is not marginal and it "
        "is not settled — the 2019 Law on Freedom of Religion and the 2020 litije "
        "processions brought a government down over precisely this question, and the 2023 "
        "census was taken four years later with the dispute live. Filing all of them as "
        "canonical asserts the Serbian Church's claim; filing any as `.other` requires a "
        "number nobody has. The parent says what the source says.",
    "Islamska":
        "-> islam, the parent, with no branch. Montenegro's Muslims are overwhelmingly "
        "Hanafi Sunni — Bosniaks in Rožaje, Plav, Petnjica and Bijelo Polje in the "
        "Sandžak north, Albanians in Ulcinj and Tuzi in the south — and the census does not "
        "say so. **Note that `Islamska` here is one cell**: unlike the 2011 census, which "
        "offered `Islamska` and `Muslimanska` separately and split the community between a "
        "faith answer and an ethno-religious one, 2023 asks once. That is a simplification "
        "of the form rather than of the country.",
    "Katolička":
        "-> christianity.catholic, the parent. Montenegro's Catholics are **Albanian and "
        "Croat, and they are two separate communities in two corners of the country** — the "
        "Albanians of Tuzi and Ulcinj in the south-east, and the Croats of the Bay of Kotor, "
        "whose Catholicism is Venetian and six centuries old. The Archdiocese of Bar is the "
        "primatial see of Serbia by title and one of the oldest in the region. At 3.07% this "
        "is a small category with a very sharp geography, which is the same reason Kosovo's "
        "1.75% was worth drawing.",
    "Ostale hrišćanske":
        "-> christianity, the ROOT, which is the node for an answer that names Christianity "
        "and no church. 2,287 people. The same call mk2021.py makes for `Christians`, "
        "au2021.py for 'Christianity, nfd' and cz2021.py for `křesťanství`. It sits beside "
        "named Protestant and Witness cells, so it is a residual *within* Christianity "
        "rather than a synonym for one of them.",
    "Ateista":
        "-> secular, with `Agnostik`, and NOT `unaffiliated`. Both are stated positions "
        "rather than the absence of an answer, which is exactly the distinction "
        "branches.py draws between the two nodes. See the module docstring: Montenegro "
        "offers no no-religion box at all, so `unaffiliated` is not in this country's "
        "coverage and its 2.49% secular must not be read as Czechia's 47%.",
    "Agnostik":
        "-> secular, with `Ateista`. 2,106 people. Kept as the same node rather than split, "
        "because the tree has no atheist/agnostic children and inventing them for one "
        "country would be §2's 'a node earns its place by being countable somewhere' read "
        "backwards.",
    "Ostalo":
        "-> unknown (§6.3a). MONSTAT prints this cell **beside** `Ostale vjere`, so it is "
        "not another religion residual — that one is already spoken for — and it publishes "
        "no definition of it. 562 people, 0.09%. It goes to `unknown` on that node's own "
        "test: the source counted them and what they practise is not determinable from the "
        "table. **This is the weakest call in the file**, and the alternative — excluding it "
        "outright as §3.5 does for refusals — would differ by 562 people in a country of "
        "622,537. If a MONSTAT methodology note ever defines it, this should be revisited.",
    "Ostale vjere":
        "-> other.me, a per-source residual (§3.11). See branches.py.",
    "Jehovini svjedoci":
        "-> christianity.witnesses. 168 people, and the smallest named cell on the map "
        "outside Portugal's Belmonte. Worth keeping rather than folding into "
        "`Ostale hrišćanske`: the source named it, and §2 says the tree grows where a "
        "source reaches.",
    "Budisti":
        "-> buddhism, the parent, 100 people. No school is named and none could be guessed; "
        "Montenegro has no historic Buddhist community and this is recent and urban.",
}

MAP = {
    "Pravoslavna": "christianity.orthodox",
    "Islamska": "islam",
    "Katolička": "christianity.catholic",
    "Protestanti": "christianity.protestant",
    "Jehovini svjedoci": "christianity.witnesses",
    "Ostale hrišćanske": "christianity",
    "Agnostik": "secular",
    "Ateista": "secular",
    "Budisti": "buddhism",
    "Ostale vjere": "other.me",
    "Ostalo": "unknown",
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
