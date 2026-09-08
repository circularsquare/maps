"""BHAS Popis 2013 religion classification -> religiondots taxonomy.

Eight categories plus the universe total, at municipality. **Only four of the eight are
religions and three of those are the three that define the country**, which makes this one
of the shallowest question designs on the map — shallower than Croatia's twelve, on the
same peninsula, about a population with the same denominational variety.

    50.70%  Islamska            -> islam
    30.75%  Pravoslavna         -> christianity.orthodox.canonical
    15.19%  Katolička           -> christianity.catholic
     1.15%  Ostali              -> other.ba   (a NEW node)
     0.93%  Nisu se izjasnili   -> EXCLUDED (declined to declare)
     0.79%  Ateist              -> secular
     0.31%  Agnostik            -> secular
     0.19%  Bez odgovora        -> EXCLUDED (no answer at all)

**THE SHALLOWNESS IS THE POINT AND IT IS NOT AN ACCIDENT OF THE FORM.** The 2013 census
asked religion (`vjeroispovijest`) beside ethnicity (`etnička/nacionalna pripadnost`) and
mother tongue, and the three answers are near-substitutes in Bosnia: Bosniak/Muslim,
Serb/Orthodox, Croat/Catholic. A form that offered ten Christian denominations would have
been answered as though it offered two, because what the question actually measures here is
which of three communities a person belongs to. That is a fact about the country rather
than a defect in the instrument, and it is why the map of Bosnia is legible at a glance and
tells you almost nothing about doctrine.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Ukupno":
        "the unit's own population total, not a category.",
    "Nisu se izjasnili":
        "32,700 people, 0.93%, who declined to declare a religion. Not a religion and not "
        "a report of no religion — `Ateist` and `Agnostik` are separate answers taken by "
        "38,669 between them. Spec §3.5, so these people are not drawn. **Its geography "
        "is Sarajevo**: Centar 4.28% and Novo Sarajevo 4.16% against 0.93% nationally, "
        "the same urban and educated pattern the refusal takes in Czechia, Hungary and "
        "Kosovo.",
    "Bez odgovora":
        "6,588 people, 0.19%, for whom no answer was recorded at all. BHAS keeps this "
        "apart from `Nisu se izjasnili`, which is an active refusal, so this file does "
        "too — Serbia's §9p pair, and the same reasoning. **It has one strange unit and "
        "it is worth writing down: Fojnica is 6.24%, 771 of 12,356 people, against 0.19% "
        "nationally and 2.69% in the next-highest.** That is 33x the national rate in a "
        "single municipality and it is an enumeration artefact rather than a fact about "
        "Fojnica — the sort of thing §3.8 predicts and no note can repair. Not drawn "
        "either way.",
}

REVIEW = {
    "Islamska":
        "-> islam, the parent, with no branch. Bosnia's Muslims are Hanafi Sunni under "
        "the Islamic Community (Islamska zajednica), an institution with an unbroken line "
        "to the Ottoman period and an unusual degree of autonomy, and the census does not "
        "say so. What is lost by not splitting is what xk2024.py loses for Kosovo — the "
        "**Sufi tekkes**, Naqshbandi and Qadiri, which have a continuous Bosnian presence "
        "and which no census category would have found anyway, because a Bosnian dervish "
        "answers `Islamska`. 1,790,454 people, 50.7%, and it makes this **the only "
        "majority-Muslim country in Europe on this map besides Kosovo and Albania**.",
    "Pravoslavna":
        "-> christianity.orthodox.canonical. The Serbian Orthodox Church, canonical "
        "throughout, with the metropolitanate of Dabar-Bosnia at Sarajevo and the "
        "eparchies of Banja Luka, Zvornik-Tuzla, Zahumlje and Bihać-Petrovac. 1,085,760 "
        "people. branches.py splits Orthodoxy by communion rather than by jurisdiction, "
        "so this sits with the Romanian, Greek and Russian churches.",
    "Katolička":
        "-> christianity.catholic, the PARENT, not christianity.catholic.latin. The "
        "category is `Catholic` with no rite. Bosnia's Catholics are Latin-rite and "
        "Croat, and the Bosnian church is unusual in one respect worth recording even "
        "though the tree cannot express it: **the Franciscan province of Bosna Argentina "
        "has held the pastoral care of Bosnian Catholics since the 1290s** and continued "
        "through four centuries of Ottoman rule, which is why the diocesan structure is "
        "thin and the friaries — Fojnica, Kreševo, Kraljeva Sutjeska — are the old "
        "centres. hr2021.py and cz2021.py file their equivalents at the same node.",
    "Ateist":
        "-> secular, and `Agnostik` with it. branches.py draws the line at whether a "
        "POSITION is stated, and both of these state one: BHAS offers them as bare "
        "`Ateist` and `Agnostik`, with none of the 'not a believer' gloss that sends "
        "Serbia's and North Macedonia's equivalents to `unaffiliated` instead. This is "
        "ro2021.py's call for `Ateu / Agnostic`, made the same way for the same reason. "
        "**Bosnia has no no-religion answer at all** — the form's only irreligious "
        "options are these two positions — so nothing here lands on `unaffiliated`, and "
        "the 1.10% they sum to is not comparable with the no-religion shares elsewhere "
        "on this map. It is a floor rather than a measurement.",
    "Agnostik":
        "-> secular, with `Ateist`. 10,816 people, and kept as its own source category "
        "so §2.4's deferred matching can split them later if a node is ever wanted. Both "
        "are overwhelmingly Sarajevo: Centar is 8.60% irreligious against 1.10% "
        "nationally, and the four Sarajevo municipalities plus Tuzla hold most of it.",
    "Ostali":
        "-> other.ba, a per-source residual (§3.11). It carries Bosnia's Jews, its "
        "Protestants and its unclassifiable write-ins, and it has one sharp and "
        "unexplained peak at Velika Kladuša. See branches.py.",
}

MAP = {
    "Islamska": "islam",
    "Pravoslavna": "christianity.orthodox.canonical",
    "Katolička": "christianity.catholic",
    "Ostali": "other.ba",
    "Ateist": "secular",
    "Agnostik": "secular",
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
