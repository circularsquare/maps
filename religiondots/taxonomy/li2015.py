"""Liechtenstein — Volkszählung 2015 religion classification -> religiondots taxonomy.

Eleven categories plus the universe total, over 11 communes. **The smallest country on this
map**, and one of the finest per head at 3,420 people per unit.

    73.36%  Roman Catholic                    -> christianity.catholic.latin
     6.97%  No religious affiliation          -> unaffiliated
     6.29%  Protestant Reformed               -> christianity.reformed
     5.89%  Islamic religious communities     -> islam
     3.27%  Not stated                        -> EXCLUDED (non-response)
     1.25%  Christian Orthodox                -> christianity.orthodox.canonical
     1.19%  Protestant Lutheran               -> christianity.lutheran
     0.69%  Other Protestant communities      -> christianity.protestant
     0.48%  Buddhistic religious communities  -> buddhism
     0.34%  Other religious communities       -> other.li
     0.28%  Other Christian communities       -> christianity

**THE FORM SEPARATES REFORMED FROM LUTHERAN, WHICH ALMOST NOTHING ELSE HERE DOES.** Two
Protestant state-recognised churches exist in Liechtenstein — the Evangelische Kirche
(Reformed, Swiss-facing) and the Evangelisch-Lutherische Kirche (Austrian- and German-facing)
— and the census counts them apart, on 2,365 and 447 people respectively. Most European
sources on this map offer one Protestant box for a population a thousand times larger.

**THERE IS NO JEWISH CELL**, and that is a fact about the form rather than the country:
Liechtenstein's Jewish residents are inside `Other religious communities`, 128 people in
total. Selecting Judaism therefore leaves Liechtenstein **unlit**, which is `coverage.py`'s
correct answer — the question was not put.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Religion - total":
        "the unit's own population total, not a category.",
    "Not stated":
        "1,229 people, 3.27%, and it is not irreligion — `No religious affiliation` is a "
        "separate answer taken by 2,623. spec §3.5 applies and these people are not drawn. "
        "**Its geography is flat**, 1.2% in Ruggell to 3.8% in Schaan, which is unlike the "
        "urban, educated pattern the refusal takes in Czechia, Hungary and Kosovo; in a "
        "country of 37,622 with a mandatory census it is closer to ordinary form attrition "
        "than to a stance.",
}

REVIEW = {
    "Roman Catholic":
        "-> christianity.catholic.latin. **73.36%, and the Roman Catholic Church is the "
        "state church** — Article 37 of the constitution names it the Landeskirche, which "
        "makes Liechtenstein one of the last places in Europe where that is literally true. "
        "Disestablishment has been under discussion since 2012 and has not happened. Filed "
        "on `.latin` rather than the `christianity.catholic` parent because the form names "
        "the Roman church specifically and offers no other Catholic cell.",
    "Protestant Reformed":
        "-> christianity.reformed. The Evangelische Kirche im Fürstentum Liechtenstein, "
        "which is Swiss Reformed in descent and looks across the Rhine. 2,365 people. Kept "
        "apart from the Lutherans below because the source keeps them apart.",
    "Protestant Lutheran":
        "-> christianity.lutheran. The Evangelisch-Lutherische Kirche im Fürstentum "
        "Liechtenstein, Austrian and German in descent, 447 people and a separate "
        "state-recognised body. **This distinction is the reason to draw the country's "
        "Protestant cells at all** — at 0.4% of a country of 37,000 it is 447 individuals, "
        "and almost no other source on this map can separate Reformed from Lutheran.",
    "Other Protestant communities":
        "-> christianity.protestant, which holds the ANSWER 'Protestant' where no body is "
        "named and is deliberately not a parent of the Protestant families (branches.py). "
        "259 people, the residual of a three-cell Protestant question.",
    "Christian Orthodox":
        "-> christianity.orthodox.canonical. 472 people, and they are almost entirely "
        "foreign nationals — 338 of the 472 — which is the Balkan and Eastern European "
        "labour migration rather than a historic community. Planken has none at all.",
    "Other Christian communities":
        "-> christianity, the ROOT, for an answer that is Christian and names no church. "
        "105 people. It sits beside named Catholic, Reformed, Lutheran, other-Protestant "
        "and Orthodox cells, so it is genuinely what is left when none of those fits.",
    "Islamic religious communities":
        "-> islam, the parent, with no branch. **5.89% and much the largest non-Christian "
        "group**, which is high for a microstate and is Turkish and Bosnian in origin — the "
        "guest-worker migration that built the industrial belt. **13.1% of foreign nationals "
        "against 2.2% of Liechtenstein citizens**, the sharpest citizenship split in the "
        "table, and its geography follows: Eschen 11.4% and Gamprin 8.0% against Planken "
        "0.2% and Schellenberg 1.2%.",
    "Buddhistic religious communities":
        "-> buddhism, the parent. 180 people, no school named, and nothing that could be "
        "guessed from a country this size.",
    "Other religious communities":
        "-> other.li, a per-source residual (§3.11). See branches.py — and note it is where "
        "Liechtenstein's Jews are, because the form has no Jewish cell.",
    "No religious affiliation":
        "-> unaffiliated and NOT `secular`. A report of no affiliation rather than a stated "
        "atheist or humanist position, which is branches.py's distinction between the two. "
        "6.97% — **the lowest in Western Europe on this map** and about a fifth of "
        "Switzerland's, twenty kilometres away.",
}

MAP = {
    "Roman Catholic": "christianity.catholic.latin",
    "Protestant Reformed": "christianity.reformed",
    "Protestant Lutheran": "christianity.lutheran",
    "Other Protestant communities": "christianity.protestant",
    "Christian Orthodox": "christianity.orthodox.canonical",
    "Other Christian communities": "christianity",
    "Islamic religious communities": "islam",
    "Buddhistic religious communities": "buddhism",
    "Other religious communities": "other.li",
    "No religious affiliation": "unaffiliated",
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
