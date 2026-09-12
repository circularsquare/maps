"""Austrian Volkszählung 2001 religion classification -> religiondots taxonomy.

Ten categories plus the universe total, at Gemeinde (and at Zählbezirk inside Vienna).

    73.64%  Römisch-katholisch                      -> christianity.catholic.latin
    11.99%  Ohne Bekenntnis                         -> unaffiliated
     4.68%  Evangelisch                             -> christianity.protestant
     4.22%  Islamisch                               -> islam
     2.23%  Orthodox                                -> christianity.orthodox
     2.00%  Unbekannt                               -> EXCLUDED (non-response)
     0.86%  Andere christliche Gemeinschaften       -> christianity
     0.25%  Andere nichtchristliche Gemeinschaften  -> other.at
     0.10%  Israelitisch                            -> judaism
     0.02%  Griechisch-katholisch                   -> christianity.catholic.eastern

**THE TWO `ANDERE` CELLS ARE NOT SPLIT, AND THAT IS A DECISION RATHER THAN A LIMITATION.**
`sources/at.py` proves an exact decomposition of both against UNSD's 31 national rows — the
Christian cell is Jehovah's Witnesses 23,206 + Old Catholics 14,621 + eleven more, the
non-Christian cell is Buddhists 10,402 + Hindus 3,629 + Sikhs 2,794 + Bahá'í 760 and three
others, each closing to the person with no remainder. So the split *could* be applied here,
the way Germany's ESS residual is.

It is not, and the reason is that **Austria publishes those 31 figures at ONE geography: the
country.** Germany's split buys nine Bundesländer of variation and India's buys states;
Austria's would buy none at all. Every one of 2,380 units would receive the identical internal
mix, so the map would gain a dozen legend rows while gaining not one spatial fact, and 88,977
measured people would become derived to pay for it. §14's first rule is not to estimate a
magnitude at a finer resolution than the source publishes it, and a national constant pushed
down to Gemeinde is that in its purest form.

What the reader gets instead is `note_public` naming the contents and their national counts,
which is the same information without the map asserting where any of it lives. **The
decomposition is banked in `sources/at.py`'s `CROSSWALK` and is asserted on every run**, so if
this call is ever reversed the work is ten minutes and not a day.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Insgesamt":
        "the unit's own population total, not a category.",
    "Unbekannt":
        "religion not stated. 160,662 people, 2.00%, and NOT the same answer as `Ohne "
        "Bekenntnis`, which 963,263 people gave and which is drawn. Conflating a refusal "
        "with a report of no religion is the §3.1 error this project exists to avoid. "
        "Under §3.5 these people are marked and not filled. **Which way the hole leans is "
        "measurable and is measured**: across the drawn units the `Unbekannt` share "
        "correlates +0.43 with the `Ohne Bekenntnis` share, +0.41 with the Orthodox share "
        "and −0.39 with the Roman Catholic share. So the non-response is drawn "
        "disproportionately from the least religious places and from the minority ones, and "
        "every share this map draws for Austria is therefore slightly more Catholic than "
        "Austria was. Vienna is 4.24% unknown against 0.57% in Burgenland.",
}

REVIEW = {
    "Römisch-katholisch":
        "-> christianity.catholic.latin, and NOT the `christianity.catholic` parent, "
        "because the census counts the Greek Catholics in their own cell and the Old "
        "Catholics inside `Andere christliche Gemeinschaften`, so this cell means the Roman "
        "church specifically. 5,915,421 people, 73.6% — and the single most out-of-date "
        "figure on this map, since the 2021 Mikrozensus supplement puts it at 55.2%. That "
        "fall is the country's largest religious change in a century and none of it is "
        "visible here; countries.py's note says so in as many words.",
    "Griechisch-katholisch":
        "-> christianity.catholic.eastern. The Byzantine-rite churches in communion with "
        "Rome — in Austria overwhelmingly Ukrainian, around the Barbareum in Vienna, with "
        "Hungarian, Slovak and Romanian Greek Catholics behind them. **1,853 people, the "
        "smallest cell in the table and the only one under 0.05%.** It is a separate "
        "printed column rather than part of the Roman one, which is unusual for a European "
        "census and is a legacy of Habsburg confessional bookkeeping; it is kept separate "
        "here for the same reason.",
    "Orthodox":
        "-> christianity.orthodox, the PARENT, and deliberately not "
        "`christianity.orthodox.canonical` the way Switzerland's equivalent cell is mapped. "
        "The reason is arithmetic rather than doctrinal: `sources/at.py`'s crosswalk shows "
        "this column is UNSD's Orthodox (159,115) plus Greek Orthodox (18,533) plus "
        "**Armenian Apostolic (1,824)**, and the Armenian church is Oriental Orthodox, not "
        "in the Eastern Orthodox communion at all — it sits under `christianity.oriental` "
        "on this tree. 1.0% of the cell is therefore in a different branch from the other "
        "99%, the census does not separate them at any geography, and the honest node is "
        "the one that is true of all of it. The cell is mostly Serbian and Romanian "
        "migration and has more than doubled since, to 4.9% in 2021.",
    "Evangelisch":
        "-> christianity.protestant, which on this tree holds the ANSWER 'Protestant' and "
        "is deliberately not a parent of anything (branches.py). Austria's Evangelische "
        "Kirche is a united body, **A.B. und H.B.** — Augsburg Confession (Lutheran) and "
        "Helvetic Confession (Reformed) — and the census prints one undivided cell for "
        "both, as does UNSD. Mapping it to `christianity.lutheran` would be the more "
        "informative call and is very likely true of about 96% of these 376,150 people, but "
        "it is a split the source does not make and §3.1 forbids adding one. The geography "
        "is worth reading anyway: Protestantism in Austria is not evenly thin but "
        "concentrated in Burgenland (13.3%) and Kärnten (10.3%) against 2.2% in Vorarlberg "
        "and 2.4% in Tirol, which is the Counter-Reformation's map — the Carinthian valleys "
        "kept a crypto-Protestant population through it, and Burgenland was Hungarian.",
    "Andere christliche Gemeinschaften":
        "-> christianity, the root, because the cell is thirteen bodies with nothing in "
        "common but not being one of the four named above. From the crosswalk, in order: "
        "Jehovah's Witnesses 23,206, Old Catholics 14,621, Free Christian Community 7,186, "
        "Evangelical 4,892, Seventh-day Adventists 4,220, New Apostolic 4,217, Church of "
        "England 2,317, Latter-day Saints 2,236, Baptists 2,108, Christian Community "
        "1,428, Methodists 1,263, Christengemeinschaft 1,152, Mennonites 381. Several of "
        "those have their own node and would take it happily; see the module docstring for "
        "why they are not given one here.",
    "Andere nichtchristliche Gemeinschaften":
        "-> other.at. 19,750 people, 0.25%, and it is a genuine mixture rather than a tail: "
        "Buddhists 10,402, Hindus 3,629, Sikhs 2,794, `other religions` 1,745, Bahá'í 760, "
        "Unification 297, Shinto 123. **Buddhism is more than half of it**, which is worth "
        "saying because Austria recognised Buddhism as a public-law religious society in "
        "1983, the first country in Europe to do so. Under §3.11 the residual is drawn as a "
        "residual; what it holds is named in `note_public` rather than smeared across 2,380 "
        "Gemeinden at a constant rate.",
    "Ohne Bekenntnis":
        "-> unaffiliated rather than `secular`. The Austrian form asked for a "
        "Religionsbekenntnis and this is the box for having none; it is a statement about "
        "membership of a Religionsgesellschaft, which in Austria is a legal category "
        "carrying a church-tax obligation, and not a statement of belief. Nothing in the "
        "census supports reading it as atheism, and `secular` on this tree is for sources "
        "that asked. 963,263 people, 12.0% in 2001 and 22.4% by 2021 — the fastest-growing "
        "answer in the country, and in Vienna already 25.6% at the census.",
}

MAP = {
    "Römisch-katholisch": "christianity.catholic.latin",
    "Griechisch-katholisch": "christianity.catholic.eastern",
    "Orthodox": "christianity.orthodox",
    "Evangelisch": "christianity.protestant",
    "Andere christliche Gemeinschaften": "christianity",
    "Israelitisch": "judaism",
    "Islamisch": "islam",
    "Andere nichtchristliche Gemeinschaften": "other.at",
    "Ohne Bekenntnis": "unaffiliated",
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
    if c in MAP:
        return MAP[c]
    raise KeyError("at2001: unmapped source category %r" % cat)
