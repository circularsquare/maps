"""Afrobarometer Cameroon religion -> religiondots taxonomy.

Seven categories at 12 units (the 10 regions, with Mfoundi/Yaoundé and Wouri/Douala apart), on
COD-PS 2025. `Christian` is the card's umbrella answer with every church folded back in except the
two drawn on their own; `Muslim` likewise; the rest are the card's own boxes. `sources/cm.py`'s
docstring has the argument for which churches are drawn and `sources/cm.md` the record. The same
instrument and construction as `tz2022.py` and `ng2022.py`.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Christian":
        "-> christianity, the bare branch, holding everything Christian except Presbyterians and "
        "Baptists: Roman Catholics (35% of the pooled respondents), `Christian only` (10.5%), "
        "Evangelical (5.7%; in Cameroon most likely the Reformed Eglise Evangelique du Cameroun "
        "of the Ouest and Littoral, not `christianity.evangelical`), Pentecostal (4.7%), Lutheran "
        "(2.4%), Adventist, Orthodox and the small boxes. Catholics are not drawn because their "
        "share moves 26.3-41.4% by round while `Christian only` doubles; Evangelical moves "
        "2.7-8.9%. **Lutherans are level (2.1-2.4%) and pass the split-half (+0.61) and are still "
        "folded in**: they live in Adamaoua, Nord and Extreme-Nord, where 18-35% of Christians "
        "name no church, so a Lutheran share would be drawn short exactly where it lives "
        "(`sources/cm.py::unnamed_where_they_live`, 22.6% against 13.0% nationally). **Orthodox** "
        "(68 respondents) is the card's Christian box and stays here, though 13 of round 9's 21 "
        "are in Nord and 10 of round 6's 20 in Extreme-Nord, which looks like one team's coding; "
        "the census counted Orthodox at 0.5% in 2005. Round 7's Cameroon-only `Protestant` code "
        "(17) names no church and stays here too.",
    "Presbyterian":
        "-> christianity.reformed.presbyterian. 9.36% of the pooled survey, 9.00% as drawn: the "
        "Presbyterian Church in Cameroon of the Nord-Ouest (27.2%) and Sud-Ouest (22.5%) and the "
        "Eglise Presbyterienne Camerounaise of the Sud (22.1%) and Centre, which the card cannot "
        "tell apart. Drawn on its own because its share holds at 8.6-10.9% across rounds while "
        "`Christian only` goes from 6.7% to 13.5%, it passes the split-half at +0.860, and it "
        "lives where Christians name their church (unnamed share 10.7% where its respondents are, "
        "against 13.0% nationally). The census's 2005 Protestant 26.3% is the outside witness at "
        "family level: the survey's Protestant bodies come to about 27% without `Christian only`. "
        "A floor all the same, by however many Presbyterians answered just `Christian`. A "
        "reviewer (sources/cm.md §4) would have folded every church; this is the call to reverse "
        "if that view wins.",
    "Baptist":
        "-> christianity.baptist. 3.59% of the pooled survey, 3.54% as drawn: the Cameroon "
        "Baptist Convention of the Nord-Ouest (10.5%) and Sud-Ouest (10.2%), the Union des "
        "Eglises Baptistes of the Littoral (6.9%) and Wouri, and a Baptist presence in "
        "Extreme-Nord (4.2%). Level at 3.2-4.1% by round, split-half +0.800. **The closest call "
        "of the two**: its unnamed share is 12.5% against the national 13.0%, pulled up by the "
        "Extreme-Nord respondents. None of Adamaoua's 309 interviews was a Baptist, so it is "
        "drawn at zero there.",
    "Muslim":
        "-> islam, with no branch. The card's `Sunni only` box takes 53 people over five rounds "
        "and 1.4-2.0% in round 7 against 0.2% in round 9, and the Tijaniyya 12, so no branch "
        "share would survive pooling. Adamaoua 63.1%, Nord 41.6%, Extreme-Nord 38.4%.",
    "None":
        "-> unaffiliated. 3.80% of the pooled survey and 3.82% as drawn, with its own geography "
        "(split-half +0.839): Ouest 6.8%, Littoral 6.7%, Mfoundi 6.3%, almost none in the "
        "Nord-Ouest and Sud-Ouest. The card offers `Traditional/ethnic religion` as a separate box "
        "on every pooled round (`sources/cm.py::report_card` reads the value labels), so `None` is "
        "what these respondents chose with the traditional answer in front of them: step 2 of "
        "the draft no-religion procedure. It is a self-description. The 2005 census's `Libre "
        "penseur` (free-thinker) was 3.2%, which is the same order.",
    "Traditional/ethnic religion":
        "-> indigenous.african. 0.55% of the pooled survey, 27 respondents over five rounds, and "
        "a floor: the card offers it as a peer of Christian and Muslim, so it counts only people "
        "who give it as their one religion. **The 2005 census counted 5.6% animist** (9.7% of "
        "rural Cameroon), ten times the survey's share; the map draws the survey, and the note "
        "says so. Fails the split-half (+0.255) and is under the 1% floor, so it is spread by "
        "the residual (worst 1.05x its national share in a unit where nobody gave it).",
    "Other":
        "-> other.cm. 1.19% of the pooled survey, the card's `Other` (58) and `Bahai` (4). It "
        "passes the split-half (+0.633) but is not placed: all 58 `Other` answers are in rounds "
        "5-7 and nobody chose the box in rounds 8 or 9 although it was on both cards, so its "
        "pooled share depends on which rounds are in (Izala's case, `ng2022.py`). Spread by the "
        "residual, worst 1.71x in Ouest.",
}

MAP = {
    "Christian": "christianity",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Baptist": "christianity.baptist",
    "Muslim": "islam",
    "Traditional/ethnic religion": "indigenous.african",
    "None": "unaffiliated",
    "Other": "other.cm",
}

# spec §7a-i-1: every row is measured at the node it is drawn on; nothing is inferred downward.
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
