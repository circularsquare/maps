"""
IHSI ECVMAS 2012 `I_H04` (Haiti) -> religiondots taxonomy.

**Twelve answers, and it is a Haitian card rather than a continental one.** ECVMAS's
individual questionnaire asks *"Quelle est votre religion ?"* of everyone aged ten and over
and offers the seven churches a Haitian interviewer expects to hear named, a catch-all
Protestant box, Vodou, Islam, a specify box and a no-religion box. Named for the survey year,
per the registry convention.

    47.64%  Catholique          -> christianity.catholic.latin
    17.55%  Baptiste            -> christianity.baptist
    10.78%  Pentecôtiste        -> christianity.pentecostal
     9.22%  Autre protestant    -> christianity.protestant
     6.75%  Aucune              -> unaffiliated
     3.15%  Adventiste          -> christianity.adventist
     1.50%  Vaudou              -> afrodiasporic.vodou
     1.30%  Méthodiste          -> christianity.methodist
     1.11%  Autre               -> other.ht
     0.54%  Episcopale          -> christianity.anglican.episcopal
     0.46%  Témoin de Jéhovah   -> christianity.witnesses
     0.00%  Musulman            -> islam

Those are the shares as drawn, off `data/normalized/ht.csv`. Every row is `modelled` in §7's
sense: a survey share against a COD-PS 2024 projection, in a country whose last census was in
2003. **And the universe is people aged ten and over**, applied to the whole population;
`sources/ht.py` measures what that costs against the 2003 census, which asked everyone.

## `Vaudou` IS A NEW NODE AND IT IS THE ONE TO READ THE CAVEAT ON

`afrodiasporic.vodou` was added with this country. Haiti is where the family's best-known
tradition lives and it had no node, because the countries drawn before it counted Candomblé,
Umbanda, Revival Zion, Orisha and the Spiritual Baptists instead. The number is a floor and
`branches.py`'s note says why at length: Vodou is served alongside Catholicism far more often
than instead of it, and a questionnaire with one box collects the church.

## `Autre protestant` GOES TO THE UNSPECIFIED PROTESTANT NODE, NOT TO `evangelical`

The Dominican Republic's `EVANGÉLICA` (`do2019.py`) went to `christianity.evangelical`
because it was the only non-Catholic box beside three named ones and *evangélico* is the
Dominican word for a Protestant. **Haiti's card is the opposite shape.** It names Baptists,
Pentecostals, Adventists, Methodists, Episcopalians and Witnesses individually and then adds
`Autre protestant` for whoever is left, which is `christianity.protestant`'s own definition:
a Protestant this source declined to place. Routing it to `evangelical` would assert that the
Church of God, the Nazarenes and the Salvation Army congregations it holds are all
evangelical in the sense the node means, which the questionnaire never asked.

**It is not a small residual.** At 9.22% it is larger than the Adventists, Methodists,
Episcopalians and Witnesses put together, and ECVH 2001's card is the reason to believe most
of it is Pentecostal in practice rather than historic Protestant: that survey offered
`Eglise de Dieu` as its own answer and got 9.4% of the country, more than its
`Pentecôtiste` box, and ECVMAS has no Church of God box at all. `sources/ht.py` treats the
two as one bloc for the cross-check for exactly that reason.
"""

EXCLUDED = {}

REVIEW = {
    "Autre protestant":
        "-> christianity.protestant, the unspecified node, rather than "
        "christianity.evangelical. 9.22% of the country and the fourth largest answer. The "
        "card names six Protestant bodies individually and then offers this, so it is a "
        "Protestant the source declined to place, which is what the node is for. The "
        "Dominican Republic's `EVANGÉLICA` went the other way (do2019.py) because there the "
        "one box was the ONLY general Protestant answer and carried the word Dominicans "
        "use. The Church of God, which ECVH 2001 counted separately at 9.4% of Haiti and "
        "which has no box here, is most of what this cell is likely to hold, and it is "
        "Pentecostal; `christianity.pentecostal` was rejected anyway, because a cell whose "
        "own label says `other Protestant` cannot be assigned to a named family.",
    "Vaudou":
        "-> afrodiasporic.vodou, a node added with this country. Filed beside Candomblé and "
        "the Spiritual Baptists rather than under Christianity, on spec §3.3's rule that a "
        "genuinely double descent gets its own node. **The 1.50% is a floor and the whole "
        "geography is one department**: Artibonite at 5.9% against 1.2% for the next "
        "highest, and dropping it collapses the national spread from 5.7 points to 1.2. "
        "Both IHSI instruments agree on that heartland and on Nord-Est at the bottom, and "
        "on nothing in between, which is why sources/ht.py carries an OVERRIDE for it and "
        "note_public tells the reader to read the ends only.",
    "Aucune":
        "-> unaffiliated, and NOT `unchurched`, matching do2019.py and against the three "
        "LAPOP countries in this region. The difference is the card: LAPOP prints `Ninguna "
        "(cree en un Ser Superior…)` and offers `Agnóstico o ateo` separately, ECVMAS "
        "prints `Aucune` and asks nothing about belief. 6.75% of the country, and **it is "
        "the largest answer this map does not place**: its split-half median is +0.56 "
        "against a bar of +0.65 and ECVH 2001's card had no no-religion box at all, so "
        "there is no outside witness to override on. Every department is drawn at 6.75%.",
    "Episcopale":
        "-> christianity.anglican.episcopal rather than the bare anglican node. The Église "
        "Épiscopale d'Haïti is a diocese of the Episcopal Church in the United States and "
        "has been since 1861, so the specific node is the accurate one rather than an "
        "inference. 0.54%, and drawn at the national rate: split-half +0.33.",
    "Autre":
        "-> other.ht. 1.11%, and the only answer of the twelve with a NEGATIVE split-half "
        "rank (-0.25), so it is drawn at the national rate in every department. The 2003 "
        "census counted Mormons separately at 5,683 people and ECVMAS has no box for them, "
        "so they are the largest identifiable group inside this cell.",
    "Musulman":
        "-> islam. **One respondent in 17,977**, weighting to 0.004% of the country, which "
        "rounds to nothing in every department. The 2003 census found 2,013 Muslims, 0.02% "
        "of Haiti, so the order of magnitude agrees and this is a real if very small "
        "population rather than a coding error. Kept rather than excluded because §2.4 "
        "keeps the source's own answers and a zero-count row is honest about what the "
        "survey found.",
}

MAP = {
    # ---------------------------------------------------------------- Catholic
    "Catholique": "christianity.catholic.latin",

    # ---------------------------------------------------------------- named Protestant bodies
    "Baptiste": "christianity.baptist",
    "Pentecôtiste": "christianity.pentecostal",
    "Adventiste": "christianity.adventist",
    "Méthodiste": "christianity.methodist",
    "Episcopale": "christianity.anglican.episcopal",
    "Témoin de Jéhovah": "christianity.witnesses",

    # ---------------------------------------------------------------- Protestant, unplaced
    "Autre protestant": "christianity.protestant",

    # ---------------------------------------------------------------- Afro-diasporic
    "Vaudou": "afrodiasporic.vodou",

    # ---------------------------------------------------------------- other religions
    "Musulman": "islam",
    "Autre": "other.ht",

    # ---------------------------------------------------------------- no religion
    "Aucune": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1). Every row here is `modelled` rather than `derived` — §7b's
# test, applied in countries.py::_ht_counts — and the roll-up is about where a DERIVED row
# was actually counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for an ECVMAS answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
