"""
ONE ENHOGAR-MICS6 2019 `HC1A` (Dominican Republic) -> religiondots taxonomy.

**Six answers, and it is a national card rather than a continental one.** MICS6's household
questionnaire asks *"¿A cuál religión pertenece el jefe o la jefa del hogar?"* and offers the
four bodies a Dominican interviewer would expect to hear named, plus a specify box and a
no-religion box. Named for the survey year, per the registry convention.

    52.71%  CATÓLICA                        -> christianity.catholic.latin
    22.39%  EVANGÉLICA                      -> christianity.evangelical
    20.58%  NINGUNA RELIGIÓN                -> unaffiliated
     1.97%  ADVENTISTA                      -> christianity.adventist
     1.36%  OTRA RELIGIÓN (Especifique)     -> other.do
     0.99%  TESTIGO DE JEHOVÁ               -> christianity.witnesses

Those are the shares as drawn, off `data/normalized/do.csv`. Every row is `modelled` in §7's
sense: a survey share against the 2022 census population. **And every row is the religion of
the household's HEAD**, applied to everyone living in that household, which is what MICS's own
tabulations do and is a modelling step this file cannot undo. `countries.py`'s `note_public`
says so to the reader.

## THE NO-RELIGION CELL IS `unaffiliated` AND NOT `unchurched`, WHICH IS THE OPPOSITE OF THE
## THREE LAPOP COUNTRIES

`gt2023.py`, `sv2023.py` and `ec2023.py` all route their no-religion answer to `unchurched`,
because the AmericasBarometer prints the gloss *cree en un Ser Superior pero no pertenece a
ninguna religión* on the card and offers `Agnóstico o ateo` as a separate box. **MICS6 prints
`NINGUNA RELIGIÓN` and nothing else.** There is no belief qualifier and no atheist box, so
what this cell holds is a report of no religion, which is `unaffiliated`'s own definition and
what Canada, Australia and the UK measure. Routing it to `unchurched` would assert a claim
about belief that this questionnaire never asked about.

The two instruments come out close on the total: 20.58% here against LAPOP's 18.02% for the
two cells added together (§9cf's cross-check), so the difference between the cards is in how
they split it and not in how big it is.

## `EVANGÉLICA` IS ONE BOX FOR EVERY NON-CATHOLIC CHURCH THAT IS NOT ONE OF THE OTHER THREE

In Dominican usage *evangélico* is the general word for a Protestant, and the card gives
Adventists and Jehovah's Witnesses their own answers beside it, which is exactly the
condition `christianity.evangelical`'s note in `branches.py` describes: the node is for
sources that name Evangelical BESIDE something else rather than as a synonym for Protestant.
There is no separate `Protestante` box here, unlike LAPOP's card, so historic Protestant
bodies — the Dominican Evangelical Church, the Episcopalians, the Free Methodists and the
Anglophone *cocolo* congregations of the eastern sugar towns — are inside this cell and
cannot be separated from the Pentecostal majority.

That matters for one province in particular. **La Romana is 49.3% `EVANGÉLICA` against 25.3%
Catholic**, the highest this cell reaches anywhere in the country, and it is where the British
West Indian migration to the sugar mills landed. It is the largest cell in four provinces in
all: La Romana, Samaná, San Pedro de Macorís and La Altagracia. This file cannot say how much of
that is Pentecostal and how much is a Methodist or Anglican inheritance.
"""

EXCLUDED = {}

REVIEW = {
    "EVANGÉLICA":
        "-> christianity.evangelical rather than christianity.protestant. The card names "
        "Adventists and Witnesses separately and has no `Protestante` box at all, so this "
        "one answer carries both the Pentecostal churches and the historic Protestant ones. "
        "`christianity.evangelical`'s own note in branches.py says the node is for sources "
        "that name Evangelical BESIDE Protestant; here it is named INSTEAD of it, which is "
        "the closer of the two available readings but not the same thing. The alternative "
        "was `christianity.protestant`, the unspecified node, and it was rejected because "
        "*evangélico* is what Dominican respondents say and spec §2.4 keeps the source's own "
        "word where it can. 22.39% of the country and 49.3% of La Romana.",
    "NINGUNA RELIGIÓN":
        "-> unaffiliated, and NOT `unchurched`, which is where the three LAPOP countries in "
        "this region send their no-religion answer. The difference is the card: LAPOP prints "
        "`Ninguna (cree en un Ser Superior…)` and offers atheism separately, MICS6 prints "
        "`NINGUNA RELIGIÓN` and asks nothing about belief. 20.58% of the country, and its "
        "geography is measured rather than spread: **42.9% of Pedernales and 42.7% of "
        "Baoruco against 4.9% of La Vega**, split-half +0.91.",
    "ADVENTISTA":
        "-> christianity.adventist, the bare family node rather than a Dominican child of "
        "it. The Seventh-day Adventist Church is the body the card means and effectively the "
        "only Adventist body in the country, but the questionnaire says `ADVENTISTA` and not "
        "`Adventista del Séptimo Día`, so the more specific node would be this file's "
        "inference and not the source's word. 1.97%, three times Mexico's share and a little "
        "above Peru's 1.52%, though well below the Anglophone Caribbean, where Grenada is "
        "13.3% and Jamaica 12.3%. Measured (+0.55): 5.1% of Hato Mayor and 4.1% of San "
        "Cristóbal against 0.15% of San José de Ocoa.",
    "TESTIGO DE JEHOVÁ":
        "-> christianity.witnesses. 0.99%, and **the one category here NOT drawn where the "
        "survey found it**: its split-half rank is -0.02 against a bar of +0.35, so every "
        "province gets the national share. sources/do.py has the argument, including why "
        "its chi-square of 4e-27 is not a licence to override. Nobody is deleted; only the "
        "claim to know where they are is withdrawn.",
    "OTRA RELIGIÓN (Especifique)":
        "-> other.do. The specify text is not in the public microdata, so unlike Uruguay's "
        "`Otra` this cell cannot be read at all. branches.py's note has what its geography "
        "suggests and is careful to say that Dominican Vodú is not in it: Vodú is practised "
        "in the same south-western provinces and its practitioners answer Catholic, which is "
        "the Haitian and Cuban pattern.",
}

MAP = {
    # ---------------------------------------------------------------- Catholic
    "CATÓLICA": "christianity.catholic.latin",

    # ---------------------------------------------------------------- everything else Christian
    "EVANGÉLICA": "christianity.evangelical",
    "ADVENTISTA": "christianity.adventist",
    "TESTIGO DE JEHOVÁ": "christianity.witnesses",

    # ---------------------------------------------------------------- no religion
    "NINGUNA RELIGIÓN": "unaffiliated",

    # ---------------------------------------------------------------- other religions
    "OTRA RELIGIÓN (Especifique)": "other.do",
}

# No COLUMNS dict (spec §7a-i-1). Every row here is `modelled` rather than `derived` — §7b's
# test, applied in countries.py::_do_counts — and the roll-up is about where a DERIVED row
# was actually counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for an ENHOGAR-MICS6 answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
