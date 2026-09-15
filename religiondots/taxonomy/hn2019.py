"""
INE ENDESA-MICS 2019 `HC1` (Honduras) -> religiondots taxonomy.

**Seven answers and a non-response, on a Honduran card.** The household questionnaire asks the
religion of the household head, `HC1`, and names five bodies before `OTRO` and `NINGUNA`.
Named for the survey year, per the registry convention. Shares as drawn, off
`data/normalized/hn.csv` (of people drawn):

    41.87%  EVANGELICA              -> christianity.evangelical
    41.12%  CATOLICA                -> christianity.catholic.latin
    14.87%  NINGUNA RELIGION        -> unaffiliated
     0.74%  TESTIGOS DE JEHOVA      -> christianity.witnesses
     0.67%  ADVENTISTA              -> christianity.adventist
     0.47%  MORMON                  -> christianity.latterday
     0.25%  OTRO(ESPECIFIQUE)       -> other.hn
            NO RESPONDE             excluded, 0.47% of people

Every row is `modelled` (§7): a survey share on INE's 2024 projection, and every row is the
head's religion applied to the whole household. This is the Dominican Republic's card with a
Mormon box added, and the calls below follow `taxonomy/do2019.py` wherever the two agree.
"""

EXCLUDED = {
    "NO RESPONDE":
        "The head's religion was not given: 91 households of 20,669, 0.47% of people once "
        "weighted and laid on the projection. Not a religion; in `gap=`.",
}

REVIEW = {
    "EVANGELICA":
        "-> christianity.evangelical rather than christianity.protestant, the same call as "
        "taxonomy/do2019.py on the same MICS card. Adventists, Witnesses and Latter-day Saints "
        "have their own answers and there is no `Protestante` box, so the historic Protestant "
        "churches are inside this answer with the Pentecostal ones and cannot be separated. "
        "41.87% of Honduras, and 60.3% of Gracias a Dios, where the card also has no Moravian "
        "box. Carries its own geography (split-half median +0.886).",
    "NINGUNA RELIGION":
        "-> unaffiliated, not `unchurched`, for taxonomy/do2019.py's reason: the card prints a "
        "bare `NINGUNA RELIGION` with no belief qualifier and no atheist box. 14.87%, from 24.5% "
        "of the Bay Islands to 2.6% of Gracias a Dios, split-half +0.841. LAPOP, whose card "
        "splits believers without a church from atheists, gives 12.2% for the two together on "
        "the same departments.",
    "TESTIGOS DE JEHOVA":
        "-> christianity.witnesses. 0.74%, and drawn on its own department shares, which the "
        "Dominican Witnesses were not: the median over 400 cluster halvings is +0.522 against "
        "a bar of +0.475, the narrowest pass here (63.5% of halvings clear the bar), with a "
        "chi-square of 2e-5 and no cluster holding more than 3% of the answer. Highest in "
        "Cortés at 1.4%.",
    "ADVENTISTA":
        "-> christianity.adventist, the family node, as in taxonomy/do2019.py: the card says "
        "`ADVENTISTA`, not `Adventista del Séptimo Día`. 0.67%. FAILS the rank test (+0.358) "
        "and is still drawn at 8.28% in Islas de la Bahía, which is the highest Adventist "
        "department in both halves of all 400 halvings (75 households in 27 of its 42 "
        "clusters); every other department gets the Adventist share of the other seventeen, "
        "0.61%. sources/hn.py's STANDOUTS carries the rule and asserts it.",
    "MORMON":
        "-> christianity.latterday. 0.47%, fails the rank test (+0.263) and no department "
        "stands apart (the Bay Islands top both halves in 23% of halvings), so it is drawn at "
        "its national share in every department.",
    "OTRO(ESPECIFIQUE)":
        "-> other.hn, a new per-country node like other.do. The specify text is not in INE's "
        "files. 0.25%, drawn flat: as measured it is 2.29% of Gracias a Dios (every one of the "
        "17 households has a Misquito head) and 1.87% of the Bay Islands, but the two are the "
        "top pair in only 85% of halvings. branches.py's note has the detail.",
}

MAP = {
    "CATOLICA": "christianity.catholic.latin",
    "EVANGELICA": "christianity.evangelical",
    "TESTIGOS DE JEHOVA": "christianity.witnesses",
    "MORMON": "christianity.latterday",
    "ADVENTISTA": "christianity.adventist",
    "OTRO(ESPECIFIQUE)": "other.hn",
    "NINGUNA RELIGION": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): nothing here is `derived`, every row is `modelled`.


def resolve(category):
    """religiondots branch for an ENDESA-MICS answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
