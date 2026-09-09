"""
EBRD Life in Transition Survey III `q922` (Kyrgyz Republic, 2015-16) -> religiondots taxonomy.

**Nine answers, and the card is the whole of what Kyrgyzstan has.** No Kyrgyz census has ever
asked about religion — `sources/kg.md` establishes that against the 2022 questionnaire rather
than against the publications — so there is no counted list to compare this one with and no
finer instrument to fall back on. Named for the field year, per the registry convention.

    89.23%  MUSLIM                                    -> islam
     6.76%  ORTHODOX CHRISTIAN                        -> christianity.orthodox
     1.94%  ATHEISTIC / AGNOSTIC / NONE               -> unaffiliated
     0.91%  BUDDHIST                                  -> buddhism
     0.78%  OTHER CHRISTIAN, INCLUDING PROTESTANT     -> christianity
     0.19%  OTHER                                     -> other.kg
     0.14%  JEWISH                                    -> judaism
     0.03%  CATHOLIC                                  -> christianity.catholic
     0.02%  Refusal                                   -> EXCLUDED

Every row is `modelled` in §7's sense: a survey share against the National Statistical
Committee's 1 January 2026 resident population. Two of the nine carry their own oblast
geography and six are spread at the national rate inside each oblast's residual, but that is
a `sources/` decision about placement and changes what none of these answers MEANS.

## THE ISLAM CELL IS BARE, AND IN THIS COUNTRY THAT COSTS SOMETHING REAL

`islam` with no school under it. The card offers one Muslim box and asks nothing further, so
there is no evidence here for `islam.sunni.hanafi`, which is what Kyrgyzstan's Muslims almost
entirely are: the Muftiate is Hanafi, the madrasa curriculum is Hanafi, and the country has no
Shia population of any size. **Drawing the school anyway would be putting 6.6 million people
on a node this source cannot see**, and §14.3 forbids exactly that. Türkiye (`tr2021.py`) is
the country on this map that has the school because its source asked; Kyrgyzstan does not.

## `unaffiliated` AND NOT `secular`, WHICH IS A MERGED BOX AND NOT A CHOICE

LiTS prints `ATHEISTIC / AGNOSTIC / NONE` as one answer, so `secular` cannot be separated out
of it. `cy2021.py` met the same shape in CYSTAT's `Atheist/No Religion` and made the same
call, for the same reason: the no-religion reading is the larger part of a merged box and
folding the merged box into `secular` would assert a stated position that most of the people
in it did not state. 1.94%, which is low for a post-Soviet country and is a floor rather than
a measurement: this is a face-to-face interview in a country where 89% answer Muslim.

## `christianity` BARE FOR THE OTHER-CHRISTIAN BOX

`OTHER CHRISTIAN, INCLUDING PROTESTANT` is the Christian residual after Orthodox and Catholic
have their own boxes, and `bs2022.py` mapped the identically-shaped `OTHER CHRISTIAN
DENOMINATION` to bare `christianity` for the identical reason. Naming it `christianity.
protestant` would be the tempting call, because Kyrgyzstan's non-Orthodox Christians are
mostly Baptists, Pentecostals, Lutherans and Adventists — but the box also has to hold the
Jehovah's Witnesses, who are not Protestant on this tree, and the label's own head is *other
Christian*. Fourteen respondents; the honest node is the one that claims only what the answer
says.
"""

# One respondent refused the question. Excluded rather than drawn, and `sources/kg.py` runs
# §3.5's lean check on it, which reports that a single respondent in a single oblast supports
# no measurement of a lean in either direction.
EXCLUDED = {
    "Refusal": "one respondent in 1,500, in Chui oblast",
}

REVIEW = {
    "MUSLIM":
        "-> islam, the bare family node, for 89.23% of the country. The card has one Muslim "
        "box. Kyrgyzstan is Hanafi Sunni in every published description of it, and none of "
        "that is in this source, so the school is not drawn. See the module docstring.",
    "ATHEISTIC / AGNOSTIC / NONE":
        "-> unaffiliated and NOT `secular`. LiTS merges three answers into one box, so the "
        "stated-position reading cannot be separated from the no-religion one; `cy2021.py` "
        "is the precedent and has the argument. 1.94%, which is a floor: a face-to-face "
        "interview in a country that is 89% Muslim is not where Soviet-era irreligion "
        "shows up, and nothing published says how much larger the real figure is.",
    "BUDDHIST":
        "-> buddhism, and **the national figure is a ceiling rather than a measurement**. "
        "Sixteen respondents tick it, 0.91%, and six of them are in Osh oblast and four in "
        "Batken, the two most rural and most uniformly Muslim oblasts in the country; "
        "Bishkek, which has whatever Buddhist community Kyrgyzstan has, returns one. Its "
        "split-half median is -0.15 with 68% of halves negative, the worst of any category "
        "here, so it is spread at the national rate and not where the survey put it. "
        "`sources/kg.py` reads the pattern as a keying artefact and says so; the people are "
        "still drawn, because dropping them would move them somewhere else.",
    "OTHER CHRISTIAN, INCLUDING PROTESTANT":
        "-> christianity, the bare family node, following `bs2022.py`. Fourteen "
        "respondents, 0.78%. Not `christianity.protestant`: the box is a residual whose own "
        "label says *other Christian*, and it has to hold the Jehovah's Witnesses, who sit "
        "outside Protestantism on this tree. Kyrgyzstan's registered Protestant bodies are "
        "real and visible — Baptists since the 19th-century German and Ukrainian "
        "settlements, and a Pentecostal presence built since 1991 — and this cell cannot "
        "tell them from each other.",
    "CATHOLIC":
        "-> christianity.catholic, the bare node rather than `.latin`. One respondent, in "
        "Bishkek. Kyrgyzstan's Catholics are the Latin-rite Apostolic Administration at "
        "Jalal-Abad, so `.latin` would probably be right and one respondent is not evidence "
        "for it. Drawn because the partition is closed and dropping the row would move that "
        "person somewhere else, not because the survey can see Kyrgyz Catholicism.",
    "JEWISH":
        "-> judaism. Two respondents, one in Bishkek and one in Issyk-Kul. Kyrgyzstan's "
        "Jewish "
        "community is largely an Ashkenazi wartime evacuation that has since emigrated, and "
        "the survey cannot see its size; this draws as a §4.3 presence ring at every dot "
        "value this map offers.",
    "OTHER":
        "-> other.kg. Four respondents. `taxonomy/branches.py` carries the node note, and "
        "the thing worth reading there is what is NOT in this box: Kyrgyz shamanic and "
        "mazar practice has no answer on this card and the people who take part in it "
        "answer Muslim.",
}

MAP = {
    # ---------------------------------------------------------------- Islam
    "MUSLIM": "islam",

    # ---------------------------------------------------------------- Christianity
    "ORTHODOX CHRISTIAN": "christianity.orthodox",
    "CATHOLIC": "christianity.catholic",
    "OTHER CHRISTIAN, INCLUDING PROTESTANT": "christianity",

    # ---------------------------------------------------------------- everything else
    "BUDDHIST": "buddhism",
    "JEWISH": "judaism",
    "ATHEISTIC / AGNOSTIC / NONE": "unaffiliated",
    "OTHER": "other.kg",
}

# No COLUMNS dict (spec §7a-i-1). Every row here is `modelled` rather than `derived` — §7b's
# test, applied in countries.py::_kg_counts — and the roll-up is about where a DERIVED row was
# actually counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a LiTS III answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
