"""
EBRD Life in Transition Survey III `q922` (Belarus, 2015-16) -> religiondots taxonomy.

No Belarusian census asks about religion: the 2019 form (Belstat Form 2N) runs 25 questions with
nationality at 11 and language at 12-13 and no religion item (`sources.md`
§scout-2026-09-14-taiwan-belarus-gabon). So this card is the only count of Belarusians by
religion that reaches the map, and every row is `modelled`. Named for the field year, per the
registry convention. `sources/by.md` has the build.

    73.55%  ORTHODOX CHRISTIAN                        -> christianity.orthodox
    14.59%  ATHEISTIC / AGNOSTIC / NONE               -> unaffiliated
     9.21%  CATHOLIC                                  -> christianity.catholic
     1.02%  BUDDHIST                                  -> buddhism
     0.76%  OTHER                                     -> other.by
     0.50%  OTHER CHRISTIAN, INCLUDING PROTESTANT     -> christianity
     0.27%  JEWISH                                    -> judaism
     0.11%  MUSLIM                                    -> islam
            Refusal (4 respondents)                   -> EXCLUDED

Shares are weighted, refusals left out. Where each is placed is a `sources/by.py` decision and
changes what none of these answers means: Catholics follow the Catholic Church's own diocesan
counts, and the other seven are spread at the national rate inside each oblast's remainder.
"""

EXCLUDED = {
    "Refusal": "four respondents in 1,504 (two in Mogilev, one in Grodno, one in Minsk city)",
}

REVIEW = {
    "ORTHODOX CHRISTIAN":
        "-> christianity.orthodox, the bare node. Almost all of it is the Belarusian Orthodox "
        "Church, the Moscow Patriarchate's exarchate, but the card asks one Orthodox question "
        "and ask 004's rule is to record what the source can distinguish. Old Believers, who "
        "have their own box on the EVS 2017 Belarus card, can only be in here or in OTHER.",
    "ATHEISTIC / AGNOSTIC / NONE":
        "-> unaffiliated and not `secular`, following `kg2016.py` and `cy2021.py`: LiTS merges "
        "three answers into one box, so a stated non-religious position cannot be separated "
        "from having no religion.",
    "CATHOLIC":
        "-> christianity.catholic, the bare node. Belarus has a small Greek Catholic church "
        "beside the Latin-rite majority (the EVS 2017 Belarus card lists it separately), and "
        "this box cannot tell them apart. 137 respondents.",
    "BUDDHIST":
        "-> buddhism, and **the national figure is a ceiling rather than a measurement**. "
        "Thirteen respondents, 1.02% weighted, which would be about 92,000 people; Belarus's "
        "registered Buddhist communities are a handful and the EVS 2017 Belarus card has no "
        "Buddhist box at all. The answers are spread over ten PSUs (one each in five Minsk "
        "city PSUs, three in one Postavy PSU), so this looks like the keying slip `kg2016.py` found in "
        "Kyrgyzstan, where code 2 sits next to code 1, none. It passes the split-half "
        "(p=0.037, one pass in seven tests) and is overridden to the national rate in "
        "`sources/by.py::OVERRIDE`. Kept rather than excluded to match Kyrgyzstan's treatment "
        "of the same card; excluding it would move 1% of the country into the other answers "
        "in proportion.",
    "OTHER CHRISTIAN, INCLUDING PROTESTANT":
        "-> christianity, the bare family node, following `kg2016.py` and `bs2022.py`. Seven "
        "respondents. Belarus's Pentecostals, Baptists and Adventists are real and registered "
        "in numbers, but the box also has to hold Jehovah's Witnesses and anyone else outside "
        "the named churches, and its own label says other Christian.",
    "OTHER":
        "-> other.by. Seventeen respondents, fourteen of them in two PSUs (8 in Molodechno, 6 "
        "in Ostrovets, each one interviewer), so it is mostly two interviewers rather than a "
        "group. `taxonomy/branches.py` carries the node note.",
    "JEWISH":
        "-> judaism. Five respondents. The 2019 census counted 13,705 Jews by nationality "
        "(0.15%); the drawn 0.27% is a survey figure on five people and is within its error.",
    "MUSLIM":
        "-> islam, the bare node. One respondent, in Minsk city, so no test is possible and it "
        "is spread at the national rate (`sources/by.py::UNTESTED`). About 10,000 people, "
        "against the 8,445 Tatars the 2019 census counts by nationality, who are the "
        "community this stands for.",
}

MAP = {
    "ORTHODOX CHRISTIAN": "christianity.orthodox",
    "CATHOLIC": "christianity.catholic",
    "OTHER CHRISTIAN, INCLUDING PROTESTANT": "christianity",
    "ATHEISTIC / AGNOSTIC / NONE": "unaffiliated",
    "BUDDHIST": "buddhism",
    "JEWISH": "judaism",
    "MUSLIM": "islam",
    "OTHER": "other.by",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, not `derived`, so there is no counted
# column to roll a row back to. Same as kg2016.py.


def resolve(category):
    """religiondots branch for a LiTS III answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
