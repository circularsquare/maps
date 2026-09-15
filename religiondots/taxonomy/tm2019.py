"""
Central Asia Barometer `Religion_M` (Turkmenistan, waves 4-6, 2018-2019) -> religiondots taxonomy.

**Six answers came back, all drawn.** The card is Uzbekistan's (`uz2019.py`): Christian / Muslim /
Jewish / a believer of another faith / a believer of no particular faith / a non-believer /
other (volunteered) / refused / don't know. Nobody in the three Turkmen waves answered Jewish,
refused or said they did not know. Turkmenistan's 2022 census did not ask religion and no
Turkmen census since independence has published one, so there is no counted list to set this
against. Named for the last field year, per the registry convention.

    Muslim                                -> islam
    Christian                             -> christianity
    A non-believer                        -> secular
    A believer of no particular faith     -> unchurched
    A believer of another faith           -> other.tm
    Other (vol.)                          -> other.tm
    Jewish                                -> judaism     (on the card, nobody answered it)
    Don't Know (vol.)                     -> EXCLUDED    (on the card, nobody answered it)
    Refused (vol.)                        -> EXCLUDED    (on the card, nobody answered it)

Every row is `modelled` in §7's sense: survey answers within each census nationality, applied to
the 2022 census's nationality counts by velayat (`sources/tm.py`).

## ISLAM AND CHRISTIANITY ARE THE BARE FAMILY NODES

One Muslim box and one Christian box, as in Uzbekistan. Turkmen Muslims are Hanafi Sunni with
no school asked here. The Christians are mostly Russian Orthodox by who answers the box (236 of
the 245 give Russian, Armenian or Ukrainian nationality), but Armenians are Armenian Apostolic
and the box also holds the Protestant and Jehovah's Witness converts whom the state pressures.
§2.6: a church is never assigned from outside the source.

## THE TWO NO-RELIGION ANSWERS STAY APART

As `uz2019.py`: *a believer of no particular faith* is `unchurched`, *a non-believer* is
`secular`.
"""

EXCLUDED = {
    "Don't Know (vol.)": "volunteered; nobody in waves 4-6",
    "Refused (vol.)": "volunteered; nobody in waves 4-6",
}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, for about 98% of the country as drawn. One Muslim box "
        "and no school.",
    "Christian":
        "-> christianity, the bare family node. Mostly Russian Orthodox by the nationality of "
        "the people who answer it, but the box also holds Armenian Apostolic answers and the "
        "Protestant and Jehovah's Witness converts, and the survey cannot separate them.",
    "A non-believer":
        "-> secular, following uz2019.py and kz2021.py. Seven respondents in 4,500.",
    "A believer of no particular faith":
        "-> unchurched, following uz2019.py. One respondent, wave 4.",
    "A believer of another faith":
        "-> other.tm. One respondent, wave 4.",
    "Other (vol.)":
        "-> other.tm, with the card's `another faith`. One respondent, wave 4.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "Jewish": "judaism",
    "A believer of no particular faith": "unchurched",
    "A non-believer": "secular",
    "A believer of another faith": "other.tm",
    "Other (vol.)": "other.tm",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and the roll-up is about where a
# DERIVED row was counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a Central Asia Barometer answer, or None if off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
