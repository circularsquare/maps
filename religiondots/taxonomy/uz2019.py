"""
Central Asia Barometer `Religion_M` (Uzbekistan, waves 1-6, 2017-2019) -> religiondots taxonomy.

**Nine answers on the card, seven drawn.** No Uzbek census has ever asked about religion, the
2026 one included (`sources/uz.md` §1, against the printed questionnaire), so there is no
counted list to set this against. Named for the last field year, per the registry convention.

    Muslim                                -> islam
    Christian                             -> christianity
    A believer of no particular faith     -> unchurched
    A non-believer                        -> secular
    Other (vol.)                          -> other.uz
    A believer of another faith           -> other.uz
    Jewish                                -> judaism
    Don't Know (vol.)                     -> EXCLUDED
    Refused (vol.)                        -> EXCLUDED

Every row is `modelled` in §7's sense: a pooled survey share applied to the National
Statistics Committee's 1 January 2026 regional populations. Which answers carry their own
regional geography is a `sources/uz.py` decision and changes what none of them means.

## ISLAM IS THE BARE FAMILY NODE

The card has one Muslim box. Uzbekistan's Muslims are overwhelmingly Hanafi Sunni, with a small
Shia community in Bukhara and Samarkand, and none of that is in this source. Wave 1's phone
re-contact asked Sunni / Shia / Ismaili and 300 of its 387 Uzbek answers were *don't know*
(§11ao), which is Tajikistan's finding again (§11ak) and not a school to draw from.

## CHRISTIANITY IS THE BARE FAMILY NODE TOO

One Christian box. It is mostly Russian Orthodoxy: 208 of the 251 Christians in the pool give
Russian, Armenian, Ukrainian, Belarusian, German or Korean ethnicity. But Armenians are
Armenian Apostolic, Germans Lutheran or Catholic, and many Koreans Protestant, and the
pressured groups in Uzbekistan (Protestant converts from Muslim families, Jehovah's Witnesses)
answer the same box. §2.6: a church is never assigned from outside the source.

## THE TWO NO-RELIGION ANSWERS STAY APART

The card separates *a believer of no particular faith* from *a non-believer*, which a merged
`atheist/agnostic/none` box (LiTS, `kg2016.py`) cannot. The first is `unchurched`, as
`al2023.py`, `cr2023.py` and `ru2012.py` file a believer without an affiliation; the second is
`secular`, as `kz2021.py` files Kazakhstan's census `Неверующие` and `ru2012.py` Arena's
"I do not believe in God".
"""

# Excluded from the partition rather than drawn; `sources/uz.py` runs §3.5's lean check.
EXCLUDED = {
    "Don't Know (vol.)": "volunteered; about 0.6% of the weighted pool",
    "Refused (vol.)": "volunteered; about 0.2% of the weighted pool",
}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, for about 95% of the country. One Muslim box and no "
        "school; see the module docstring for why wave 1's phone follow-up is not a source "
        "for one.",
    "Christian":
        "-> christianity, the bare family node. Mostly Russian Orthodox by the ethnicity of "
        "the people who answer it, but the box also holds Armenian Apostolic, Lutheran, "
        "Catholic and Protestant answers and the survey cannot separate them.",
    "A believer of no particular faith":
        "-> unchurched, following al2023.py, cr2023.py and ru2012.py. A stated belief with no "
        "affiliation, which is what that node is for.",
    "A non-believer":
        "-> secular, following kz2021.py (`Неверующие`) and ru2012.py. A face-to-face "
        "interview in a country where 95% answer Muslim is not where irreligion shows up, "
        "so the figure is a floor.",
    "A believer of another faith":
        "-> other.uz. One respondent in 9,000, in wave 3.",
    "Other (vol.)":
        "-> other.uz, with the card's `another faith`. Volunteered by the respondent rather "
        "than read from the card; eighteen respondents, thirteen of them in wave 4.",
    "Jewish":
        "-> judaism. Four respondents, spread at the national rate because four cannot show "
        "a geography. About 22,000 people at 1:1,000, which is a size the survey cannot "
        "vouch for.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "Jewish": "judaism",
    "A believer of no particular faith": "unchurched",
    "A non-believer": "secular",
    "A believer of another faith": "other.uz",
    "Other (vol.)": "other.uz",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and the roll-up is about where a
# DERIVED row was counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a Central Asia Barometer answer, or None if off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
