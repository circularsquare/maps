"""ESS `rlgblg` x `rlgdnlv` -> religiondots taxonomy. Latvia's citizens and recognised non-citizens.

ESS asks `rlgblg` (do you consider yourself as belonging to any particular religion or
denomination; `scrlgblg` in round 10's self-completion file) and then the denomination. Latvia's
country card `rlgdnlv` is in rounds 4, 9, 10 and 11 and is finer than the harmonised `rlgdnm`; its
labels arrive in English. sources/lv.py proves it nests.

    Catholic                                -> christianity.catholic.latin
    Lutheran                                -> christianity.lutheran
    Russian or Greek Orthodox               -> christianity.orthodox.canonical
    Other Orthodox Denominations            -> christianity.orthodox.oldbeliever
    Baptist                                 -> christianity.baptist
    Other Protestant Denominations          -> christianity.protestant
    Christian, denomination not specified   -> christianity
    Other Christian Denominations           -> christianity
    Jewish                                  -> judaism
    Islam                                   -> islam
    Eastern religions                       -> other.lv
    Other Non-Christian Religions           -> other.lv
    (rlgblg = No)                           -> unaffiliated
    (refusal / don't know / no answer)      -> excluded, spec §3.5
"""

CATHOLIC = "Catholic"
LUTHERAN = "Lutheran"
ORTHODOX = "Russian or Greek Orthodox"
OTHER_ORTHODOX = "Other Orthodox Denominations"
BAPTIST = "Baptist"
OTHER_PROTESTANT = "Other Protestant Denominations"
CHRISTIAN_UNSPECIFIED = "Christian, denomination not specified"
OTHER_CHRISTIAN = "Other Christian Denominations"
JEWISH = "Jewish"
ISLAM = "Islam"
EASTERN = "Eastern religions"
OTHER_NONCHRISTIAN = "Other Non-Christian Religions"
NO_RELIGION = "No religion"

# The answers the pooled rounds produce. sources/lv.py asserts the citizen pool against this set in
# both directions (fi.py §8).
SOURCE = {CATHOLIC, LUTHERAN, ORTHODOX, OTHER_ORTHODOX, BAPTIST, OTHER_PROTESTANT,
          CHRISTIAN_UNSPECIFIED, OTHER_CHRISTIAN, JEWISH, ISLAM, EASTERN, OTHER_NONCHRISTIAN,
          NO_RELIGION}

# No answer is split across nodes in Latvia; sources/lv.py still asks, so a later split lands in
# the normalised file as its own source categories rather than inside resolve().
SPLIT_NOTE = ""


def split(cat):
    return {cat: 1.0}


EXCLUDED = {
    "__refused__":
        "Everyone whose belonging answer is Refusal, Don't know or No answer, and everyone who "
        "answered Yes and then declined to name a denomination. spec §3.5 marks a refusal rather "
        "than filling it; sources/lv.py prints the share and `gap` states it.",
}

REVIEW = {
    CATHOLIC:
        "-> christianity.catholic.latin. The Roman Catholic Church in Latvia is four Latin dioceses "
        "(Riga, Jelgava, Liepaja, Rezekne-Aglona) and reported 304,759 members for 2025 to the "
        "Ministry of Justice. Latgale's Catholicism is the Polish-Lithuanian inheritance of Inflanty; "
        "no Eastern Catholic church is registered at any size.",
    LUTHERAN:
        "-> christianity.lutheran. The answer names the church: the Evangelical Lutheran Church of "
        "Latvia (700,000 members reported for 2025), the German Evangelical Lutheran Church in Latvia, "
        "the Augsburg Confession congregations and the Latvian Evangelical Lutheran Church Abroad's "
        "parishes in Latvia. Self-identification, not membership: the survey's share is well under "
        "the roll's, and sources/lv.md prints both.",
    ORTHODOX:
        "-> christianity.orthodox.canonical, ee2021.py's and lt2021.py's call for the same church "
        "family next door. In Latvia this is the Latvian Orthodox Church, under the Moscow Patriarchate "
        "until the Saeima declared it autocephalous in September 2022 (250,000 members reported for "
        "2025), and a small autonomous church under Constantinople (240). The card's wording, "
        "`Russian or Greek`, names no Latvian body; nobody here is Greek.",
    OTHER_ORTHODOX:
        "-> christianity.orthodox.oldbeliever, lt2021.py's and pl2021.py's node. The card offers "
        "`Russian or Greek Orthodox` and `Other Orthodox Denominations`, and the other Orthodox body "
        "of any size in Latvia is the Old Believers' Pomor Church, priestless, whose Grebenshchikov "
        "congregation in Riga is one of the largest Old Believer communities in the world, with the "
        "rest concentrated in Latgale. The survey agrees: the answer is 6.6-10.5% of Latgale's "
        "respondents in rounds 4, 9 and 11 and at most 4.7% anywhere else in any round. Two things are "
        "named rather than fitted. The Pomor Church's reported roll (2,039 for 2025) is far below "
        "the survey's share, and the ministry's 2013 report says why: the church counts only "
        "members with a vote at congregational meetings (2,355 that year) and reported 41,877 "
        "people attending its services, 36,712 in the Pomor Church and 5,165 in the Rezekne "
        "cemetery congregation. And in round 11 (2023-24), among all respondents, the answer's share of all Orthodox jumps from "
        "about 13% to 32%, which is plausibly respondents of the Latvian Orthodox Church answering "
        "`other` after its 2022 separation from Moscow; sources/lv.md has the figures.",
    BAPTIST:
        "-> christianity.baptist. The answer names the body: the Union of Baptist Churches in Latvia "
        "(6,194 members reported for 2025) and a few autonomous Baptist congregations.",
    OTHER_PROTESTANT:
        "-> christianity.protestant, the root of the Protestant branches, because the answer names "
        "none: Pentecostals, Adventists, Methodists, the New Generation church and the Reformed "
        "congregations are all on the ministry's list and all plausibly in this box.",
    CHRISTIAN_UNSPECIFIED:
        "-> christianity, the root. The respondent said Christian and named no church.",
    OTHER_CHRISTIAN:
        "-> christianity, the root, following se2024.py, no2024.py and dk2024.py: the answer names no "
        "body, and in Latvia it holds Jehovah's Witnesses (2,104 reported for 2025), Latter-day "
        "Saints, New Apostolic Christians and anyone who did not see their church on the card.",
    ISLAM:
        "-> islam, the root, ee2021.py's call and not dk2024.py's `islam.sunni`. Latvia's Muslims are "
        "a few thousand people: Volga Tatars and Bashkirs (Sunni), Azerbaijanis (mostly Shia) and "
        "Central Asians, eight autonomous congregations with 175 members reported for 2025. With no "
        "majority school large enough to assert, the root is the honest node.",
    JEWISH:
        "-> judaism, the root. The Riga Jewish religious community and ten other congregations; the "
        "answer counts no movement.",
    EASTERN:
        "-> other.lv with the next answer, for gr2024.py's, se2024.py's and dk2024.py's reason: the box "
        "counts several traditions as one, and the ministry's list has Buddhist, Hindu, Vaishnava and "
        "Sukyo Mahikari congregations, so choosing one would invent a fact about a respondent.",
    OTHER_NONCHRISTIAN:
        "-> other.lv, with whoever the card did not name. In Latvia that certainly includes the "
        "Dievturi, the Latvian folk-religion revival registered as a religious union since 1990 "
        "(602 members reported for 2025), and the Baha'i.",
    NO_RELIGION:
        "-> unaffiliated. Everyone who answered NO to the belonging question. It includes a large "
        "share of the Lutheran and Catholic churches' reported members, which is the gap between the "
        "roll and the survey. ESS offers no atheist answer, so nothing in Latvia reaches `secular`.",
}

MAP = {
    CATHOLIC: "christianity.catholic.latin",
    LUTHERAN: "christianity.lutheran",
    ORTHODOX: "christianity.orthodox.canonical",
    OTHER_ORTHODOX: "christianity.orthodox.oldbeliever",
    BAPTIST: "christianity.baptist",
    OTHER_PROTESTANT: "christianity.protestant",
    CHRISTIAN_UNSPECIFIED: "christianity",
    OTHER_CHRISTIAN: "christianity",
    JEWISH: "judaism",
    ISLAM: "islam",
    EASTERN: "other.lv",
    OTHER_NONCHRISTIAN: "other.lv",
    NO_RELIGION: "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
