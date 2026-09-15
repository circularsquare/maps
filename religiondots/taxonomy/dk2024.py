"""ESS `rlgblg` x `rlgdnm` -> religiondots taxonomy. Denmark's Danish-citizen half.

ESS asks `rlgblg` (do you consider yourself as belonging to any particular religion or
denomination) and then the denomination. Denmark's country card `rlgdndk` exists in round 5 only
and is the harmonised `rlgdnm` in Danish, code for code, so the four rounds (5, 6, 7, 9) are
pooled on `rlgdnm`, whose labels arrive in English. sources/dk.py proves the equivalence.

    Protestant                        -> christianity.lutheran
    Roman Catholic                    -> christianity.catholic.latin
    Eastern Orthodox                  -> christianity.orthodox
    Other Christian denomination      -> christianity
    Jewish                            -> judaism
    Islam                             -> islam.sunni
    Eastern religions                 -> other.dk
    Other Non-Christian religions     -> other.dk
    (rlgblg = No)                     -> unaffiliated
    (refusal / don't know / no answer)-> excluded, spec §3.5

THE ORTHODOX ANSWER IS NOT SPLIT, and that is sources.md §9cz's check run and coming back the usual
way. Aarhus University's Center for Samtidsreligion, which Danmarks Statistik itself points to for
membership of faith communities, lists the approved communities with the members each reports
(`religion-i-danmark/rel-aarbog09/statistik/alle`, the 2009 column):

    Serbian Orthodox Church in Denmark          7,000   Eastern
    Russian Orthodox congregation, Copenhagen   1,000   Eastern
    Russian Orthodox congregation, Hobro          100   Eastern
    Romanian Orthodox congregation                500   Eastern
    Macedonian Orthodox Church                    500   Eastern
    Coptic Orthodox Church                        250   Oriental
    Armenian Apostolic Church                   (none printed)
    Assyrian Church of the East                   270   Church of the East

9,100 of 9,620, **94.6%, Eastern**. Norway's Eritrean churches and Sweden's Syriac dioceses have no
counterpart of that size here, so the shortcut Austria and the UK take is right for Denmark, on an
old and self-reported list. The Ethiopian Orthodox church (approved 2011) and any Syriac or
Eritrean congregation outside the approval system are not on it, which is the direction the error
runs, and it is a few hundred people among 14 respondents in four rounds.
"""

LUTHERAN = "Protestant"
CATHOLIC = "Roman Catholic"
ORTHODOX = "Eastern Orthodox"
OTHER_CHRISTIAN = "Other Christian denomination"
JEWISH = "Jewish"
ISLAM = "Islam"
EASTERN = "Eastern religions"
OTHER_NONCHRISTIAN = "Other Non-Christian religions"
NO_RELIGION = "No religion"

# The nine answers the pooled rounds produce. sources/dk.py asserts the pool against this set in
# both directions (fi.py §8).
SOURCE = {LUTHERAN, CATHOLIC, ORTHODOX, OTHER_CHRISTIAN, JEWISH, ISLAM, EASTERN,
          OTHER_NONCHRISTIAN, NO_RELIGION}

# No answer is split across nodes in Denmark; sources/dk.py still asks, so a later split lands in
# the normalised file as its own source categories rather than inside resolve().
SPLIT_NOTE = ""


def split(cat):
    return {cat: 1.0}


EXCLUDED = {
    "__refused__":
        "Everyone whose `rlgblg` is Refusal, Don't know or No answer, and everyone who answered "
        "Yes and then declined to name a denomination. spec §3.5 marks a refusal rather than "
        "filling it; sources/dk.py prints the share and `gap` states it.",
}

REVIEW = {
    LUTHERAN:
        "-> christianity.lutheran, not christianity.protestant, although the answer is the "
        "harmonised `Protestant` and the Danish card's `Protestantisk` names no church. The "
        "Church of Denmark is the Evangelical Lutheran state church and held about three "
        "quarters of the population on its roll across the survey years (DST KM6, printed by "
        "sources/dk.py), while the largest Protestant free churches on Center for Samtidsreligion's "
        "1 January 2009 list are the Baptists 5,260, the Pentecostals 5,158, the Apostolic Church "
        "3,000, the Adventists 2,537, the Mission Covenant 2,200 and the Methodists 2,006, about "
        "20,000 together, under 0.4% of Denmark. So nearly everyone in this cell is Lutheran, "
        "which is the call fi2024.py and se2024.py make for their state churches, and the free "
        "churches are inside it unseparated. **Self-identification, not membership**: the survey's "
        "share is well under the roll's, and sources/dk.md prints both.",
    ORTHODOX:
        "-> christianity.orthodox, unsplit. The module docstring has the 2009 approved-community "
        "counts: 94.6% of the listed Orthodox members are in Eastern churches (Serbian 7,000). "
        "Sweden split its cell and Norway split its cell; Denmark's list gives no reason to.",
    OTHER_CHRISTIAN:
        "-> christianity, the root, following se2024.py, no2024.py and be2024.py: the answer names "
        "no body, and in Denmark it holds Jehovah's Witnesses, Latter-day Saints and anyone who "
        "did not see their church in `Protestant`, which are different branches.",
    CATHOLIC:
        "-> christianity.catholic.latin. Denmark is one Latin diocese, Copenhagen; 37,123 members "
        "on the 2009 approved-community list, largely Polish, Vietnamese, Filipino, Tamil and "
        "Croatian in origin. Eastern-rite Catholics are not separable in this half.",
    ISLAM:
        "-> islam.sunni, se2024.py's and no2024.py's call. Denmark's Muslims are mostly Turkish, "
        "Pakistani, Somali, Bosnian, Palestinian and Arab Sunni in origin; the Shia minority "
        "(Iraqi, Iranian, Afghan, Lebanese) and the Alevis among Turkish Danes are not separable "
        "in this half, and the Shia dots come from the foreign half only.",
    JEWISH:
        "-> judaism, the root. Det Mosaiske Troessamfund in Copenhagen is nearly all of organised "
        "Danish Jewry; the answer counts no movement.",
    EASTERN:
        "-> other.dk with the next answer, for gr2024.py's, se2024.py's and no2024.py's reason: "
        "the box counts several traditions as one, and choosing Buddhism over Hinduism would "
        "invent a fact about a respondent.",
    OTHER_NONCHRISTIAN:
        "-> other.dk, with whoever the card did not name: the Baha'i, the Norse pagan associations, "
        "and any religion neither Christian nor Eastern.",
    NO_RELIGION:
        "-> unaffiliated. Everyone who answered NO to `rlgblg`. This includes a large share of "
        "the Church of Denmark's own members, which is the whole gap between the roll and the "
        "survey. ESS offers no atheist or humanist answer, so nothing in Denmark reaches `secular`.",
}

MAP = {
    LUTHERAN: "christianity.lutheran",
    CATHOLIC: "christianity.catholic.latin",
    ORTHODOX: "christianity.orthodox",
    OTHER_CHRISTIAN: "christianity",
    JEWISH: "judaism",
    ISLAM: "islam.sunni",
    EASTERN: "other.dk",
    OTHER_NONCHRISTIAN: "other.dk",
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
