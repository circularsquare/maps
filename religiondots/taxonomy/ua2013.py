"""ESS `rlgblg` x `rlgdnm` -> religiondots taxonomy. Ukraine, everyone (sources/ua.py).

The pooled rounds 2-6 carry only the harmonised card in all five, so the pattern is on `rlgdnm`.
Its `Roman Catholic` holds the Greek Catholics, and sources/ua.py splits it on the Ukrainian cards of
rounds 4-6 and 11 into the two answers below.

    Eastern Orthodox                         -> christianity.orthodox
    Greek-Catholic church                    -> christianity.catholic.eastern
    (Other) Roman Catholic denominations     -> christianity.catholic.latin
    Protestant                               -> christianity.protestant
    Other Christian denomination             -> christianity
    Jewish                                   -> judaism
    Islam                                    -> islam.sunni
    Eastern religions                        -> other.ua
    Other Non-Christian religions            -> other.ua
    (rlgblg = No)                            -> unaffiliated
    (refusal / don't know / no answer)       -> excluded, spec §3.5
"""

ORTHODOX = "Eastern Orthodox"
CATHOLIC = "Roman Catholic"
GREEK_CATHOLIC = "Greek-Catholic church"
LATIN_CATHOLIC = "(Other) Roman Catholic denominations"
PROTESTANT = "Protestant"
OTHER_CHRISTIAN = "Other Christian denomination"
JEWISH = "Jewish"
ISLAM = "Islam"
EASTERN = "Eastern religions"
OTHER_NONCHRISTIAN = "Other Non-Christian religions"
NO_RELIGION = "No religion"

# The answers the pool may produce, before the Catholic split. sources/ua.py asserts against it.
POOL_SOURCE = {ORTHODOX, CATHOLIC, PROTESTANT, OTHER_CHRISTIAN, JEWISH, ISLAM, EASTERN,
               OTHER_NONCHRISTIAN, NO_RELIGION}
SOURCE = (POOL_SOURCE - {CATHOLIC}) | {GREEK_CATHOLIC, LATIN_CATHOLIC}

SPLIT_NOTE = ""


def split(cat):
    return {cat: 1.0}


EXCLUDED = {
    "__refused__":
        "Everyone whose belonging answer is Refusal, Don't know or No answer, and everyone who "
        "answered Yes and then named no denomination. spec §3.5 marks a refusal rather than filling "
        "it; sources/ua.py prints the share and `gap` states it.",
}

REVIEW = {
    ORTHODOX:
        "-> christianity.orthodox, the branch, and not a jurisdiction. The harmonised card has one "
        "Eastern Orthodox answer in all five pooled rounds. Ukraine's own card splits it, and splits it "
        "differently on either side of the gap: rounds 4-6 (2009-2013) offer the Moscow Patriarchate, "
        "the Kyiv Patriarchate, the Autocephalous Church and other Orthodox; round 11 (2023-24) offers "
        "the Orthodox Church of Ukraine, the Moscow Patriarchate and Orthodox of no patriarchate. The "
        "Kyiv Patriarchate and the Autocephalous Church merged into the OCU in December 2018, and "
        "self-identification with the Moscow Patriarchate fell from about a fifth of adults in 2013 to "
        "5% in 2025 (Razumkov), so a pooled jurisdiction share describes no year. Drawing either "
        "card's jurisdictions by oblast is a §14 question in wartime and is Anita's; the ask is filed "
        "and sources/ua.py prints both splits. Canonical status is also contested (the OCU is "
        "recognised by Constantinople and three other churches), which is why the branch and not "
        "`christianity.orthodox.canonical`.",
    GREEK_CATHOLIC:
        "-> christianity.catholic.eastern. The Ukrainian Greek Catholic Church, in communion with Rome, "
        "which is the node's own example. It is also where the Ruthenian Greek Catholic eparchy of "
        "Mukachevo in Zakarpattia belongs, which the card does not separate.",
    LATIN_CATHOLIC:
        "-> christianity.catholic.latin. The Roman Catholic Church's Latin dioceses (Kyiv-Zhytomyr, "
        "Lviv, Kamianets-Podilskyi and others), historically Polish and in Zakarpattia Hungarian. "
        "Round 11's card calls it `Roman Catholic Church`; sources/ua.py counts both labels here.",
    PROTESTANT:
        "-> christianity.protestant, the root of the Protestant branches. The card names no church, and "
        "Ukraine's largest Protestant bodies by DESS's registered communities are Baptist, Pentecostal, "
        "Seventh-day Adventist and charismatic, with the Reformed Church of Zakarpattia's Hungarians and "
        "the Lutherans small. Choosing one would invent a fact.",
    OTHER_CHRISTIAN:
        "-> christianity, the root, following se2024.py, no2024.py, dk2024.py and lv2024.py: Jehovah's "
        "Witnesses, Latter-day Saints and anyone who did not see their church on the card.",
    JEWISH:
        "-> judaism, the root; the answer names no movement.",
    ISLAM:
        "-> islam.sunni, dk2024.py's call and not lv2024.py's root. Before 2014 Ukraine's Muslims were "
        "overwhelmingly Crimean Tatars (248,193 in the 2001 census) and Volga Tatars, both Sunni; the "
        "Shia minority (Azerbaijanis, 45,176 in 2001) is too small to leave the answer on the root. The "
        "survey puts the answer in Crimea.",
    EASTERN:
        "-> other.ua with the next answer, for gr2024.py's, se2024.py's and lv2024.py's reason: the box "
        "counts several traditions as one.",
    OTHER_NONCHRISTIAN:
        "-> other.ua, with whoever the card did not name: the RUNVira and Ridnovira native-faith "
        "communities, Krishna consciousness, the Baha'i.",
    NO_RELIGION:
        "-> unaffiliated. Everyone who answered NO to the belonging question. ESS offers no atheist "
        "answer, so nothing in Ukraine reaches `secular`. Razumkov's own question finds about half as "
        "many (15.5% in 2025) because its list offers `just Christian`, which ESS does not.",
}

MAP = {
    ORTHODOX: "christianity.orthodox",
    GREEK_CATHOLIC: "christianity.catholic.eastern",
    LATIN_CATHOLIC: "christianity.catholic.latin",
    PROTESTANT: "christianity.protestant",
    OTHER_CHRISTIAN: "christianity",
    JEWISH: "judaism",
    ISLAM: "islam.sunni",
    EASTERN: "other.ua",
    OTHER_NONCHRISTIAN: "other.ua",
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
