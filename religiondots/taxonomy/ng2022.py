"""Afrobarometer Nigeria religion -> religiondots taxonomy.

Seven categories, at state. Three of them are the Afrobarometer's own answer boxes verbatim;
`Christian` and `Muslim` are that card's umbrella answers with their denominational children
folded back in; `Catholic` and `Anglican` (since 2026-10-03) are carved back out of `Christian`,
Catholics from the three rounds whose Catholic share matches the NDHS, Anglicans levelled by the
Global Flourishing Study. `sources/ng.py`'s
docstring has the argument and `sources/ng.md` §5 the record.

**THE INTERESTING MAPPING QUESTIONS ARE BOTH ONES THIS SOURCE CANNOT ANSWER**, and Nigeria is
the country on this map where that costs the most. The card names about twenty Christian
denominations and, on the Muslim side, Sunni, Shia, Ismaili, Izala and three Sufi
brotherhoods, and Nigerians fill nearly all of them. A Nigerian denominational tree is
therefore *visible* in the raw data and is still not drawable, because the share who name a
denomination rather than answering `Christian only` swings 27.9 points between rounds with no
trend. `source_category` carries the grouped wording per §2.4 and `sources/ng.py` prints the
raw crosstab on every build, so a later source that settles the split can open these cells
rather than re-deriving them.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Christian":
        "-> christianity, the bare branch: every Christian but Catholics and Anglicans (37.0% "
        "of Nigeria as drawn), and still the largest single cell on this map drawn at a parent "
        "node. **No Nigerian census since 1963 has asked about religion at all.** Inside it: "
        "the Redeemed Christian Church of God and the rest of the Pentecostal sector, the "
        "Methodists and Baptists out of the 19th-century missions, the Aladura churches (Christ "
        "Apostolic, Cherubim and Seraphim, the Celestial Church of Christ), ECWA in the Middle "
        "Belt, and the Church of Christ in Nations on the Plateau. **Pentecostals are the big "
        "one and are not drawn**: the Global Flourishing Study (2023) puts them at 39.9% of "
        "Nigerian Christians, and it and the Afrobarometer do not rank the states alike once "
        "Catholics are taken out (+0.20, p 0.15), so nothing checks where they are. The GFS "
        "codes 87% of Plateau's and 72% of Bauchi's Christians `Orthodox`, most likely COCIN and "
        "ECWA members, so its small boxes are not used either.",
    "Anglican":
        "-> christianity.anglican. 4.2% of Nigeria as drawn, 8.1% of Christians. Level from the "
        "Global Flourishing Study 2023 (`REL3_Y1`): 10.0% of non-Catholic Christians, applied to "
        "the non-Catholic Christians left once the NDHS-levelled Catholics are out. The "
        "Afrobarometer's named Anglicans in rounds 4-6 are 8.9% of non-Catholic Christians, so the "
        "GFS level sits just above that floor and only 1.9% of `Christian only` is given to it. "
        "Each state's share averages the two surveys by respondents. The two surveys rank 27 "
        "states alike at +0.550 (p 0.0014). Anambra 22.1%, Enugu 15.6%, Imo 12.8%, Ekiti and "
        "Delta 12%; near zero in the far north.",
    "Catholic":
        "-> christianity.catholic. 10.2% as drawn, 19.9% of Christians. Each state's Catholics "
        "as a share of its Christians in rounds 4-6 (2008-2015), the rounds whose 19-21% matches "
        "the four NDHS reports' 19.4-23.9%; rounds 7-9 give 4-13% as `Christian only` swells. "
        "Eight far-northern states with under 30 Christian respondents in those rounds take the "
        "national 19.7%. Split-half on Christians, rounds 4-6, 27 states: +0.753. The Church's "
        "diocesan statistics by cathedral state rank the states alike (+0.833, 31 states). "
        "**Short where Christians name no church**: in Lagos, Rivers, Ogun and Oyo 58-73% of "
        "Christians answered `Christian only` in those rounds, and Lagos is drawn 5.8% Catholic "
        "where its archdiocese claims 25%. Zero in Kwara and Osun, where none of 40 and 84 "
        "Christians named it.",
    "Muslim":
        "-> islam, with no branch. **Nigeria is overwhelmingly Sunni Maliki with a large "
        "Sufi presence, and `islam.sunni` would still be an inference rather than a "
        "reading.** The survey's own Sunni, Shia, Ismaili, Izala, Tijaniyya, Qadiriyya and "
        "Mouridiyya boxes take 719 of 11,909 respondents between them, and `Izala` is on the "
        "card in rounds 4, 5 and 7 and absent from the other three, so a pooled share of any "
        "of them measures the questionnaire. "
        "**The Shia cell is a §14 refusal as well as an arithmetic one.** `Shia` and `Shia "
        "only` take 58 respondents across fourteen years; the Islamic Movement in Nigeria "
        "has been proscribed since 2019 and its members killed in numbers, and §14.4's rule "
        "2 draws a persecuted group no finer than the state publishes, which here is not at "
        "all. Neither reason needs the other.",
    "Traditional/ethnic religion":
        "-> indigenous.african. 0.32% of the pooled survey, and **read it as a floor** for "
        "the reason sources.md §11b gives for the whole continent and lr2022.py and mw2018.py "
        "repeat: the card offers this as a peer of `Christian` and `Muslim`, so it cannot "
        "see anyone who is both, and in Nigeria a great many people are. Ifá and òrìṣà "
        "practice in the Yoruba south-west, the Ọdịnala of the south-east and the masquerade "
        "societies of the middle belt all run through populations that answer Christian or "
        "Muslim when asked for a religion. **Its drawn geography is flat**, at the national "
        "rate in every state, because 0.32% is under the 1% floor §11ad set for placing a "
        "category on survey evidence. That is a real loss here: the traditionalists this "
        "does see are not evenly spread, and a flat rate puts as many in Sokoto as in Osun.",
    "Other":
        "-> other.ng. 0.20% of the pooled survey, 26 respondents over six rounds. The "
        "Afrobarometer offers no specify-text with it, so unlike `other.uy` there is nothing "
        "to read; what it holds is anyone the card's thirty-odd named answers did not fit, "
        "which in Nigeria would include the Baha'i, the Grail Movement, Eckankar and the "
        "Hindu and Buddhist communities of Lagos. Drawn flat, for the same reason as the "
        "traditionalists.",
    "None":
        "-> unaffiliated. 0.17% of the pooled survey, and the smallest irreligion share of "
        "any country on this map. One node, so nothing goes to `secular`: the survey does "
        "offer separate `Atheist` and `Agnostic` boxes and **two people in fourteen years "
        "chose them**, which cannot support a node of its own. ke2019.py, mw2018.py, "
        "hr2021.py and lr2022.py make the same call. **Drawn flat**, and here that is a size "
        "result rather than a measured one: at 0.17% it is under the eligibility floor and "
        "is not tested at all, so the map is not claiming that Nigerian irreligion has no "
        "geography, only that this survey cannot see one.",
}

MAP = {
    "Christian": "christianity",
    "Catholic": "christianity.catholic",
    "Anglican": "christianity.anglican",
    "Muslim": "islam",
    "Traditional/ethnic religion": "indigenous.african",
    "Other": "other.ng",
    "None": "unaffiliated",
}

# What this source actually MEASURED, for spec §7a-i-1's roll-up: every row is measured at the
# node it is drawn on, because the five grouped categories are the five nodes. Nothing here is
# inferred downward, so `inferred dots: not shown` removes nothing from Nigeria. (Six since
# 2026-10-03; `Catholic` is measured at christianity.catholic, a share the survey's respondents gave.)
COLUMNS = {v: v for v in MAP.values()}


def _key(cat):
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}
_FOLDED_EXCLUDED = {_key(k) for k in EXCLUDED}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = str(cat)
    if c in EXCLUDED or _key(c) in _FOLDED_EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))
