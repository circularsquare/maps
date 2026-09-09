"""Afrobarometer Nigeria religion -> religiondots taxonomy.

Five categories, at state. Three of them are the Afrobarometer's own answer boxes verbatim;
`Christian` and `Muslim` are that card's umbrella answers with their denominational children
folded back in. `sources/ng.py`'s docstring has the argument for the folding and
`sources/ng.md` the acquisition record.

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
        "-> christianity, the bare branch, and it is the largest single cell on this map "
        "drawn at a parent node. **No Nigerian census has ever split it, because no Nigerian "
        "census since 1963 has asked about religion at all.** What is inside it is not a "
        "mystery, only unmeasured: the Catholic Church of the south-east, the Anglican "
        "Church of Nigeria and the Methodists out of the 19th-century missions, the Aladura "
        "churches (Christ Apostolic, Cherubim and Seraphim, the Celestial Church of Christ), "
        "the Redeemed Christian Church of God and the rest of the Pentecostal sector that is "
        "now the largest Christian bloc in the country, and the Church of Christ in Nations "
        "on the Plateau. The pooled Afrobarometer can see all of them and cannot level them: "
        "Roman Catholic runs 11.4% of respondents in round 4 and 2.2% in round 8, which is "
        "the fieldwork rather than the country. Left at the parent rather than guessed at.",
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
    "Muslim": "islam",
    "Traditional/ethnic religion": "indigenous.african",
    "Other": "other.ng",
    "None": "unaffiliated",
}

# What this source actually MEASURED, for spec §7a-i-1's roll-up: every row is measured at the
# node it is drawn on, because the five grouped categories are the five nodes. Nothing here is
# inferred downward, so `inferred dots: not shown` removes nothing from Nigeria.
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
