"""LISGIS Liberia 2022 census religion -> religiondots taxonomy.

Five categories, at county. They are the census's own five and its own wording, from Table
A13 of the *Final Results*; the county geography is fitted from the pooled Afrobarometer
(`sources/lr.py`), and the mapping is unaffected by that — a Liberian Christian is on the
same node whichever county the fit puts them in.

**THE INTERESTING MAPPING QUESTION IS THE ONE THIS SOURCE CANNOT ASK.** Afrobarometer's own
card names Pentecostal, Methodist, Lutheran, Baptist, Roman Catholic, Presbyterian, Anglican,
Adventist and a dozen more, and Liberia's pooled respondents fill every one of them — so a
denominational tree for Liberia is *visible* in the raw data and is still not drawable, for
the reason `sources/lr.py`'s docstring gives at length: the share who name a denomination
rather than answering `Christian only` runs from 23.1% to 72.0% between rounds, so any
denominational split would be a measurement of the fieldwork. `source_category` carries the
census's wording per §2.4, and the raw answers are printed on every build, so a later source
that settles the split can open this cell rather than re-deriving it.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Christian":
        "-> christianity, the bare branch. 4,458,286 people, 84.9%, and the whole of "
        "Liberian Christianity in one cell. No Liberian census has ever split it: the 2022 "
        "questionnaire defines the box as `all Christian denomination churches` and stops "
        "there, and the 2008 census and the 2012 analytical monographs do the same. **What "
        "is inside it is not a mystery, only unmeasured** — the Liberian Methodists out of "
        "the 19th-century settler churches, the Lutherans of the Muhlenberg mission on the "
        "St Paul, a large Pentecostal and independent sector, the Baptists, the Catholics "
        "of the Monrovia archdiocese, and the Episcopalians who ran the country's oldest "
        "schools. The pooled Afrobarometer can see all of them and cannot level them "
        "(module docstring). Left at the parent rather than guessed at.",
    "Muslim":
        "-> islam, with no branch, because neither the census nor the survey gives one. "
        "628,859 people, 12.0%, and **by a wide margin the most concentrated distribution "
        "in the country**: Grand Cape Mount is 78.4% Muslim and Bomi 56.5%, against 12.0% "
        "nationally, with Gbarpolu 24.1% and Lofa 23.3% behind them and Rivercess at zero. "
        "That is the Mandingo and Vai north-west, on the Sierra Leone and Guinea borders, "
        "and it is one contiguous block rather than a scatter. Overwhelmingly Sunni Maliki; "
        "`islam.sunni` would be an inference rather than a reading, and the survey's own "
        "Sunni/Shia/Ismaili boxes take 25 of 7,163 respondents between them, which is a "
        "count of who understood the question rather than a split.",
    "Traditional African Religion":
        "-> indigenous.african. 25,445 people, 0.48%. **Read it as a floor**, for the "
        "reason sources.md §11b gives for the whole continent and mw2018.py repeats: the "
        "census offers this box as a peer of `Christian`, so it cannot see anyone who is "
        "both, and in Liberia a great many people are — the Poro and Sande societies run "
        "through the Mande and Mel-speaking interior and their members answer Christian or "
        "Muslim when asked for a religion. The 2008 census counted 20,134 of them, 0.58% of "
        "a smaller Liberia, so the two enumerations agree. "
        "**Do not use the 2022 report's Figure 3.5.2 for the trend.** It plots traditional "
        "religion at 18% in 1984, 2.3% in 2008 and 3.2% in 2022, and the last two contradict "
        "the same report's own Figure 3.5.1 and Table A13 (0.5% and 25,445 people) and the "
        "2008 row UNSD holds (0.58%). Two of its three columns are wrong, so the 1984 figure "
        "on it is not usable either, and no figure from it is quoted anywhere here. "
        "**Its drawn geography is flat**, at the national rate in every county, because "
        "0.48% is under the 1% floor §11ad set for placing a category on survey evidence.",
    "Other religion":
        "-> other.lr. 3,431 people, 0.065%, the smallest residual of its kind on this map. "
        "The census report names Eckankar, Baha'i and Shintoism as its examples. Drawn flat "
        "for the same reason as the traditionalists; see the node's note.",
    "No religion":
        "-> unaffiliated. 134,166 people, 2.56%. One cell, so nothing goes to `secular`, "
        "which needs a separately-counted atheist or humanist answer — the same call "
        "ke2019.py, mw2018.py and hr2021.py make. **Drawn flat**, and that is a measured "
        "result rather than a size problem: at 1.24% of the pooled survey it clears the "
        "eligibility floor and is tested, and its split-half rank correlation across the "
        "fifteen counties is +0.114 against a +0.524 bar (§14.16). The survey cannot show "
        "that Liberian irreligion has a geography, so the map does not give it one.",
}

MAP = {
    "Christian": "christianity",
    "Muslim": "islam",
    "Traditional African Religion": "indigenous.african",
    "Other religion": "other.lr",
    "No religion": "unaffiliated",
}

# What this source actually MEASURED, for spec §7a-i-1's roll-up: every row is measured at the
# node it is drawn on, because the census's five categories are the five nodes. Nothing here
# is inferred downward, so `inferred dots: not shown` removes nothing from Liberia.
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
