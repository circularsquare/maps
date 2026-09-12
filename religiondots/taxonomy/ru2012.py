"""Sreda "Arena" 2012 answer options -> religiondots taxonomy.

Eighteen offered answers to one question, at federal subject. Seventeen of them are
religious positions and one is a refusal.

WHAT IS UNUSUAL ABOUT THIS MAPPING. Every other source on this map names INSTITUTIONS and
leaves the tree to work out what they are — 372 American religious bodies, 129 Filipino
ones, 216 Polish ones. Arena names POSITIONS, in the first person, and several of them are
defined by what they are not: "Orthodox, but not the Russian Orthodox Church, and not an Old
Believer"; "Islam, but neither Sunni nor Shia"; "Christianity, but not Orthodox, Catholic or
Protestant". Those are precise and they map cleanly, which is a nicer problem than the usual
one. The cost is on the other side: nobody names a church, so 58 million Russian Orthodox
arrive as one node and stay there.

THE THREE ANSWERS THAT DECIDE THIS COUNTRY'S MAP are the ones about belief rather than
belonging, and the tree already had all three, from three different countries:

    41.14%  ROC                             -> christianity.orthodox.canonical
    25.16%  believes in God, no religion    -> unchurched      (Czechia, 2026-09-02)
    13.02%  does not believe in God         -> secular         (Canada's atheists)

`unchurched` was built for 960,201 Czechs. Russia puts about 35.6 million people on it —
thirty-seven times as many, and more than the node's other five countries combined. It is
now the second largest answer in the country and the largest instance of that category
anywhere on the map.

EXCLUDED holds categories deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Difficult to answer":
        "5.46% of respondents, about 7.8 million people at the 2021 population. A "
        "non-response, not an answer, and spec §3.5 says it is marked rather than filled — "
        "so Russia is drawn at 94.5% of the population of the subjects it covers, and "
        "note_public says so. Not `unaffiliated` (that is a report of no religion, and "
        "Arena asks it separately as 'I do not believe in God'), not `unrecorded` (that is "
        "for a register that never asked; this one asked and got no answer).",
}

REVIEW = {
    "I am Orthodox, and belong to the Russian Orthodox Church":
        "-> christianity.orthodox.canonical. Unambiguous — the answer names the church — "
        "and worth stating anyway because it is 58.6 million people, the largest single "
        "node any one country puts on this map apart from the American residual and "
        "Germany's `unrecorded`. The ROC is canonical Eastern Orthodoxy and this is the "
        "node hr2021.py and ro2021.py use for the same thing.",
    "I am Orthodox, but do not belong to the Russian Orthodox Church, and I am not an Old "
    "Believer":
        "-> christianity.orthodox.other, which is where a body with no branch to belong to "
        "goes. The answer is defined by exclusion and the tree has a node for exactly that "
        "shape. 2.1M people; in Russia it covers the Russian Orthodox Church Outside "
        "Russia, the True Orthodox and catacomb communities, and the Georgian, Armenian "
        "(which is Oriental, not Eastern) and other diaspora jurisdictions, none of which "
        "Arena separates. NOT `christianity.orthodox.oldcalendarist`, which would assert a "
        "specific reason for the separation that this answer does not give.",
    "I am Orthodox, and I am an Old Believer":
        "-> christianity.orthodox.oldbeliever, a node that has existed since the first US "
        "source and never carried more than a few thousand people. 464,461 here. "
        "**Read the NATIONAL figure and distrust the map of it.** 0.32% of a 56,900-person "
        "survey is about 180 respondents spread over 79 subjects, so the per-subject "
        "pattern is mostly sampling noise — Arena's top subject is Smolensk at 1.5%, and "
        "the historic Old Believer concentrations are elsewhere entirely. The national "
        "total is the part worth reading.",
    "I profess Islam, but am neither Sunni nor Shia":
        "-> islam, the parent, and this is the largest Muslim answer in Russia at 4.66% "
        "against Sunni's 1.66%. It is a real answer rather than a residual: Arena offered "
        "Sunni and Shia as separate options and most Russian Muslims declined both, which "
        "is consistent with how Islam is described in the Volga republics in particular. "
        "§6.6 — a branch that carries dots is a category.",
    "I profess Christianity, but do not consider myself Orthodox, Catholic, nor Protestant":
        "-> christianity, the root. The same call mk2021.py makes for `Христијани`, "
        "au2021.py for 'Christianity, nfd' and ca2021.py for 'Christian'. NOT "
        "`christianity.protestant`, which is the node for an answer that names no body but "
        "does say Protestant; this answer explicitly rules Protestantism out. NOT "
        "`christianity.other`, which branches.py reserves for bodies with no branch rather "
        "than for people who name none. 5.8M people, the fifth largest answer.",
    "I practice traditional religion of my ancestors, worship the gods and the forces of "
    "nature":
        "-> indigenous.northeurasian, a node added for this source. See branches.py for "
        "what is and is not in it — the short version is that the geography carries the "
        "meaning the single category cannot, and that a small unseparable part of it is "
        "Rodnovery, which is a modern reconstruction rather than a continuous tradition.",
    "I do not believe in God":
        "-> secular, not `unaffiliated`. branches.py draws the line at whether a POSITION "
        "is stated: `unaffiliated` is a report of no religion, `secular` is a stated "
        "non-theistic position, and 'I do not believe in God' is plainly the second. It is "
        "also the same wording ca2021.py files under `secular` for Canada's atheists. "
        "**Nothing in Russia goes to `unaffiliated` at all**, which is unusual — Arena "
        "never offers 'no religion' as an option. The people who would answer that way in "
        "another country are split here between this node and `unchurched`, and the split "
        "is 13.0% to 25.2%.",
    "I follow Eastern religions and spiritual practices (Hinduism, Krishnaism, other)":
        "-> other.ru, reluctantly and for hr2021.py's reason: the tree has no node for "
        "'some Eastern religion, unspecified' and picking one would invent a fact. See the "
        "note on other.ru in branches.py — Buddhism is asked separately and taken by "
        "659,032 people, so this is the remainder after it.",
    "I profess the Protestant (Lutheran, Baptist, Evangelical, Anglican)":
        "-> christianity.protestant, the node for a Protestant answer that names no body. "
        "The four in Arena's parenthesis are examples offered to the respondent, not "
        "categories counted — there is one cell for all of them, 305,000 people, and "
        "nothing recoverable below it. Pentecostalism is asked separately and is its own "
        "row, which is why it is not in this list.",
    "Other":
        "-> other.ru. 0.61%, Arena's own residual. Per source, per spec §3.11.",
}

MAP = {
    # --- Orthodoxy, split three ways, which is the thing this source does better than
    #     any census on the map.
    "I am Orthodox, and belong to the Russian Orthodox Church":
        "christianity.orthodox.canonical",
    "I am Orthodox, but do not belong to the Russian Orthodox Church, and I am not an Old "
    "Believer":
        "christianity.orthodox.other",
    "I am Orthodox, and I am an Old Believer":
        "christianity.orthodox.oldbeliever",

    # --- the rest of Christianity
    "I profess Christianity, but do not consider myself Orthodox, Catholic, nor Protestant":
        "christianity",
    "I profess Catholicism": "christianity.catholic",
    "I profess the Protestant (Lutheran, Baptist, Evangelical, Anglican)":
        "christianity.protestant",
    "I profess Pentecostalism": "christianity.pentecostal",

    # --- Islam, and the reason islam.sunni and islam.shia exist at all
    "I profess Islam, but am neither Sunni nor Shia": "islam",
    "I profess Sunni Islam": "islam.sunni",
    "I profess Shia Islam": "islam.shia",

    # --- the other families
    "I profess Buddhism": "buddhism",
    "I profess Judaism": "judaism",
    "I practice traditional religion of my ancestors, worship the gods and the forces of "
    "nature": "indigenous.northeurasian",
    "I follow Eastern religions and spiritual practices (Hinduism, Krishnaism, other)":
        "other.ru",
    "Other": "other.ru",

    # --- belief without belonging, and unbelief. Between them, 38% of Russia.
    "I believe in God (in a higher power), but do not profess a particular religion":
        "unchurched",
    "I do not believe in God": "secular",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
