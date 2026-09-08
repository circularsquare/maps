"""Census 2021 (Bulgaria) religion classification -> religiondots taxonomy.

**Eleven drawn categories on 265 municipalities: three measured and eight derived.** NSI's
municipal sheet offers `Християнско`, `Мюсюлманско`, `Юдейско`, `Друго`, `Нямам` and three
ways of not answering. The first two are undivided and are **replaced** by `bg_split.py`
before they get here.

    measured, straight off the 2021 workbook at the obshtina
     4.7%  Нямам                    -> unaffiliated
     0.10% Друго                    -> other.bg
     0.03% Юдейско                  -> judaism

    derived — 2011 oblast composition, raked to NSI's published 2021 national breakdown
    62.8%  Източноправославно       -> christianity.orthodox.canonical    4,091,780
     1.07% Протестантско            -> christianity.protestant               69,852
     0.59% Католическо              -> christianity.catholic                 38,709
     0.21% Друго християнско        -> christianity.other                    13,927
     0.08% Арменско апостолическо   -> christianity.oriental                  5,002

    derived — 2011 oblast composition only, no 2021 national target exists
     9.3%  Мюсюлманско сунитско     -> islam.sunni                          604,875
     0.45% Мюсюлманско шиитско      -> islam.shia                            29,470
     0.07% Мюсюлманско неуточнено   -> islam                                  4,363

(shares of the 6,519,789 enumerated, not of those who answered; NSI's own headline
percentages, Christian 71.5% and Muslim 10.8%, are shares of the 5,903,108 who were asked.)

**WHY THE TWO COLUMNS ARE SPLIT AT ALL, AND WHY THAT REVERSES THIS FILE'S FIRST VERSION.**
Built undivided, Bulgaria drew as one flat Christianity colour between Romania, Serbia,
Greece and North Macedonia, all of which sit on `christianity.orthodox.canonical`. Anita,
2026-09-08: *"bulgaria looks kinda out of place as it's the only one in the area where we
dont have christianity breakdown to orthodox and not. so it displays as generic light
yellow"*, and then *"yes 2011 oblast composition is ideal."* The change of colour at four
borders was an artefact of what two statistical offices published, which is exactly the kind
of false edge spec §6.6 warns a coarse mapping produces.

**The first version cited §14.4 rule 1 to refuse this, and that reading was too narrow.**
Rule 1 forbids estimating a magnitude a source does not publish. Here every magnitude is
NSI's own: the national denomination totals are the 2021 census's, each obshtina's Christian
and Muslim totals are the 2021 census's, and only the *distribution of the former across the
latter* is modelled, from the same state's 2011 census. That is §14.10's amendment, which
permits the map to run the model itself when the magnitude is the host state's, the
coefficients are documented and the output is checked against something independent. All
three hold, and `bg_split.py` carries the checks.

**EVERY DERIVED ROW ROLLS BACK TO WHAT WAS COUNTED.** `COLUMNS` below names the 2021 column
each row came out of, so `inferred dots: not shown` redraws `christianity` and `islam` at the
obshtina, which is precisely the measured table. Nothing is asserted that cannot be undone in
the interface.

**THE LIMIT, STATED PLAINLY.** The composition is uniform inside an oblast, because 28 units
is the finest geography any Bulgarian census publishes it at. The clearest casualty is the
Catholics: most of the 38,709 are Banat Bulgarians in Rakovski, and this spreads them across
Plovdiv oblast. The Muslim split is weaker still, resting on the 2011 shape with no 2021
national total to rake to.

**THREE COLUMNS ARE NOT DRAWN AND NONE OF THEM HIDES A RELIGION.** This is the opposite of
Slovakia's `ostatné`, where a residual demonstrably contained named churches and was drawn on
§6.12. Here `Друго` is its own column, so the two declining answers and the register
population contain nothing recoverable: they are 20.7% of the country and 20.7% of nothing.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Общо":
        "the unit's own population total, not a category.",
    "Християнско":
        "NOT off the tree — REPLACED. bg_split.py divides these 4,219,270 people into "
        "Eastern Orthodox, Catholic, Protestant, Armenian Apostolic and other Christian, "
        "and those rows are what is drawn. Resolving this column as well would draw "
        "Bulgaria's Christians twice. It is still what the census measured at the obshtina, "
        "which is why it is in COLUMNS below rather than simply deleted.",
    "Мюсюлманско":
        "NOT off the tree — REPLACED, exactly as `Християнско` is, by Sunni, Shia and "
        "unspecified rows from bg_split.py. See COLUMNS.",
    "Не мога да определя":
        "259,235 people, 4.0%, who ticked the offered box `I cannot determine`. An answer "
        "in the §9aq sense, since the form prints it, but not an answer that names a "
        "religion, and the tree has no node for declining to place oneself. `unaffiliated` "
        "would be wrong: `Нямам` is the no-religion box and these people did not tick it. "
        "hr2021.py excludes Croatia's `Ne izjašnjavaju se` on the same reasoning.",
    "Не желая да отговоря":
        "472,606 people, 7.2%, who ticked `I do not wish to answer`. The religion question "
        "has been voluntary at every Bulgarian census since 1992 and this is the box that "
        "says so; a refusal is not a religion.",
    "Непоказано1":
        "616,681 people, 9.5%, added from administrative registers because the census could "
        "not reach them, and never asked anything. A coverage residual rather than a "
        "refusal, and NSI keeps the two apart in its own footnote, so this file does too. "
        "This is mk2021.py's 132,260 case exactly. The trailing `1` is the footnote marker "
        "and is part of the header string, retained verbatim per §2.4.",
}

REVIEW = {
    "Източноправославно":
        "-> christianity.orthodox.canonical, the node Romania, Serbia, Greece and North "
        "Macedonia are all drawn on, which is the point of doing this at all. **4,091,780 "
        "people and 97.3% of Bulgarian Christians.** The count is NSI's published 2021 "
        "national figure to the person; only its distribution across the 265 obshtini is "
        "modelled, from the 2011 oblast composition.",
    "Католическо":
        "-> christianity.catholic, the PARENT and not `.latin`, because the census says "
        "`Католическо` with no rite. hr2021.py and cz2021.py file theirs the same way. "
        "38,709 people, and their real geography is much sharper than an oblast: most of "
        "Bulgaria's Catholics are Banat Bulgarians in **Rakovski** and around Plovdiv, plus "
        "a Svishtov community. This split puts them across Plovdiv oblast evenly, which is "
        "the honest limit of an oblast-level shape and is the single largest thing lost by "
        "not having a municipal breakdown.",
    "Протестантско":
        "-> christianity.protestant, the 'named no body' node, which is what the category "
        "is. 69,852 people, and heavily Roma: the 2021 census reports 32,325 Protestants "
        "among people who gave their ethnicity as Roma, 12.4% of that group, against 0.7% "
        "of ethnic Bulgarians.",
    "Арменско апостолическо":
        "-> christianity.oriental, the PARENT, which is au2021.py's, ee2021.py's and "
        "ge2014.py's call for the same body. 5,002 people. A "
        "`christianity.oriental.armenian` child is arguable on Georgia's 109,041 and would "
        "have to re-point all four countries at once; ge2014.py records why it was not "
        "taken.",
    "Друго християнско":
        "-> christianity.other. 13,927 people, and the one category here with no 2011 "
        "counterpart at all, so it is seeded at its national share in every oblast and "
        "asserts no geography whatever. That is the correct shape for 'we know the total "
        "and nothing about where'.",
    "Мюсюлманско сунитско":
        "-> islam.sunni. 604,875 people. **This is the weaker of the two splits and the "
        "reason is worth stating**: the Christian one is raked to a published 2021 national "
        "breakdown, and no such 2021 breakdown exists for Islam, so the Sunni/Shia division "
        "rests on the 2011 oblast shape alone.",
    "Мюсюлманско шиитско":
        "-> islam.shia. 29,470 people, Bulgaria's Alevi or Kazalbash, in Razgrad, Silistra, "
        "Targovishte and Sliven, and **the largest Shia population in Europe outside "
        "Turkey**. Nothing else on this map holds them. The figure is corroborated by a "
        "publication it was not built from: UNSD table 28 reports Bulgaria's 2021 `Muslim` "
        "as 611,129 against NSI's 638,708, and the 27,579 difference sits in UNSD's "
        "`Other`, which is a 2021 Muslim sub-category its classification could not code. "
        "The model lands 6.9% above that.",
    "Мюсюлманско неуточнено":
        "-> islam, the root, for the 2011 form's own `Мюсюлманско` cell: people who wrote "
        "Muslim without a branch. 4,363 people.",
    "Юдейско":
        "-> judaism. 1,736 people, the smallest drawn category in the country, and about "
        "half of them are in Sofia. UNSD table 28 carries the same figure under its own "
        "label `Yiddish`, which is the Demographic Yearbook's classification rather than "
        "Bulgaria's word.",
    "Друго":
        "-> other.bg. 6,451 people, 0.10%. Genuinely a tail rather than a coarse cell: it "
        "sits beside a `Друго` in the ethnicity table and beside the four Christian bodies "
        "and Judaism, so it is what is left after those, not a residual standing in for "
        "them.",
    "Нямам":
        "-> unaffiliated. 305,102 people, 4.7%. The form offers one box, `I have no "
        "religion`, with no atheist, agnostic or non-believer option, so nothing here "
        "belongs on `secular` — the same shape as Slovakia's `bez náboženského vyznania` "
        "and unlike Croatia, which asks the two apart.",
}

MAP = {
    # Measured at the obshtina, straight off the 2021 workbook.
    "Юдейско": "judaism",
    "Друго": "other.bg",
    "Нямам": "unaffiliated",
    # DERIVED, from bg_split.py. The two undivided 2021 columns, split by the 2011 oblast
    # composition raked to NSI's published 2021 national breakdown. `Християнско` and
    # `Мюсюлманско` themselves are no longer drawn; COLUMNS below is what a reader gets back
    # when inferred dots are hidden.
    "Източноправославно": "christianity.orthodox.canonical",
    "Католическо": "christianity.catholic",
    "Протестантско": "christianity.protestant",
    "Арменско апостолическо": "christianity.oriental",
    "Друго християнско": "christianity.other",
    "Мюсюлманско сунитско": "islam.sunni",
    "Мюсюлманско шиитско": "islam.shia",
    "Мюсюлманско неуточнено": "islam",
}

# spec §7a-i-1: the node each derived row's OWN SOURCE COLUMN names at the obshtina it is
# drawn on. Bulgaria's 2021 census measured `Християнско` and `Мюсюлманско` at every one of
# the 265 units, so `inferred dots: not shown` redraws exactly the table bg.csv holds, and
# not an ancestor-walk approximation of it.
COLUMNS = {
    "Християнско": "christianity",
    "Мюсюлманско": "islam",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
