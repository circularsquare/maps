"""
Hagstova Føroya, Census 2011, MT325 (congregation association by district) -> religiondots
taxonomy.

**Six answers by the 7 districts, people aged 15 and over who answered form question E23.**
E23 is voluntary and says "select all that apply", so MT325 counts ticks; `sources/fo.py` scales
each district's five body columns to people (the module docstring there says how and why), and
`None` is printed as people. National figures after the rescale, of 34,436 who answered:

    National Lutheran Church                27,002 ticks  23,510  68.3%  -> christianity.lutheran
    Plymouth Brethren                        5,381 ticks   4,619  13.4%  -> christianity.plymouth     REVIEW
    Christian missionary movements           4,085 ticks   3,463  10.1%  -> christianity.lutheran     REVIEW
    No congregation association              1,243 people  1,243   3.6%  -> unaffiliated              REVIEW
    Charismatic, evangelical congregations   1,262 ticks   1,091   3.2%  -> christianity.pentecostal  REVIEW
    Other congregations (district cells)       585 ticks     510   1.5%  -> christianity              REVIEW

EXCLUDED holds the universe row and the two kinds of non-answer.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total number of persons":
        "The universe, everyone aged 15 and over (37,965), equal to MT1's population aged 15+ "
        "in every district.",
    "Not stated":
        "2,369 people who returned the form and left the voluntary question blank. Which way it "
        "leans is unknown; MT321's religion question, asked beside it, has 2,210 blanks.",
    "Not queried":
        "1,160 people aged 15 and over who did not return the form at all (the table's note: "
        "'did not fill out the query form'). No lean can be read.",
}

REVIEW = {
    "Christian missionary movements":
        "-> christianity.lutheran, the same node as the National Church. **4,085 ticks, 3,463 "
        "people after the rescale.** The form names what the box means: 'Missiónshús, "
        "Meinigheitshús, Salvation Army, KFUM, KFUK, etc.', and Hagstova's Faroese label is "
        "`Samkomur nærri fólkakirkjuni`, congregations close to the National Church. The mission "
        "houses and congregation houses are the Lutheran lay revival that works inside the "
        "church, and KFUM/KFUK are its youth bodies; most people ticking this also ticked the "
        "National Church (4,596 ticked more than one box). The Salvation Army is Holiness, not "
        "Lutheran, and is inside the same box with no count of its own. Not `christianity.pietist`, "
        "whose description is the free-church line (Evangelical Covenant, Evangelical Free), "
        "which these movements stayed out of. Cost of the call: the movement's own geography "
        "(16.6% of Eysturoy's answers, 6.0% of S-streymoy's) is not a colour of its own.",

    "Plymouth Brethren":
        "-> christianity.plymouth, the parent, NOT `christianity.plymouth.open`. **5,381 ticks, "
        "4,619 people; 30.7% of Norðoyar's answers.** The Faroese Brøðrasamkomur are usually "
        "described as Open Brethren, but nothing read for this build says so, and Hagstova's "
        "label (`Brøðrasamkomur`) and the form's box name only the Brethren. Moving it to "
        "`.open` is a one-line change once a source is read. 713 people ticked both this and "
        "the National Church (MT325's one named pair).",

    "Charismatic, evangelical congregations":
        "-> christianity.pentecostal, the parent, NOT `.charismatic`. **1,262 ticks, 1,091 "
        "people.** The form's example is `Hvítusunnusamkomur`, the Pentecostal congregations, "
        "which are the classical Pentecostal line; `christianity.pentecostal.charismatic` is "
        "for movements outside it. Any Vineyard-type congregation that ticked this is on the "
        "parent too, which is not wrong for it.",

    "Other congregations":
        "-> christianity, the root, as an unspecified Christian answer. **585 ticks over the "
        "districts, 510 people.** E23 asks only about Christian churches, so `Other` is another "
        "Christian body. In a district the cell also holds the Adventist (93), Catholic (167), "
        "Orthodox (93) and Jehovah's Witness (126) answers, which are suppressed there and "
        "counted only nationally; together with the national Other (106) they are the 585 "
        "exactly. Not `christianity.other`, whose description excludes residuals. Splitting the "
        "cell by the national composition would draw four bodies under one dot in total with no "
        "district figure behind any of them.",

    "No congregation association":
        "-> unaffiliated. **1,243 people**, as printed, not scaled. It is the answer `None` to a "
        "question about Christian congregations, so a Muslim, Hindu or Buddhist who answered "
        "would tick it too: MT321 counts 124 people naming one of six non-Christian religions and "
        "149 another belief, nationally. MT321's own `No religious belief` is 1,397. Those "
        "non-Christians cannot be taken out of this cell (the two questions have different "
        "respondents), and at 273 people they are under a third of a dot.",
}

COLUMNS = {
    # Every category is counted at the district, which is the unit drawn, so no row is derived
    # and nothing rolls up. The rescale is within a district's own cells (sources/fo.py).
}

MAP = {
    "National Lutheran Church":                "christianity.lutheran",
    "Christian missionary movements":          "christianity.lutheran",
    "Plymouth Brethren":                       "christianity.plymouth",
    "Charismatic, evangelical congregations":  "christianity.pentecostal",
    "Other congregations":                     "christianity",
    "No congregation association":             "unaffiliated",
}


def resolve(category):
    """Source category -> node, or None for a category deliberately not on the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
