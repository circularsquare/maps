"""ESS `rlgdnm` -> religiondots taxonomy. Greece's Greek-citizen half.

The European Social Survey asks `rlgblg` — do you consider yourself as belonging to any
particular religion or denomination — and only then `rlgdnm`, which one. So the category list
below is eight religions plus a "Not applicable" that is not a non-response at all: it is
everyone who answered no to the first question, and it is the second largest answer in Greece.

    1   Roman Catholic                 -> christianity.catholic.latin
    2   Protestant                     -> christianity.protestant
    3   Eastern Orthodox               -> christianity.orthodox.canonical.greek
    4   Other Christian denomination   -> christianity
    5   Jewish                         -> judaism
    6   Islam                          -> islam.sunni
    7   Eastern religions              -> other.gr
    8   Other Non-Christian religions  -> other.gr
    66  Not applicable                 -> unaffiliated
    77/88/99  Refusal / DK / No answer -> excluded, spec §3.5

THE CATEGORY LIST SHRINKS AS THE SAMPLE SHRINKS, and that is a property of a survey rather
than of Greece. Round 5 offers nine codes and Greek respondents use eight of them; round 11
uses five, and **`Islam` is not among them** — not because Greece stopped having Muslims but
because 2,757 respondents reached none. Pooling the three rounds is what keeps the small
categories alive at all, and even pooled they are a handful of people each. Read the Orthodox
and unaffiliated shares; distrust everything below one percent.

AND THE ONE CATEGORY ESS CANNOT SEE, which is why this file has a tenth entry. Greece's
recognised Muslim minority of Western Thrace — 100,000-120,000 people under the 1923 Treaty of
Lausanne — are Greek citizens, and ESS finds essentially none of them: nine weighted
respondents in round 5, zero in rounds 10 and 11, against a true share near 29% of their
region. They are Turkish- and Pomak-speaking and a Greek-language national sample does not
reach them. `sources/gr.py` therefore SPLITS them out of the Anatoliki Makedonia-Thraki
citizen population using the published minority figure, which is spec §3.1's permitted
operation and the same move UCIDE performs for Spain (§9y). Without it the map would say the
historically Muslim region of Greece is its least Muslim one.
"""

EXCLUDED = {
    "Refusal": "ESS's 77. A refusal to answer, so spec §3.5 marks it rather than filling "
               "it. Tiny in Greece — under half a percent of respondents in every round.",
    "Don't know": "ESS's 88, and the same treatment.",
    "No answer": "ESS's 99, and the same treatment.",
}

REVIEW = {
    "Eastern Orthodox":
        "-> christianity.orthodox.canonical.GREEK, not the bare `canonical`. Naming the "
        "Church of Greece is what the geography justifies: this is a Greek national sample "
        "and the Orthodox of Greece are its faithful, apart from the Dodecanese and Crete, "
        "which are under the Ecumenical Patriarchate directly. That distinction is real and "
        "is not drawn, because ESS does not ask it and no node exists for it. 89% of the "
        "country and the largest single node Greece puts on the map.",
    "Not applicable":
        "-> unaffiliated, and this is the most consequential call in the file. It is not a "
        "missing value: it is everyone who answered NO to `rlgblg`, which in Greece is about "
        "6% of citizens. branches.py's line is whether a POSITION is stated — `unaffiliated` "
        "is a report of not belonging, `secular` is a stated non-theistic stance — and 'I do "
        "not belong to a religion' is plainly the first. **Nothing in Greece reaches "
        "`secular` at all**, because ESS never offers atheist or agnostic as a denomination; "
        "ge2014.py makes the same call for the same reason.",
    "Other Christian denomination":
        "-> christianity, the root, following ru2012.py for Arena's 'Christianity, but not "
        "Orthodox, Catholic nor Protestant' and mk2021.py for `Христијани`. The answer names "
        "no body and rules out the three that have codes of their own, so the parent is the "
        "honest place. In Greece it is largely Jehovah's Witnesses — the country's largest "
        "non-Orthodox Christian body, with a long history of conscientious-objection cases — "
        "and Old Calendarists, and neither is separable here.",
    "Eastern religions":
        "-> other.gr with 'Other Non-Christian religions', for hr2021.py's and ru2012.py's "
        "reason: the tree has no node for 'some Eastern religion, unspecified' and picking "
        "one would invent a fact. Buddhism and Hinduism are not offered separately by ESS.",
    "Islam":
        "-> islam.sunni. Greece's Muslims are Sunni almost throughout — the Thracian minority "
        "is Hanafi, and so are the Egyptian, Syrian, Pakistani and Bangladeshi communities of "
        "Athens. **The Bektashi and Alevi Pomaks of the Rhodope mountains are the exception "
        "and are not separable**: no source counts them, ESS does not ask, and the tree's "
        "`alevism` node would be an assertion rather than a reading. Said here rather than "
        "silently rolled in.",
    "Muslim minority of Thrace (Treaty of Lausanne)":
        "-> islam.sunni. NOT an ESS category — sources/gr.py adds it, and taxonomy/gr2024.py "
        "carries it so that the operation is visible in gr.csv and checkable by "
        "tools/check_mapping.py. See the docstring: ESS reaches essentially none of this "
        "population and the alternative to splitting it out is a false statement about a "
        "recognised national minority. The magnitude is the Council of Europe's ECRI figure "
        "of 100,000-120,000, quoted in the US State Department's religious-freedom reports; "
        "the midpoint is used and the range is in note_public.",
    "Monastic community of Mount Athos":
        "-> christianity.orthodox.canonical.greek. Also not an ESS category. Mount Athos is "
        "NUTS `ELZZZ`, an extra-regio unit of 1,811 people that no sample will ever contain, "
        "and it is an autonomous Orthodox monastic state where only Orthodox monks may "
        "reside. Drawing it from mainland Macedonia's mixture would put irreligious and "
        "Muslim dots on the Holy Mountain; drawing it as Orthodox is the one authored cell "
        "in this country and it is about as safe as an authored cell gets.",
}

MAP = {
    "Roman Catholic": "christianity.catholic.latin",
    "Protestant": "christianity.protestant",
    "Eastern Orthodox": "christianity.orthodox.canonical.greek",
    "Other Christian denomination": "christianity",
    "Jewish": "judaism",
    "Islam": "islam.sunni",
    "Eastern religions": "other.gr",
    "Other Non-Christian religions": "other.gr",
    "Not applicable": "unaffiliated",
    # --- the two categories sources/gr.py adds; see REVIEW
    "Muslim minority of Thrace (Treaty of Lausanne)": "islam.sunni",
    "Monastic community of Mount Athos": "christianity.orthodox.canonical.greek",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
