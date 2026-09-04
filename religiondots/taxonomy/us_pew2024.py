"""
Pew's 2023-24 Religious Landscape Study -> religiondots roots. The self-identification half
of spec.md §3.5a.

WHAT THIS FILE IS NOT. It is not a category mapping in the sense of `ca2021.py` or `pl2021.py`,
and it deliberately maps almost none of Pew's 149 categories. §3.5a settles that **the survey's
totals are taken at the root and nowhere else**: the denominations sit inside the root as ASARB's
structure and are never asked to clear a survey line of their own, because nine of nineteen
state-level subtractions tested came out negative and every one was a body that keeps a
baptismal register rather than a membership list. So `southern-baptist-convention` and
`global-methodist-church` are in the data, are not in this file, and are not supposed to be.
Matching Pew's denominations to ASARB's 372 bodies is the "meticulous cross-source matching"
spec §2.4 defers, and §3.5a removes the reason to do it.

WHAT IT IS. A **cut** across Pew's tree: the shallowest set of nodes that (a) partitions the
state's adults exactly once and (b) reaches a religiondots root. Most of the cut is Pew's own
top level. It descends in exactly two places, both because a single Pew node spans several of
our roots:

  * `other-christian` -> `metaphysical-family`, because Spiritualism and New Thought are roots
    of ours and Pew files them under Christianity;
  * `something-else`, whose three children and six grandchildren run from Unitarian
    Universalism to Wicca to Native American religions.

Everything below the cut is structure Pew publishes and this file ignores. Everything above it
is a Pew grouping (`christians`, `others`) that is not a religiondots node and never becomes one.

A CUT ENTRY IS A SET OF ROOTS, NOT ONE ROOT. `other-world-religions` is one published line
covering Sikhs, Daoists, Bahá'ís, Zoroastrians and their neighbours, and it cannot be split at
n=36,908 (§3.5a's third open question, and §4.4's problem). Recording it as the set it covers is
what keeps the arithmetic honest: the residual for that set is the line minus the ASARB rolls of
every root in it, so the 178,727 Bahá'ís ASARB counts are subtracted rather than drawn twice.
Reading it as a single opaque bucket would double them. Every other entry happens to name one
root, and the shape is the same.

THE ARITHMETIC THIS FEEDS, from §3.5a: per state and per root,
`residual = self_id - Σ rolls`, spread over the state's counties in proportion to
`county population - county adherents`, floored at zero, with any overflow charged to that
state's unaffiliated. This file supplies the `self_id` term and nothing else.

WHAT PEW CANNOT REACH. Of the roots ASARB actually counts adherents in -- christianity, islam,
judaism, buddhism, hinduism, unitarianuniversalist, bahai -- every one has a line here. The
other ten ASARB bodies report congregations and zero adherents (§4.4), so no root gets a roll
this cut cannot see, and the cap-and-record case of §3.5a stays where §3.5a found it.
"""

# ---------------------------------------------------------------------------------------
# THE CUT.  Pew category -> the religiondots paths that category's people belong to.
#
# Keys are `name` in data/normalized/us_pew.csv. A node that is not here and not an ancestor
# of something here is below the cut and unused. `apply()` enforces that the cut is a
# partition of each state, so an added or renamed Pew category fails loudly rather than
# quietly dropping people.
# ---------------------------------------------------------------------------------------
CUT = {
    # ---- christians: Pew's top level, unchanged except where it reaches outside Christianity.
    # Pew's evangelical / mainline / historically Black split is a classification of
    # traditions, not of bodies, and nothing downstream uses it -- all three are the same root.
    "evangelical-protestant": ("christianity",),
    "mainline-protestant": ("christianity",),
    "historically-black-protestant": ("christianity",),
    "catholic": ("christianity",),
    "latter-day-saint-mormon": ("christianity",),
    "orthodox-christian": ("christianity",),
    "jehovahs-witness": ("christianity",),

    # `other-christian` is NOT here: it is opened up, one level down.
    "others-in-the-other-christian-tradition": ("christianity",),
    # `metaphysical-family` is NOT here either; its three children go three ways.
    "spiritualist": ("spiritualism",),
    "unity-church": ("newthought",),
    "other-metaphysical-christian-other-christian-trad": ("christianity",),

    # ---- others: four named religions, one irreducible lump, and `something-else` opened up.
    "jewish": ("judaism",),
    "muslim": ("islam",),
    "buddhist": ("buddhism",),
    "hindu": ("hinduism",),

    # One line, 776,032 adults, published in 33 states. Pew's own contents for it are Sikh,
    # Daoist, Bahá'í, Zoroastrian, Jain and Shinto; we hold each as a root and cannot split
    # the line, so the set is recorded and the residual is taken against all of it at once.
    # Only `bahai` has a non-zero ASARB roll (178,727), so the rest are inert today and are
    # listed anyway, because the day one of them gets a roll is the day a silent list is
    # wrong.
    "other-world-religions": ("sikhism", "daoism", "bahai", "zoroastrianism", "jainism",
                              "shinto"),

    # `something-else` is NOT here, nor are `unitarians-and-other-liberal-faiths` or
    # `new-age`; all three are opened up.
    "unitarian": ("unitarianuniversalist",),
    "humanist": ("secular",),
    "deist": ("unchurched",),
    "spiritual-but-not-religious": ("unchurched",),
    "eclectic-a-bit-of-everything-i-have-my-own-beliefs": ("unchurched",),
    "other-in-the-unitarian-and-other-liberal-faiths-family": ("other.us",),
    "pagan-or-wiccan": ("paganism",),
    "other-in-the-new-age-family": ("esoteric",),
    "native-american-religions": ("indigenous.northamerican",),

    # ---- unaffiliated. Atheist and Agnostic are stated positions and go to `secular`, which
    # is where Canada's identical answers go; "nothing in particular" is the absence of one.
    # branches.py draws exactly this line and this is the second source to need it.
    "atheist": ("secular",),
    "agnostic": ("secular",),
    "nothing-in-particular": ("unaffiliated",),
}

# Categories deliberately kept off the tree.
EXCLUDED = {
    "no-answer":
        "3,592,631 adults, 1.40% of the country -- item non-response to the religion "
        "question. Not a religion and not 'no religion', which is its own answer at 48.3M. "
        "Excluded from the dots, exactly as Czechia's 'Neuvedeno' (30.05%), Ireland's 'Not "
        "stated' (6.7%), India's 'Religion not stated' and Brazil's 'Não sabe' are. "
        "CONSEQUENCE WORTH STATING: spec §3.5a says the re-basing makes each county's drawn "
        "total come to its population exactly, and with this excluded it comes to 98.60% of "
        "it. The share runs 0.09% (Massachusetts) to 4.88% (Alaska) between states, a "
        "54-fold spread, so it is not a uniform haircut either.",
}

# Defensible but arguable, recorded so the reasoning is not lost and can be overturned.
REVIEW = {
    "spiritual-but-not-religious":
        "-> unchurched. 1,839,514 adults, the largest single judgement in this file and the "
        "only one big enough to matter. Pew reaches it through `something else` -> "
        "`Unitarians and other liberal faiths`, so the respondent named a religion and this "
        "is what they named. `unchurched` is branches.py's 'People who report religious "
        "belief AND explicitly no institutional affiliation' -- Czechia's 960,201 believers "
        "who name no church, the classic believing-without-belonging category -- and that is "
        "what 'spiritual but not religious' says in four words. It is NOT `unaffiliated`, "
        "which is a report of no religion, and NOT `secular`, which is a stated non-theistic "
        "position. The imperfection: the Czech answer affirms belief and declines a church, "
        "while this one often declines the WORD religion rather than the institution. The "
        "alternative was to leave it inside a Unitarian Universalist bucket four times its "
        "size, which would be wrong about more people.",
    "deist":
        "-> unchurched, with `eclectic`. 59,683 adults in 11 states. Deism is theistic, so "
        "`secular` ('organised non-theistic bodies and stated secular positions') is the one "
        "place it certainly does not go; belief without an institution is what is left.",
    "eclectic-a-bit-of-everything-i-have-my-own-beliefs":
        "-> unchurched. 278,277 adults. 'I have my own beliefs' is a belief held outside any "
        "body, which is the node. `other.us` was the alternative and says less.",
    "metaphysical-family":
        "opened up rather than mapped, which moves 273,314 adults -- 0.11% of the country -- "
        "out of Christianity, against Pew's own classification of them as Christian. Done "
        "because Spiritualism and New Thought are roots in branches.py and Canada's "
        "'Spiritualist' already maps to one of them, so leaving these inside Christianity "
        "would put a Canadian and an American Spiritualist in different places. It costs "
        "nothing else: both roots have a zero ASARB roll, so no negative residual can come "
        "of it. The third child, `other metaphysical Christian`, stays in Christianity -- "
        "Christian Science is the large body in it, though Religious Science and Divine "
        "Science are New Thought and are misfiled by this call.",
    "other-world-religions":
        "the six roots listed are Pew's contents for the line, not ours, and the boundary is "
        "theirs to draw. Arithmetically it barely matters -- five of the six have no ASARB "
        "roll at all -- but if a future source gives Sikhs in America a number, this line "
        "stops being the right home for them and the entry has to be revisited rather than "
        "extended.",
    "historically-black-protestant":
        "-> christianity, like the other two Protestant traditions. Pew's tradition split is "
        "the organising idea of the whole study and it dissolves here, which looks like a "
        "loss and is not: the traditions are not bodies, ASARB enumerates bodies, and §3.5a "
        "uses Pew for root totals only. The split survives in the data file for anyone who "
        "wants it.",
}

# Pew nodes that are opened up instead of mapped -- listed so `apply()` can tell an intended
# descent from a category it has never seen.
OPENED = {
    "other-christian": ("metaphysical-family", "others-in-the-other-christian-tradition"),
    "metaphysical-family": ("spiritualist", "unity-church",
                            "other-metaphysical-christian-other-christian-trad"),
    "something-else": ("unitarians-and-other-liberal-faiths", "new-age",
                       "native-american-religions"),
    "unitarians-and-other-liberal-faiths": (
        "unitarian", "humanist", "deist", "spiritual-but-not-religious",
        "eclectic-a-bit-of-everything-i-have-my-own-beliefs",
        "other-in-the-unitarian-and-other-liberal-faiths-family"),
    "new-age": ("pagan-or-wiccan", "other-in-the-new-age-family"),
}


def resolve(name):
    """The religiondots paths a Pew category covers, or None if it is not on the cut."""
    return CUT.get(name)


def apply(rows):
    """One state's rows -> [(paths, adults, sample_size)], with the partition enforced.

    `rows` is every row of data/normalized/us_pew.csv for one state, as dicts with at least
    `name`, `parent`, `adults` and `sample_size`. Raises rather than returning a short answer,
    because the failure this guards against -- a Pew category that is neither on the cut nor
    below it -- removes people from the map without any downstream step noticing.
    """
    present = {r["name"]: r for r in rows}
    parent_of = {r["name"]: r["parent"] for r in rows}

    # Anything on the cut whose parent was opened up must actually be there: a descent that
    # loses its children would hand the state a hole the size of the parent.
    for opened, kids in OPENED.items():
        if opened not in present:
            continue
        got = [k for k in kids if k in present]
        if not got:
            raise ValueError(f"{opened!r} is present but none of its children are; the cut "
                             f"cannot descend and {present[opened]['adults']:,.0f} adults "
                             f"would be dropped")
        total = sum(present[k]["adults"] for k in got)
        if abs(total - present[opened]["adults"]) > 1.0:
            raise ValueError(f"{opened!r} is {present[opened]['adults']:,.2f} but the "
                             f"children on the cut sum to {total:,.2f}")

    out, seen = [], set()
    for name, row in present.items():
        if name in CUT:
            out.append((CUT[name], row["adults"], row["sample_size"]))
            seen.add(name)

    # Every row must be accounted for: on the cut, above it (opened up), below it (an
    # ancestor is on the cut), or explicitly excluded.
    def at_or_below_cut(n):
        """n itself, or some ancestor of it, is on the cut — so n's people are counted."""
        while n:
            if n in CUT:
                return True
            n = parent_of.get(n, "")
        return False

    stray = [n for n in present
             if n not in CUT and n not in OPENED and n not in EXCLUDED
             and not at_or_below_cut(n)]
    if stray:
        raise ValueError(f"{len(stray)} Pew categories are neither on the cut, below it, nor "
                         f"excluded: {sorted(stray)[:8]}")
    return out
