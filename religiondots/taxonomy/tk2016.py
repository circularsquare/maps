"""Tokelau 2016 census, Table 5.8 -> religiondots taxonomy.

Seven rows, one answer per person, for the 1,197 usual residents present on census night; every
atoll column closes (sources/tk.py). The form named three churches and took any other answer in
writing (profile report, printed p.28). **One new node**, the national church.

    Congregational Christian   603  50.38%  -> christianity.reformed.congregational.tokelau  <- new
    Roman Catholic             463  38.68%  -> christianity.catholic.latin
    Presbyterian                71   5.93%  -> christianity.reformed.presbyterian
    Other Christian             50   4.18%  -> christianity
    Not stated                   9   0.75%  -> not on the tree (gap)
    No religion                  1   0.08%  -> unaffiliated

`Spiritualism and New Age religions` is 0 on every atoll in 2016, so sources/tk.py emits no row for
it and it needs no node.

Every row is counted at the atoll, which is the unit drawn, so every row is `measured` and
COLUMNS is empty.
"""

EXCLUDED = {
    "Not stated":
        "9 people, 0.75% (7 on Atafu, 2 on Nukunonu). The workbook's contents sheet says most "
        "`not stated` answers in 2016 were people whose age was imputed, so the questions after it "
        "went unanswered: a gap in coverage, not an answer. In `gap` beside the 302 usual residents "
        "who were overseas on census night and were not asked.",
}

REVIEW = {
    "Congregational Christian":
        "-> christianity.reformed.congregational.tokelau, a node added for it. 603 people, 50.4%; "
        "77.0% of Atafu, 62.7% of Fakaofo, 9.1% of Nukunonu. The label names no church, but the "
        "form offered it as one of Tokelau's three churches, and the Congregational congregations "
        "on the atolls are one body, the Congregational Christian Church of Tokelau. By "
        "Wikipedia's article on Samoa's CCCS they were a district of that church until they formed "
        "their own in 1996; a search snippet of an International Bible Reading Association partner "
        "page (now 404) says 1997; neither was checked with the church. So it goes beside `.cccs`, "
        "`.cicc`, `.ekt`, `.kpc`, `.ncc`, `.niue` and `.cccas` as the eighth national church of the "
        "Pacific Congregational set: not on `.cccs`, which it left, and not on the parent, which "
        "would leave Tokelau the one country of the set whose national church is unnamed. `.ekt` is "
        "Tuvalu's church. A member of Samoa's CCCS living on the atolls would tick this box too, "
        "and the census cannot tell them apart.",
    "Presbyterian":
        "-> christianity.reformed.presbyterian. 71 people, 54 of them on Atafu, where 2011 counted "
        "5. The 2016 form named Presbyterian as one of its three churches. Which body the Atafu "
        "Presbyterians belong to, and why the count rose, is in none of the census documents read "
        "(sources/tk.md). Drawn as counted.",
    "Other Christian":
        "-> christianity, not christianity.other. 50 people: answers written in beside the form's "
        "three named churches, so the residual of the form rather than bodies with no branch, as "
        "Gibraltar's `Other Christian` (gi2022.py). 24 of them are on Nukunonu, where 2011 counted "
        "none; the table does not say which churches.",
    "Roman Catholic":
        "-> christianity.catholic.latin, following as2015.py. The census says Roman Catholic, and "
        "nothing in it suggests an Eastern Catholic answer. 81.8% of Nukunonu.",
}

# Every row is `measured` at the atoll; nothing rolls up.
COLUMNS = {}

MAP = {
    "Congregational Christian": "christianity.reformed.congregational.tokelau",
    "Roman Catholic": "christianity.catholic.latin",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Other Christian": "christianity",
    "No religion": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
