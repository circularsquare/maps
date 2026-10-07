"""
NORC 2016 (Cuba, question Z10, public use file; grouped in `sources/cu.py`) -> religiondots taxonomy.

Named for the survey year. `sources/cu.py` and `sources/cu.md` have the construction: NORC's
weighted national shares of 835 Cuban adults, the same mix in every province on ONEI's 2024 count,
every row `modelled` (spec §7b), because no Cuban census asks religion and the public file has no
region below the nation.

    28.28%  Catholic                                                -> christianity.catholic.latin
    16.91%  Santeria or Order of Osha                               -> afrodiasporic.santeria (NEW)
     6.47%  Christian, not Catholic (Evangelical, Protestant, other) -> christianity
    21.84%  Believe in god but do not belong to a particular religion -> unchurched
     1.13%  Atheist                                                 -> secular
    24.37%  None of the above                                       -> unaffiliated
     1.01%  Other                                                   -> other.cu (NEW)
"""

EXCLUDED = {}

REVIEW = {
    "Santeria or Order of Osha":
        "-> afrodiasporic.santeria, a node added with this country, beside Vodou, Orisha and "
        "Revival Zion (spec §3.3: a syncretism gets its own node). It adds a legend row only Cuba "
        "draws, which is why it is also ask 044's subject; the fallback is the family parent "
        "`afrodiasporic` (Brazil's and Mexico's unnamed answers sit there), which loses the name.",
    "Christian, not Catholic (Evangelical, Protestant or other Christian)":
        "-> christianity, the bare family node. The card had three boxes (Evangelical 1%, "
        "Protestant under 0.5%, Christian (other) 6% in the topline), but the public file merges "
        "them into one code, and 'Christian (other)', most of it, names no church. In Cuba it will "
        "be mostly Pentecostal, Baptist, Methodist, Adventist and Jehovah's Witness, but nothing "
        "measures the split, so no branch is drawn.",
    "Believe in god but do not belong to a particular religion":
        "-> unchurched, as Venezuela's and Guatemala's LAPOP box for the same answer: a "
        "positive report of belief without a church, not no religion.",
    "Atheist":
        "-> secular, as every other source's `Atheist` box.",
    "None of the above":
        "-> unaffiliated. The card's last real box before `Other (specify)`, after Catholic, "
        "Santería, three Christian boxes, Jewish, Muslim, Buddhist, Atheist and the believer "
        "box. With a separate `Other (specify)` offered, choosing none of the listed answers "
        "reads as no religion, which is how sources.md §11ap recorded it. Some of it may be "
        "people who practise a religion the card left out (espiritismo, Palo Monte, Abakuá) "
        "and did not write it in; nothing measures how many.",
    "Other":
        "-> other.cu, a per-source residual (spec §3.11). The specified text is not in the "
        "public file.",
}

MAP = {
    "Catholic": "christianity.catholic.latin",
    "Santeria or Order of Osha": "afrodiasporic.santeria",
    "Christian, not Catholic (Evangelical, Protestant or other Christian)": "christianity",
    "Believe in god but do not belong to a particular religion": "unchurched",
    "Atheist": "secular",
    "None of the above": "unaffiliated",
    "Other": "other.cu",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, a national survey share on a count.


def resolve(category):
    """religiondots branch for a NORC answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
