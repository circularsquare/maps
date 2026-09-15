"""Togo RGPH-5 2022 religion (UNSD table 28), on an Afrobarometer pattern -> religiondots taxonomy.

Fifteen census rows plus one residual at 6 units (the five regions, with Lomé, Golfe 1 to 5, apart).
Every row is the census's national count; `sources/tg.py` fits it to the 2022 unit populations with
the pattern of Afrobarometer rounds 5-9, and `sources/tg.md` is the record. Shares below are of the
7,823,453 people with a stated religion unless they say otherwise.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Not Stated":
        "187,619 people, 2.32% of the census count. The census's own non-response row (spec §3.5).",
    "Unknown":
        "47,100 people, 0.58%. A second non-response row in the table INSEED forwarded to UNSD, kept "
        "apart from `Not Stated` as the table keeps it; what separates the two is not documented.",
    "Not in the religion table":
        "37,326 people, 0.46%: the 2022 census counted 8,095,498 (Livret 01 Tableau 1) and the 15 "
        "religion rows UNSD holds sum to 8,058,172, 23,099 short in towns and 14,228 in the "
        "countryside. Built by `sources/tg.py` so the two margins meet; not a religion anybody gave.",
}

REVIEW = {
    "Catholic":
        "-> christianity.catholic. 1,688,420, 21.58%. The survey reads Catholics at 1.31x the census "
        "and their order across the units does not repeat between halves of the rounds (+0.371 "
        "against a null of +0.544), so the row is seeded at the national rate and drawn by what the "
        "five placed religions leave in each unit: 27.0% in Lomé, 16.7% in Centrale. The survey's "
        "own pooled shares put Savanes second (29.4%), which the map does not show.",
    "Muslim":
        "-> islam, no branch. 1,499,867, 19.17%. `Sunni only` took 23 of 847 Muslim answers. Placed "
        "by the survey (+0.886): Centrale 52.7%, Savanes 24.6%, Kara 23.5%. The survey reads Muslims "
        "at 0.75x the census, and UNSD's urban row puts the census's Muslims more urban (1.22x the "
        "whole) than the survey's (1.08x).",
    "Animist":
        "-> indigenous.african. 1,367,266, 17.48%, 87.6% of them rural. The census has a `No "
        "Religion` row beside it, so this is not a box that lumps the two (the draft procedure's "
        "case in WORKFLOW_PLAN.md does not arise). Placed by the survey's `Traditional/ethnic "
        "religion` (+0.943), which reads 0.59x the census, as the Afrobarometer does wherever a "
        "census counts traditional religion: Savanes 29.7%, Maritime 23.4%, Lomé 5.3%.",
    "No Religion":
        "-> unaffiliated. 759,447, 9.71%. The census offers `Animist` separately (step 2 of the "
        "no-religion procedure). Placed by the survey's `None` and `Atheist` (+0.600, p = 0.043, the "
        "weakest pass): Kara 15.9%, Centrale 5.5%. Traditional and None move against each other over "
        "the rounds (None 4.2% to 10.4%, traditional 11.3% to 8.1%), but not unit by unit in step "
        "(`sources/tg.py::swap_table`), both pass on their own, and the census fixes both levels, so "
        "they are placed apart.",
    "Assembly of God":
        "-> christianity.pentecostal.trinitarian, as `ck2011`, `fj2007`, `ki2015` and `mh1999`. "
        "692,659, 8.85%. The survey's Togo card had an Assembly of God box in rounds 5 and 6 only "
        "(149 answers) and nobody chose it after, while `Evangelical` rose from 4.1% to 13.9%, so "
        "the row's pattern would come from the two together; they fail the split-half (+0.514), so "
        "the row is seeded at the national rate and the pairing moves nothing drawn.",
    "Pentecostal":
        "-> christianity.pentecostal. 496,978, 6.35%. Placed by the survey's `Pentecostal` (+0.886), "
        "which holds its level across rounds (5.8-9.1%): Plateaux 10.8%, Lomé 8.2%, Centrale 1.3%.",
    "Evangelical Presbyterian Church":
        "-> christianity.reformed.presbyterian. 283,513, 3.62%: the Église Évangélique Presbytérienne "
        "du Togo, from the North German Mission among the Ewe. Placed by the survey's `Presbyterian` "
        "(+0.943, level 4.2-5.3% by round): Plateaux 9.0%, Lomé 5.0%, under 1% in the three "
        "northern regions. The survey's `Evangelical` answers are not added to it; the row's name "
        "would allow either box, and `Presbyterian` alone is the one that holds its level.",
    "Baptist":
        "-> christianity.baptist. 162,458, 2.08%. Fails the split-half (+0.429); national rate.",
    "Methodist":
        "-> christianity.methodist. 26,955, 0.34%. Passes the split-half (+0.649) on 44 answers, "
        "under the 1% floor for placing a survey category; national rate.",
    "Jehovah's Witnesses":
        "-> christianity.witnesses. 51,087, 0.65%. Passes (+0.812) on 55 answers, under the 1% "
        "floor; national rate.",
    "Adventist":
        "-> christianity.adventist, the census naming no Adventist body. 17,960, 0.23%. Fails; "
        "national rate.",
    "Other Christians":
        "-> christianity.other, as `bj2013`'s `Autres chrétiens`. 545,770, 6.98%, and the most urban "
        "row in the table (62.7% urban, 1.46x the whole), which fits the Celestial Church of Christ "
        "and the independent churches of Lomé; the census names none of them. Seeded from the "
        "survey's small named churches (Orthodox, Church of Christ, the Celestial and Zionist boxes, "
        "Independent: 2.1%), which fail (+0.229); national rate.",
    "Other Religions":
        "-> other.tg, a node added for Togo like `other.cm` and `other.bf`. 231,073, 2.95%. The "
        "survey's `Other`, Hindu and Bahá'í boxes took 31 answers and fail; national rate.",
}

MAP = {
    "Catholic": "christianity.catholic",
    "Muslim": "islam",
    "Animist": "indigenous.african",
    "No Religion": "unaffiliated",
    "Assembly of God": "christianity.pentecostal.trinitarian",
    "Pentecostal": "christianity.pentecostal",
    "Evangelical Presbyterian Church": "christianity.reformed.presbyterian",
    "Baptist": "christianity.baptist",
    "Methodist": "christianity.methodist",
    "Jehovah's Witnesses": "christianity.witnesses",
    "Adventist": "christianity.adventist",
    "Other Christians": "christianity.other",
    "Other Religions": "other.tg",
}

# spec §7a-i-1: every row is counted at the node it is drawn on; nothing is inferred downward.
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
