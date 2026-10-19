"""KNBS 2019 KPHC religious affiliation -> religiondots taxonomy.

Thirteen categories plus the universe total, at county. **The deepest religion question in
Africa**, and the only one on this map that counts either `Evangelical Churches` or
`African Instituted Churches` — two nodes were added to branches.py for it.

What makes Kenya's list good is that it splits Christianity FIVE ways where Ghana splits it
four and most of Africa splits it not at all, and that it gives Hindus, the Orthodox and
traditional religion cells of their own instead of burying them in a residual. The cost is
geography: 47 counties for 47.2M people. See sources/ke.md §2.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Don't Know":
        "73,253 people, 0.16%. The respondent did not know the religion of the person they "
        "were answering for — Kenya enumerates by household, so this is mostly about "
        "somebody else. A non-answer, not an answer (spec §3.5), and kept apart from `No "
        "religion /Atheists`, which is 10x larger and is a real answer.",
    "Not Stated":
        "6,909 people, 0.01%. Refusals and blanks. Same reasoning as `Don't Know`, and "
        "KNBS keeps the two apart, so this file does too.",
}

REVIEW = {
    "Protestant":
        "-> christianity.protestant, the 'named no body' node. 15,777,473 people, 33.4%, "
        "the largest single answer in Kenya. KNBS's 2019 Enumerator's Instructions Manual "
        "(IPUMS copy, PDF p.40) defines code 2 as \"Anglican Church of Kenya (ACK), "
        "Presbyterian Church of East Africa (PCEA), African Inland Church (AIC), Lutheran, "
        "Quakers (Friends), Methodists, Seventh Day Adventists (SDA), Salvation Army, "
        "Baptists, etc.\" So the Africa Inland Church, the Baptists, the Adventists and the "
        "Friends are all in here, not in `Evangelical Churches`; this note said otherwise "
        "until 2026-10-16. "
        "SPLIT SINCE 2026-10-16 (sources/ea_christian.py, unspecified_splits/eastafrica.md): "
        "each county's Protestants are divided at the Global Flourishing Study wave 1 (2023) "
        "shares of the churches in this cell, Anglican, Seventh-day Adventist, Presbyterian, "
        "Baptist and Methodist drawn as churches and the rest (the GFS's "
        "Independent/Evangelical box, which is mostly the Africa Inland Church, and the "
        "Lutherans) left here. In Kakamega, Bungoma and Vihiga the GFS's `Presbyterian` is "
        "also left here: GFS offers no Friends box and those counties are the Friends' "
        "heartland (Anita, 2026-10-16). Split rows are `derived` and roll back to this cell.",
    "Evangelical Churches":
        "-> christianity.evangelical, a node added for Kenya. 9,648,690 people, 20.4%. "
        "KNBS prints it as a peer of `Protestant`, not a subset. The enumerator manual (PDF "
        "p.40) defines code 3 as \"CITAM, Redeemed, Full Gospel, Pentecostal, Deliverance, "
        "PAG, Kenya Assemblies, Winners, Apostolic church, Jehovah Witnesses, etc.\": the "
        "Pentecostal and charismatic churches plus the Witnesses. The Africa Inland Church "
        "and the Baptists are in `Protestant` (this note filed them here until 2026-10-16). "
        "It still goes to an answer-node and not to `christianity.pentecostal`, because the "
        "Witnesses are in it and the census does not separate them.",
    "African Instituted Churches":
        "-> christianity.africaninstituted, a node added for Kenya. 3,292,573 people, 7.0%, "
        "and **the single most valuable category in this file**. These are the churches "
        "founded in Africa by Africans outside the mission denominations: in Kenya the "
        "Legio Maria (a Luo Catholic-derived church with its own pope), the African Israel "
        "Nineveh Church, the Nomiya Luo Church, the Akorino, and the Roho churches. Their "
        "geography is real and tight — they are a Nyanza and western Kenya phenomenon, "
        "which is visible on the map. "
        "Ghana has the same kind of church and no cell for it, so gh2021.py had to send the "
        "Musama Disco Christo Church and the Twelve Apostles to `christianity.other` and "
        "said this node was what it wanted. Kenya supplies it. That is spec §2.4's "
        "'deepening later costs nothing' working as designed: Ghana's rows still carry "
        "their `source_category` and can be moved if a Ghanaian source ever separates them.",
    "Orthodox":
        "-> christianity.orthodox, the parent, and not christianity.orthodox.canonical. "
        "201,263 people. Kenya's Orthodox are overwhelmingly the Greek Orthodox "
        "Patriarchate of Alexandria's Kenyan Orthodox Church, which is canonical — but the "
        "census says only `Orthodox`, and the Ethiopian and Eritrean communities in Nairobi "
        "are Oriental Orthodox, a different communion entirely (see the note on "
        "`christianity.oriental`). One cell covering two communions goes to the node that "
        "asserts the less, which is the same call mk2021.py makes.",
    "Other Christian":
        "-> christianity.other. 1,732,911 people, 3.7%. The enumerator manual (PDF p.41) "
        "defines code 6 as \"individuals who confess christian faith but are not affiliated "
        "to any of the above faiths, e.g. people who don't go to any church yet they "
        "consider themselves Christians.\" So it is Christians with no church, not a list "
        "of small bodies: the Adventists and the Friends are in `Protestant` and the "
        "Witnesses in `Evangelical Churches` (this note said otherwise until 2026-10-16). "
        "By zm2022.py's rule an unaffiliated-Christian answer belongs on bare "
        "`christianity`; moving it is not done here, it is a question for Anita.",
    "Islam":
        "-> islam, with no branch, because the census gives none. 5,152,194 people, 10.9%, "
        "and among the most concentrated distributions on this map: Wajir, Mandera and "
        "Garissa are each over 97%, and the coast strip runs high. Overwhelmingly Sunni "
        "Shafi'i, with Ismaili and Bohra communities in Nairobi and Mombasa; nothing here "
        "separates them, and `islam.sunni` would be an inference rather than a reading.",
    "Traditionists":
        "-> indigenous.african, the node Ghana added. 318,727 people, 0.68% — far lower "
        "than Ghana's 3.25%, and the gap is about the question rather than the country: "
        "Kenya offers thirteen boxes and the four Christian ones absorb almost everyone. "
        "Read it as a floor for the same reason as Ghana (see that note): the category is "
        "exclusive of the Christian and Muslim boxes and Kenyan traditional practice "
        "commonly accompanies one of them. KNBS's own spelling is `Traditionists`, which is "
        "not a standard word; kept verbatim per §2.4.",
    "Hindu":
        "-> hinduism. 60,287 people, and one of the few African census cells anywhere for "
        "it. Kenya's Hindus are the East African Asian community — Nairobi, Mombasa and "
        "Kisumu — and this is a case where a tiny national share is a large local one.",
    "No religion /Atheists":
        "-> unaffiliated. 755,750 people, 1.60%. One cell for both, so nothing goes to "
        "`secular`, which needs a separately-counted atheist or humanist answer. The same "
        "call hr2021.py and mk2021.py make for their combined cells. The internal spacing "
        "of KNBS's label is theirs and is kept verbatim (§2.4).",
    "Catholic":
        "-> christianity.catholic, the parent rather than .latin. Kenya's Catholics are "
        "Latin rite and the census does not say so.",
    "Other Religion":
        "-> other.ke. 467,083 people. Per source, per spec §3.11.",
}

MAP = {
    "Catholic": "christianity.catholic",
    "Protestant": "christianity.protestant",
    "Evangelical Churches": "christianity.evangelical",
    "African Instituted Churches": "christianity.africaninstituted",
    "Orthodox": "christianity.orthodox",
    "Other Christian": "christianity.other",
    "Islam": "islam",
    "Hindu": "hinduism",
    "Traditionists": "indigenous.african",
    "Other Religion": "other.ke",
    "No religion /Atheists": "unaffiliated",
    # 2026-10-16, sources/ea_christian.py: ke_split.csv replaces `Protestant` with these, at
    # GFS 2023 county shares; the remainder keeps the `Protestant` label. `derived`.
    "Protestant: Anglican": "christianity.anglican",
    "Protestant: Seventh-day Adventist": "christianity.adventist.sda",
    "Protestant: Presbyterian": "christianity.reformed.presbyterian",
    "Protestant: Baptist": "christianity.baptist",
    "Protestant: Methodist": "christianity.methodist",
}

# spec §7a-i-1: the source column each derived row rolls back to.
COLUMNS = {v: v for v in MAP.values()}
COLUMNS["Protestant"] = "christianity.protestant"


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
