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
        "the largest single answer in Kenya. In KNBS's usage this is the MAINLINE mission "
        "inheritance — the Anglican Church of Kenya, the Presbyterian Church of East "
        "Africa, the Methodists, the Lutherans and the Salvation Army — as against "
        "`Evangelical Churches` beside it. The tree could hold every one of those apart and "
        "the census names none of them, so the answer-node is what the source supports. "
        "Worth knowing that this is a NARROWER category than the word usually is: an "
        "Anglican here is a Protestant, a Baptist here is an Evangelical.",
    "Evangelical Churches":
        "-> christianity.evangelical, a node added for Kenya. 9,648,690 people, 20.4%. "
        "KNBS prints it as a peer of `Protestant`, not a subset, and in Kenya it means the "
        "faith-mission and evangelical-alliance stream: the Africa Inland Church (the "
        "largest single body in it), the Baptists, the Pentecostal Assemblies of God, "
        "Deliverance Church, the Redeemed Gospel Church. It cuts across families the tree "
        "keeps apart — some of it is Baptist, much of it is Pentecostal — which is exactly "
        "why it goes to an answer-node and not to `christianity.pentecostal` or "
        "`christianity.baptist`. Filing 9.6M people under either would assert a division "
        "the census did not make. See the node's own note for why it was not added earlier.",
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
        "-> christianity.other. 1,732,911 people, 3.7%. A genuine residual, and a much "
        "*narrower* one than Ghana's identically-named cell: Kenya has already taken the "
        "Catholics, the mainline, the evangelicals, the AICs and the Orthodox out of it, so "
        "what is left is the Adventists, the Jehovah's Witnesses, the Latter-day Saints and "
        "the Quakers — Kenya has the largest Quaker population in the world — plus the tail. "
        "The tree could hold every one of those and the census separates none of them.",
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
}


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
