"""Statistics Botswana 2011 PHC religion -> religiondots taxonomy.

Nine categories at named-locality level, from the eighteen district booklets of the
*Population and Housing Census 2011 Selected Indicators* series. The universe is the
population aged 12 and over, which is Peru's shape (§9 and `countries.py`): the under-twelves
were never asked and are in `gap=` rather than drawn as a §3.5 undercount.

**THE CATEGORY LIST IS SHALLOW ON CHRISTIANITY AND UNUSUALLY GOOD EVERYWHERE ELSE.**
`Christian` is one undivided cell holding 79.3% of the answering population, with no
denominations at all, which on `sources.md` §11b's phrase makes this a Germany-shaped
country. What redeems it is the rest of the row: Botswana gives **Badimo** its own box under
its own Setswana name, gives **Rastafarian** a box at 2,030 people, and gives Bahá'í a box at
2,074 — so the residual `Other` is 1,461 people, 0.10%, one of the smallest on this map.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the locality's own answering population, not a category. Every booklet prints it "
        "and sources/bw.py asserts the nine categories sum to it on every row.",
    "Not stated":
        "the census's own non-response cell, 1.1% of the universe. Not drawn, per §3.5: it "
        "is a property of the enumeration and not an answer anybody gave. It is in `gap=` "
        "beside the under-twelves.",
}

REVIEW = {
    "Christian":
        "-> christianity, the undivided root, because the census gives no denomination at "
        "all. 1,171,537 people, 79.3% of everyone who answered. Botswana's churches are "
        "mostly known from elsewhere: the London Missionary Society's nineteenth-century "
        "Bangwato and Bakwena missions became the United Congregational Church of Southern "
        "Africa, the Anglicans and Methodists followed, and the twentieth century added the "
        "Zion Christian Church out of South Africa and a large Pentecostal sector. **None "
        "of that is in the table**, so no branch is inferred and the whole cell sits on the "
        "root, which is the call bd2011.py makes for Bangladesh. "
        "SINCE 2026-10-03 two churches are split out of it by the Afrobarometer (`Roman "
        "Catholic`, `Seventh Day Adventist` below; sources/bw.md §10), and this row is what "
        "they leave: 92.7% of the census's Christians. "
        "Its geography is the inverse of everything else here: it is HIGHEST in the eastern "
        "corridor and the towns and lowest in the Kalahari, where Badimo and no-religion "
        "between them take a fifth of some settlements.",
    "Roman Catholic":
        "-> christianity.catholic, bare, as na2023.py and zm2022.py. NOT A CENSUS CATEGORY: "
        "sources/bw_churches.py splits each locality's census Christians at its survey unit's "
        "Catholic share of Christians, Afrobarometer rounds 4-8 (2008-2019), `derived`, "
        "rolling back to `christianity` through COLUMNS. 53,819 people aged 12+, 4.86% of "
        "Christians, and **a floor**: everyone answering `Christian only` (5% to 57% of "
        "respondents by round) stays on the parent. Drawn because the share holds while the "
        "unnamed rise (rounds 7-8 keep 0.78 of rounds 4-5, against 0.45 for named Christians) "
        "and it passes the split-half at 21 units (+0.464, null 95th +0.293). South East 13.7% "
        "of Christians, Lobatse 12.3%, Kweneng East 8.5%, Ghanzi 8.4%; Kalahari and the north "
        "about 1%. The Diocese of Gaborone counts 7.0% of its population as baptised Catholics "
        "(2013); the Pew Forum's 22% of Christians (2008-09) is not credible beside either.",
    "Seventh Day Adventist":
        "-> christianity.adventist.sda, as zm2022.py and na2023.py. Split like `Roman "
        "Catholic`: 27,273 people aged 12+, 2.46% of Christians, a floor. Keeps 0.81 of its "
        "level while the unnamed rise; split-half +0.481 (null 95th +0.285). Ngamiland West "
        "13.2% of Christians, Ngamiland East 9.6%, Chobe 8.9% (next to Namibia's Zambezi, 41% "
        "Adventist on that map), Ngwaketse 4.5% (Kanye's Adventist hospital); none found in "
        "South East or Kgalagadi. The church counted 47,590 baptised members, one in 49 people, "
        "in 2021.",
    "Badimo":
        "-> indigenous.african, the node Ghana added and Zimbabwe, Malawi and Benin have "
        "since fed. 60,613 people, 4.1%. "
        "**Badimo is the census's own word and it is not a synonym for `traditional`**: it "
        "is the Setswana plural for the ancestors, and what the box counts is people who "
        "named the ancestral cult as their religion rather than a church. No child node is "
        "minted for it (§2.4) — the same argument zw2022.py makes, that a single national "
        "cell naming one tradition does not need its own leaf when `indigenous.african` "
        "already holds exactly this. "
        "**Read it as a floor**, for the reason sources.md §11b gives for the whole "
        "continent: the box is exclusive of the Christian one, and consulting a *ngaka*, "
        "keeping the *bogwera* observances or brewing for the ancestors very commonly "
        "accompanies church membership in Botswana rather than replacing it. The number "
        "counts people who chose it INSTEAD of Christianity, which is a much smaller set "
        "than the people who practise it.",
    "No religion":
        "-> unaffiliated. 225,416 people, **15.3%, and this is the figure the country is "
        "worth drawing for.** It is far above anything else counted in sub-Saharan Africa "
        "here: Zimbabwe's is 8.3%, Zambia's census puts Christianity at 98%, and Ghana, "
        "Kenya and Malawi are all low single digits. "
        "**It is not an urban figure and that is the surprising part.** The cities and "
        "towns are 9.7% against a national 15.3%, so no-religion in Botswana rises as you "
        "leave town, which is the opposite of the pattern Kazakhstan showed and of what a "
        "reader would expect. The peaks are Kalahari settlements. "
        "**And it should be read against Benin's warning**: where a census offers "
        "`Badimo` as a box exclusive of Christianity, some people whose practice is "
        "ancestral rather than congregational will answer `no religion` instead, so the "
        "two cells are not independent of each other. The 2022 census puts the same cell "
        "at 6.9%, less than half, which no plausible amount of conversion explains and "
        "which sources/bw.md §4 treats as a question about the question rather than about "
        "Botswana.",
    "Other":
        "-> other.bw. 1,461 people, 0.10%, and a genuine tail rather than a store cupboard: "
        "Islam, Hinduism, the Bahá'í Faith and Rastafari all have their own boxes above it, "
        "so what is left is Judaism, Buddhism, Sikhism and anything the enumerator could "
        "not place. The booklets label it `Other religion (NEC)`, not elsewhere classified. "
        "Per §3.11.",
    "Rastafarian":
        "-> rastafari. 2,030 people, 0.14%. **The third census on this map to count "
        "Rastafari by name**, after Jamaica 2011 and Saint Vincent 2012, and the first "
        "outside the Caribbean. Worth keeping distinct rather than folding into `Other` "
        "for exactly that reason: it is a movement a state usually declines to count.",
    "Bahai":
        "-> bahai. 2,074 people, 0.14% — and larger than the Rastafari cell, which is "
        "unusual. The Bahá'í Faith has been organised in Botswana since the 1950s and the "
        "census has given it a box since 2001.",
    "Muslim":
        "-> islam, the root, with no branch. 10,941 people, 0.74%. Botswana's Muslims are "
        "predominantly South Asian in origin, with a more recent East and West African "
        "trading population; the census names no school or sect and none is inferred. "
        "Concentrated in the towns and in the northern trading centres.",
    "Hindu":
        "-> hinduism. 3,729 people, 0.25%. Almost entirely the Gujarati commercial "
        "community, and its geography is Gaborone and Francistown and very little else.",
}

MAP = {
    "Christian": "christianity",
    "Roman Catholic": "christianity.catholic",
    "Seventh Day Adventist": "christianity.adventist.sda",
    "Muslim": "islam",
    "Bahai": "bahai",
    "Hindu": "hinduism",
    "Badimo": "indigenous.african",
    "No religion": "unaffiliated",
    "Rastafarian": "rastafari",
    "Other": "other.bw",
}


# spec §7a-i-1: the two church rows (sources/bw_churches.py -> bw_split.csv) are `derived` out of
# the census's own `Christian` cell and carry `parent_column=Christian`, so with inferred dots
# hidden they roll back to the `christianity` the census counted in that locality.
COLUMNS = {
    "Christian": "christianity",
}


def _key(cat):
    # Whitespace only, NOT case. `tools/check_mapping.py` borrows this function to fold
    # EXCLUDED, and a lowercasing _key makes it report a deliberately excluded universe row
    # as an unmapped category. The nine labels are minted by sources/bw.py, so exact case is
    # something this file can rely on.
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
