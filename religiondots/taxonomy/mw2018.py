"""NSO Malawi 2018 PHC religious denomination -> religiondots taxonomy.

Ten denominations plus the universe total, at district. **The first source on this map to
name a single Presbyterian body**, and the first African one to separate the Anglicans from
a generic Protestant cell.

What makes Malawi's list unusual is that it is a DENOMINATION question rather than a
religion question: eight of the ten cells are Christian groupings and the other two are
Islam and No Religion. That buys real Christian detail — Catholic, CCAP and Anglican are
each their own body — and it costs everything else, because Buddhism, Hinduism, Judaism and
the Bahá'ís have no cell at district and land in one residual. See sources/mw.md §3.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
}

REVIEW = {
    "Catholic":
        "-> christianity.catholic, the parent rather than .latin. 3,028,435 people, 17.2%, "
        "the largest single named body in Malawi. Latin rite throughout and the census does "
        "not say so, which is the same call ke2019.py makes.",
    "CCAP":
        "-> christianity.reformed.presbyterian, and it is the first time any source here "
        "has put people on that node from outside the North Atlantic. 2,498,969 people, "
        "14.2%. The Church of Central Africa Presbyterian is a single body — a 1924 union "
        "of the Scottish missions, Livingstonia in the north, Blantyre in the south and the "
        "Dutch Reformed Nkhoma synod in the centre — so this is a named church and not an "
        "answer, and it goes to the named node. "
        "**Its geography is the mission map of the 1880s, still legible, and it is a "
        "three-synod pattern rather than a northern one.** CCAP is 21.2% of the Northern "
        "Region and 17.7% of the Central against 14.2% nationally, and only 8.8% of the "
        "Southern — but the top six districts are not all in the north: Mzuzu City 28.0%, "
        "Mzimba 23.9% and Rumphi 22.7% are Livingstonia, while Lilongwe City 23.2% and Dowa "
        "22.4% are Nkhoma. The southern figure is low because Blantyre synod is the "
        "smallest of the three and its ground is shared with the Anglicans and the "
        "Catholics. NSO writes the name `Church of Central African Presbyterian` in Table "
        "3.4; the body's own name is Church of Central Africa Presbyterian, and the "
        "source's spelling is kept in `source_category` per §2.4.",
    "Seventh Day Adventist/Baptist/Apostolic":
        "-> christianity.sdabaptistapostolic, a node added for Malawi and the only one on "
        "the tree that merges three traditions. 1,644,829 people, 9.4%. **This is the one "
        "arguable call in the file and it is worth stating plainly**: the tree holds "
        "`christianity.adventist`, `christianity.baptist`, `christianity.pentecostal` and "
        "`christianity.africaninstituted` separately, and every person in this cell belongs "
        "to one of them — but the census does not say which, and the three are not even the "
        "same kind of thing. Adventists are a 19th-century American body, Malawi's Baptists "
        "are mostly the post-1960s Baptist Convention, and `Apostolic` here means the "
        "African Apostolic and Apostolic Faith Mission churches, which are locally founded. "
        "Splitting them by any assumption would be inventing three numbers; `christianity."
        "other` would file them as bodies with no branch, which is the opposite of true. "
        "The merge gets its own node and the node says what is in it. See the node's note.",
    "Anglican":
        "-> christianity.anglican. 410,633 people, 2.3%. The Anglican Church of the "
        "Province of Central Africa, out of the Universities' Mission to Central Africa — "
        "and its geography is the reason it is worth having apart. **Likoma is 74.6% "
        "Anglican**, the highest denominational concentration of any district in Malawi and "
        "one of the sharpest single-body figures anywhere on this map: the UMCA put its "
        "cathedral on an island of 18 km² in 1903 and the geography has not moved. Behind "
        "it, **Ntchisi is 21.5% and Nkhotakota 15.3%** — the lakeshore mission stations — "
        "and then it falls off a cliff: Nkhata Bay is fourth at 4.2% and twenty-two "
        "districts are under 2%. Those three units hold 33.8% of Malawi's Anglicans on 4.1% "
        "of its people.",
    "Pentecostal":
        "-> christianity.pentecostal. 1,332,420 people, 7.6%. The Assemblies of God, the "
        "Living Waters Church, the Pentecostal Holiness Church and the newer independent "
        "charismatic bodies. NSO gives no branch and none is inferred.",
    "Other Christian Denominations":
        "-> christianity.other. 4,666,337 people, **26.6% — the largest cell in the "
        "country**, larger than the Catholics, and by some distance the biggest residual "
        "any source on this map hands over. That size is the honest cost of the question: "
        "NSO names five Christian bodies and pools everything else, so this cell holds the "
        "Zion churches, the Last Church of God and His Christ, the Church of Christ, the "
        "Jehovah's Witnesses, the Providence Industrial Mission, the African Methodists and "
        "the whole of Malawi's very large independent-church sector. "
        "**A great deal of it is `christianity.africaninstituted` and none of it can be "
        "moved there**, which is exactly the position gh2021.py was in before Kenya "
        "supplied that node — the difference being that Ghana's identically-named cell is "
        "12.3% and Malawi's is 26.6%, so Malawi is where the missing node costs the most. "
        "**Its geography says the cell is not uniform noise**: Nkhata Bay is 48.3% and "
        "Phalombe 48.0%, against 5.5% on Likoma and a national 26.6%. A residual that "
        "varies ninefold across districts is carrying something specific in the places it "
        "peaks (§9r's rule), and this source cannot say what. If a Malawian source is ever "
        "found that separates the Zion and Apostolic churches, this is the cell it would "
        "open, and the rows carry `source_category` so it can be done later (spec §2.4).",
    "Islam":
        "-> islam, with no branch, because the census gives none. 2,426,754 people, 13.8%, "
        "and **the most concentrated distribution in the country by a wide margin**: "
        "Mangochi is 72.7% Muslim and Machinga 67.0%, against 13.8% nationally, with Balaka "
        "34.7% and Salima 30.7% behind them and Chitipa in the far north at 0.08% — a "
        "900-fold range across 32 districts. This is the Yao lakeshore — the community "
        "converted "
        "through the 19th-century Swahili-Arab trade routes from Kilwa — and it is one "
        "contiguous block on the southern lake rather than a scatter. Overwhelmingly Sunni "
        "Shafi'i; nothing here separates the Qadiriyya and Sukuti traditions, and "
        "`islam.sunni` would be an inference rather than a reading.",
    "Traditional":
        "-> indigenous.african, the node Ghana added. 186,284 people, 1.06%. **Read it as a "
        "floor**, for the reason sources.md §11b gives for the whole continent: the cell is "
        "exclusive of the Christian and Muslim ones, and Malawian traditional practice — "
        "the Nyau societies of the Chewa above all, whose Gule Wamkulu is on UNESCO's "
        "intangible heritage list — commonly accompanies church membership rather than "
        "replacing it. A census that offers `Traditional` as a peer of `Catholic` cannot "
        "see anyone who is both, and in the Chewa centre a great many people are. "
        "Its drawn geography is still real and it is Dedza above everything else — **6.13%, "
        "50,938 people, nearly six times the national rate and more than a quarter of every "
        "traditionalist NSO counted**, on a district holding 4.7% of the population. "
        "Mzimba (3.25%) and Lilongwe (2.76%) follow; Likoma returns a bare zero.",
    "Other Denomination":
        "-> other.mw. 992,304 people, 5.65%. **It is known to contain 5,506 Buddhists and "
        "3,211 Hindus** — Table 3.4 splits the national figure three ways and Table E5 does "
        "not — and there is no table anywhere in the report that would place them, so they "
        "cannot be drawn apart. Per source, per spec §3.11; see the node's note.",
    "No Religion":
        "-> unaffiliated. 376,784 people, 2.15%. One cell, so nothing goes to `secular`, "
        "which needs a separately-counted atheist or humanist answer — the same call "
        "ke2019.py, hr2021.py and mk2021.py make. Malawi's is low — 2.15% against Ghana's "
        "4.50% and Kenya's 1.60% — and it has one real concentration: **Lilongwe district "
        "is 7.57%**, three and a half times the national rate and a third of every "
        "unaffiliated person in the country, while Lilongwe City beside it is 1.73%. That "
        "is a rural-district figure and not a capital-city one, which is the opposite of "
        "the usual shape and is worth not mis-reading.",
}

MAP = {
    "Catholic": "christianity.catholic",
    "CCAP": "christianity.reformed.presbyterian",
    "Seventh Day Adventist/Baptist/Apostolic": "christianity.sdabaptistapostolic",
    "Anglican": "christianity.anglican",
    "Pentecostal": "christianity.pentecostal",
    "Other Christian Denominations": "christianity.other",
    "Islam": "islam",
    "Traditional": "indigenous.african",
    "Other Denomination": "other.mw",
    "No Religion": "unaffiliated",
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
