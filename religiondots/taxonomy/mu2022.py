"""Statistics Mauritius 2022 HPC religious group -> religiondots taxonomy.

Thirteen groups plus the universe total, on 182 wards and village council areas.

**FIVE OF THE THIRTEEN ARE HINDU, AND THAT IS THE REASON TO DRAW THIS COUNTRY.** Before
Mauritius the tree divided Hinduism not at all — one family node and Vietnam's Cham Balamon
under it — because no source anywhere on this map splits it. India's census does not.
Guyana's, which is 24.8% Hindu, does not. Mauritius counts Marathi, Tamil, Telugu and Arya
Samaj Hindus separately, at village level, in a country that is 47.9% Hindu, and four nodes
were added to branches.py for it.

**THE PRICE IS PAID ON THE CHRISTIAN SIDE AND IT IS STEEP.** Table D5, eight pages earlier
in the same report, names sixty-odd individual bodies at island level — `La Voix de la
Delivrance`, `Peniel Tabernacle`, `Full Gospel Church`, `Christian Tamil`, `Church of
England`, `Presbyterian`, `Methodist`, `Mormon`. D6 pools all of them into four cells, one
of which is `Other Christian` at 6.2%. §3.9's trade, made by the office, and this file draws
the geography half of it. See sources/mu.md §3.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
}

REVIEW = {
    "Hindu & Other Hindu":
        "-> hinduism, the family node itself. **474,623 people, 38.5% of the country and "
        "80.4% of its Hindus.** This is the Bhojpuri-descended Sanatanist majority, the "
        "people the other four Hindu cells are counted apart FROM, and the census gives it "
        "no name of its own — `Hindu & Other Hindu` is the unspecified answer plus the "
        "tail. It goes to the parent for the same reason ke2019.py sends `Orthodox` to "
        "`christianity.orthodox`: the node that asserts the least. Note that this makes "
        "Hinduism a family whose own dots are four fifths of it, which is the opposite of "
        "the §6.15 shape and may want looking at in the legend.",
    "Tamil/Tamil Hindu":
        "-> hinduism.tamil, a node added for Mauritius. 63,950 people, 5.2%. Shaivite and "
        "Murugan-centred, with its own kovils and priesthood; Cavadee is a public holiday. "
        "**It is a southern and central community, and it is NOT a Port Louis one** — "
        "Savanne is 8.2% and Plaines Wilhems 7.1%, while Port Louis is 3.7%, below the "
        "national rate, which is the opposite of what the usual account of an early-arriving "
        "urban community would predict. The unit peak is St Julien d'Hotman VCA-East at "
        "18.0%.",
    "Telugu/Telugu Hindu":
        "-> hinduism.telugu, a node added for Mauritius. 25,216 people, 2.0%. Andhra-origin, "
        "under indenture, organised since 1947 around the Andhra Maha Sabha. Kept apart "
        "from Tamil because Mauritius keeps them apart and because the temples, languages "
        "and calendars differ — reading the two as one 'South Indian' block is exactly the "
        "error the split exists to prevent.",
    "Marathi/Marathi Hindu":
        "-> hinduism.marathi, a node added for Mauritius. 19,052 people, 1.5%. "
        "Maharashtrian-origin; Ganesh Chaturthi is its principal festival. **The tightest "
        "cluster of the four and it is the southwest coast**: La Gaulette is 27.7% and Baie "
        "du Cap 27.0%, eighteen times the national rate, on a stretch of Black River and "
        "Savanne coastline where nothing else on this map is distinctive at all.",
    "Vedic/Hindu Vedic & Aryan":
        "-> hinduism.aryasamaj, a node added for Mauritius. 7,422 people, 0.6%, and **the "
        "only one of the four that is a movement rather than a community.** Dayananda "
        "Saraswati's 1875 reform — Vedas alone, no image worship, no caste birth-right — "
        "which reached Mauritius in 1910 and split Mauritian Hinduism bitterly enough to "
        "shape its politics for decades. A Bhojpuri-speaking Arya Samajist belongs here and "
        "not in the parent, which is why it sits beside the three community nodes rather "
        "than inside one.",
    "Roman Catholic":
        "-> christianity.catholic, the parent rather than .latin. 307,515 people, 24.9%, and "
        "the single largest named body in Mauritius. The Creole and Franco-Mauritian "
        "population plus a large share of the Sino-Mauritians; Latin rite throughout and the "
        "census does not say so. **AND IT IS WHY RODRIGUES IS WORTH THE 600 KM OF EMPTY SEA "
        "IT COSTS THE MAP.** All six Rodriguan regions run 84.9%-91.9% Catholic against 24.9% "
        "nationally, and Rodrigues is 0.5% `Hindu & Other Hindu` against 38.5%: a Creole "
        "Catholic island inside a Hindu-majority republic, and the sharpest internal contrast "
        "any country on this map contains.",
    "L'Assemblee de Dieu / M.S et Guerison":
        "-> christianity.pentecostal. 11,357 people, 0.92%. Two bodies in one cell and both "
        "are Pentecostal — the Assemblies of God, and Mission Salut et Guerison, a Mauritian "
        "healing church — so unlike Malawi's three-way merge this one does not cross a "
        "family boundary and needs no answer-node. D5 splits them 10,046 / 1,311 nationally.",
    "Church of England/Protestant":
        "-> christianity.protestant, the 'named no body' node. 2,457 people, 0.20%. The cell "
        "merges the Anglicans with an unspecified Protestant remainder, so it crosses the "
        "line between a named church and an answer — but at this size, and with D5 showing "
        "only 247 Church of England nationally, sending it to `christianity.anglican` would "
        "assert far more than the merge does. The node whose whole purpose is 'Protestant, "
        "no body named' is the honest destination.",
    "Other Christian":
        "-> christianity.other. 76,883 people, 6.2%, and **the cost of the geography half of "
        "§3.9's trade.** D5 names what is in it and D6 does not: La Voix de la Delivrance, "
        "Full Gospel Church, Peniel Tabernacle, the Adventists, the Witnesses, the "
        "Presbyterians, the Methodists, `Christian Tamil`, `Eglise Chretienne` and forty "
        "more, none of them separable at any geography. A source that gave both at once "
        "would be the best Christian detail in the southern hemisphere; this one gives "
        "either.",
    "Buddhist/Chinese":
        "-> chinesefolk, and this is the most arguable call in the file. 5,053 people, "
        "0.41%. **D5 splits it nationally — Buddhist 2,178, Chinese 2,434, Other Chinese "
        "441 — and D6 does not**, so the split cannot be drawn. The cell is the "
        "Sino-Mauritian community that kept its ancestral religion, and Sino-Mauritian "
        "practice is the ordinary Chinese combination of Mahayana Buddhism, Guanyin and "
        "ancestor observance rather than two separable things; §3.3 says syncretism gets a "
        "node rather than a split, and `chinesefolk` is that node. "
        "**What is wrong with it, stated rather than hidden**: `chinesefolk`'s own note "
        "calls it 'China's own tradition', and 2,178 people here answered `Buddhist` and are "
        "now drawn as Chinese religion. The alternative — sending all 5,053 to `buddhism` — "
        "misfiles 2,875 the other way and denies the folk practice outright. Both are wrong "
        "and this one is wrong about fewer people.",
    "Islam/Muslim & Other Muslim":
        "-> islam, with no branch, because D6 gives none. 224,885 people, 18.2%. Mauritian "
        "Muslims are overwhelmingly Sunni Hanafi of Gujarati and Bihari descent, with a "
        "small Ahmadi community that D5 counts separately at island level and D6 does not; "
        "`islam.sunni` would be an inference rather than a reading. **Port Louis is 40.9% "
        "against 18.2% nationally, and inside it Ward 5 is 96.81%** — 17,058 people, and the "
        "most nearly total single-religion ward anywhere on this map outside Sulu and the "
        "Kenyan north-east. Moka (24.2%) and Savanne (21.7%) follow; Rodrigues is 0.9%.",
    "No religion":
        "-> unaffiliated. 7,753 people, 0.63% — **the lowest no-religion share of any "
        "country on this map**, below Ghana's 4.50% and Malawi's 2.15%. One cell, so nothing "
        "goes to `secular`. What geography it has is the west-coast resort strip — Flic en "
        "Flac 4.93%, Tamarin 4.31%, La Gaulette 3.94%, all in Black River — which is a "
        "resident-foreigner pattern rather than a Mauritian one.",
    "Other & Not stated":
        "-> other.mu. 6,931 people, 0.56%. **A residual that knowingly mixes an answer with "
        "a non-answer**, which no other source here does: every other census keeps `not "
        "stated` apart and §3.5 takes it off the tree, and Mauritius pools them with no "
        "split at any geography. Drawn, and read as a ceiling on Mauritius's other religions "
        "rather than a count of them. See the node's own note.",
}

MAP = {
    "Buddhist/Chinese": "chinesefolk",
    "L'Assemblee de Dieu / M.S et Guerison": "christianity.pentecostal",
    "Church of England/Protestant": "christianity.protestant",
    "Roman Catholic": "christianity.catholic",
    "Other Christian": "christianity.other",
    "Marathi/Marathi Hindu": "hinduism.marathi",
    "Tamil/Tamil Hindu": "hinduism.tamil",
    "Telugu/Telugu Hindu": "hinduism.telugu",
    "Vedic/Hindu Vedic & Aryan": "hinduism.aryasamaj",
    "Hindu & Other Hindu": "hinduism",
    "Islam/Muslim & Other Muslim": "islam",
    "No religion": "unaffiliated",
    "Other & Not stated": "other.mu",
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
