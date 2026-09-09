"""NISR RPHC-5 2022 religion -> religiondots taxonomy.

Eleven categories plus the universe total, on 30 districts. The list is conventional for an
African census in every respect but one: **`ADEPR` is a named church**, and at 2,820,813
people, 21.29%, it is the second-largest religious answer in Rwanda.

    39.91%  Catholic              -> christianity.catholic
    21.29%  ADEPR                 -> christianity.pentecostal
    14.56%  Protestant            -> christianity.protestant
    12.17%  Adventist             -> christianity.adventist
     4.18%  Other Christians      -> christianity.other
     3.04%  No Religion           -> unaffiliated
     2.00%  Muslim                -> islam
     2.00%  Other religion        -> other.rw
     0.70%  Jehovah witness       -> christianity.witnesses
     0.13%  Not stated            -> EXCLUDED (non-response, the §3.5 gap)
     0.02%  Traditional/Animist   -> indigenous.african

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the district's own population total, not a category.",
    "Not stated":
        "non-response. 17,785 people, 0.13% — the smallest §3.5 residual of any census on "
        "this map that has one at all, and it is urban where it is anything: Kicukiro "
        "0.24% and Gasabo 0.22% against 0.06% in Nyamasheke. Not drawn, and reported as "
        "the country's `gap`.",
}

REVIEW = {
    "Catholic":
        "-> christianity.catholic, the parent rather than .latin. 5,286,003 people, "
        "39.91%, and the largest answer in 27 of the 30 districts. The other three are "
        "Karongi and Nyanza, where the Adventists are, and Nyamasheke, where ADEPR is. "
        "Latin rite throughout "
        "and the census does not say so, which is the call ke2019.py, mw2018.py, bj2013.py "
        "and zw2022.py all make. **Its geography is the central plateau and the north**: "
        "Muhanga 65.02%, Rulindo 62.49%, Gakenke 60.77%, against 21.10% in Karongi and "
        "26.47% in Nyamasheke on Lake Kivu. That is the White Fathers' mission field of "
        "1900-1930 still legible a century later, and the two provinces that sit on it "
        "(Northern 51.59%, Southern 49.07%) are the two where Catholicism is still a "
        "majority.",
    "ADEPR":
        "-> christianity.pentecostal, and **this is the category worth drawing Rwanda "
        "for**. 2,820,813 people, 21.29%, and a single named denomination rather than a "
        "family: the **Association des Églises de Pentecôte du Rwanda**, which grew out of "
        "the Swedish Free Mission that reached the country from the Belgian Congo in 1940 "
        "and took its present name in 1983. **Nothing else on this map has one church "
        "counted as a census category at a fifth of a country.** The UNSD Demographic "
        "Yearbook prints the identical figure under the generic label `Pentecostal`, which "
        "is how NISR described it to the UN; the district profiles name it. "
        "**Its geography is the story of the mission and it checks out**: ADEPR's first "
        "congregation was in Rusizi, on the Congo border, and Rusizi is still its "
        "strongest district at 33.90%, with Nyamasheke next door at 30.70%. It falls away "
        "eastward across the Catholic plateau to 11.12% in Nyamagabe and 11.74% in "
        "Gakenke. Gasabo, the largest district in the country, is 29.14%. "
        "**No branch node.** ADEPR is a classical trinitarian Pentecostal body, but "
        "`christianity.pentecostal.trinitarian`'s children are all United States "
        "denominations and the census says only the name; zw2022.py's `Pentecost` makes "
        "the same call for the same reason. A node of its own was considered and not "
        "taken: it would add a legend row no other country uses, which AGENT_BRIEF §3 "
        "sends to Anita, and this cell reads correctly as Pentecostal without it.",
    "Protestant":
        "-> christianity.protestant, which holds an ANSWER and not a church. 1,928,741 "
        "people, 14.56%. In Rwanda this is the mission inheritance with the Pentecostals "
        "and the Adventists already lifted out of it: the Anglican Church of Rwanda, the "
        "Presbyterian Church in Rwanda, the Free Methodists, the Baptist associations and "
        "the Evangelical Friends of Gisenyi and Kigali. **No branch is inferred** — those "
        "bodies sit in five different places on the tree and the census gives none. "
        "**Its geography is the west and the far north**: Nyamagabe 27.65%, Karongi "
        "26.64%, Burera 25.40%, against 4.79% in Rulindo and 7.01% in Nyarugenge. Rulindo "
        "is 62.49% Catholic and 4.79% Protestant in the same district.",
    "Adventist":
        "-> christianity.adventist, the parent rather than .sda. 1,612,482 people, 12.17%, "
        "which is a very large Adventist share by any standard and the third-largest "
        "Christian answer in the country. The body is the Seventh-day Adventist Church and "
        "the census does not say so, which is why the parent is used; ag2001.py, "
        "bb2010.py, ck2011.py and ao2024.py all take the same line on the same wording. "
        "**Its geography is a tight belt through the middle of the country**: Nyanza "
        "33.94%, Karongi 27.04%, Ruhango 26.67%, Nyabihu 26.48% — and then 2.38% in "
        "Gicumbi, 3.46% in Rusizi and 3.48% in Rulindo. A fourteen-fold range across 30 "
        "districts, the widest of the four large Christian cells (Catholic 3.1x, ADEPR "
        "3.0x, Protestant 5.8x), **and it is the clearest thing the district grain buys**: "
        "at province level the same figures run only 9.48% to 14.58% and the belt is gone.",
    "Other Christians":
        "-> christianity.other. 553,174 people, 4.18%. The residual left after four named "
        "Christian cells and the Witnesses, so it holds the newer independent and "
        "charismatic congregations, the Orthodox parishes of Kigali, the Latter-day "
        "Saints and the restorationist churches. **Its geography is Kigali and the east**: "
        "Kicukiro 9.98%, Nyagatare 8.75%, Kayonza 8.67%, against 0.72% in Gakenke and "
        "0.84% in Gisagara — the inverse of the Catholic map, and the shape a young urban "
        "and resettlement-area religion has.",
    "Muslim":
        "-> islam, with no branch, because the census gives none. 265,317 people, 2.00%. "
        "Rwandan Islam is overwhelmingly Sunni of the Shafi'i school and is historically "
        "the religion of the Swahili trading quarters, which is exactly where the census "
        "finds it: **Nyarugenge is 11.31%**, nearly three times the next district and the "
        "only one above 4%, and Nyarugenge "
        "contains Nyamirambo. Then Kicukiro 3.96%, Rubavu 3.87% and Gatsibo 3.23%, and "
        "0.17% in Nyamasheke. Nothing here separates the older Muslim quarters from the "
        "post-1994 growth the literature describes, and nothing should.",
    "Jehovah witness":
        "-> christianity.witnesses. 93,131 people, 0.70%. Urban and north-western — Rubavu "
        "1.49%, Nyarugenge 1.27%, Kicukiro 1.07% — against 0.18% in Nyamasheke and "
        "Ruhango. NISR's spelling is kept in `source_category` per §2.4.",
    "Traditional/Animist":
        "-> indigenous.african. **2,112 people, 0.02%, and it is the smallest cell in the "
        "table by a factor of eight.** Two thousand people in a country of 13.2 million, "
        "against Zimbabwe's 5.02% in the same box and Botswana's Badimo at 4.1%. "
        "**Read it as a floor, and here more than anywhere.** §11b's standing point about "
        "the whole continent is that the box is exclusive of the Christian ones, and "
        "Rwanda's *kubandwa* — the Ryangombe possession cult, consultation of an "
        "*umupfumu* — has for a century been something people do alongside church "
        "membership rather than instead of it. The state has also discouraged it by name "
        "since the 1990s. A census asking which religion you belong to will not find it, "
        "and this figure should be read as the count of people who answered nothing else. "
        "**Its geography is Kigali**, of all places — Gasabo and Kicukiro at 0.03% against "
        "0.00% in Ruhango and Gisagara — which is one more reason not to read the cell as "
        "a measure of traditional practice. No child node: no source names a Rwandan "
        "tradition individually (§2.4).",
    "Other religion":
        "-> other.rw. 264,319 people, 2.00%. See the node's own note: it is the same size "
        "as the Muslim cell, which is unusual for a residual sitting beside boxes for "
        "Islam and traditional practice, and NISR publishes no breakdown of it. Per §3.11.",
    "No Religion":
        "-> unaffiliated. 402,517 people, 3.04%. One cell, so nothing goes to `secular`, "
        "which needs a separately-counted atheist or humanist answer — the call ke2019.py, "
        "mw2018.py, bj2013.py and zw2022.py all make. **It has almost no geography**, "
        "which is itself the finding: 4.16% in Nyamagabe down to 1.47% in Muhanga, a "
        "range of under three points across 30 districts and the flattest column in the "
        "table. It is not an urban figure — Kigali's three districts sit at 3.38% "
        "province-wide, near the national 3.04% — so the secularising-city reading that "
        "fits Europe does not fit here.",
}

MAP = {
    "Catholic": "christianity.catholic",
    "ADEPR": "christianity.pentecostal",
    "Protestant": "christianity.protestant",
    "Adventist": "christianity.adventist",
    "Other Christians": "christianity.other",
    "Muslim": "islam",
    "Jehovah witness": "christianity.witnesses",
    "Traditional/Animist": "indigenous.african",
    "Other religion": "other.rw",
    "No Religion": "unaffiliated",
}

# spec §7a-i-1: the level this country MEASURED each node at, so a dot inferred below one
# rolls up to the source's own column instead of vanishing. RPHC-5 measures every one of
# these directly, at district, so each node's target is itself — ao2024.py's form. It does
# nothing today, because Rwanda has no `derived` rows at all and `tools/check_rollup.py`
# reports 13,228,609 measured and zero of everything else; it is here so that a later
# source adding a split below one of these nodes has somewhere to fall back to.
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
