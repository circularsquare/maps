"""Côte d'Ivoire RGPH 2021 religion (ANStat tome 1) -> religiondots taxonomy.

Nine cells at région administrative, 29.28 million people in ordinary households, 33 units,
about 887,000 each — Kenya's grain (§9o). sources/ci.md is the source write-up.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

**THE COUNTRY IS DRAWN FOR ONE CELL AND IT IS `Harriste`.** 140,482 people, 0.48%, and it
adds the first node this map has ever had for an African Initiated Church named individually
by a census outside Benin. Everything else here is ordinary; that cell is not.

**AND THE SECOND LARGEST GROUP IN THE COUNTRY IS A RESIDUAL.** `Autres religions chrétiennes`
is 6,004,781 people, 20.5%, undivided, and the volume's own methodology says the census
collected `Evangélique`, `Céleste`, `Bouddhiste` and `Témoin de Jehova` as separate
modalities. None of the four is printed anywhere — not nationally, not by region, not by age.
So this map draws a Celestial Church geography in Benin (§9ai, 676,032 people at commune
level) and cannot draw one in the country next door, where the same church is certainly
present and is inside this cell. §3.9's trade at its worst.
"""

EXCLUDED = {
    "Total":
        "the unit's own population total, not a category.",
    "Ensemble Chrétien":
        "a SUBTOTAL, not a category. It equals Catholique + Méthodiste/Protestant + "
        "Harriste + Autres religions chrétiennes on every one of the 46 printed rows — "
        "checked in sources/ci.py, which asserts the identity within 0.20 pp rather than "
        "assuming it. Drawing it as well would count 40.3% of the country twice. Ghana's "
        "`Christian` case (gh2021.py) exactly.",
    "ND":
        "`Non déclaré` — 646,044 people, 2.21%. A NON-ANSWER, not an answer, so it comes "
        "off the tree per spec §3.5, as Kenya's `Not Stated` and `Don't Know` do "
        "(ke2019.py). It is kept apart from `Sans religion`, which is 5.7x larger and is a "
        "real answer. countries.py carries a `gap` line for it, because a blank on a dot "
        "map cannot otherwise distinguish 'nobody here' from 'nobody counted here'.",
}

REVIEW = {
    "Harriste":
        "-> christianity.africaninstituted.harrist, A NEW NODE, and the reason this country "
        "is drawn. 140,482 people, 0.48% — small, and **no other source on this map counts "
        "Harrism at all**. The Harrist Church follows **William Wadé Harris**, the Grebo "
        "preacher from Liberia who walked the lagoon coast of Côte d'Ivoire and the Gold "
        "Coast in 1913-15 in a white robe carrying a cane cross, and who is usually credited "
        "with more conversions than any missionary in African history — something over a "
        "hundred thousand people in eighteen months, most of whom the Catholic and Methodist "
        "missions inherited. What remained organised as the Église Harriste is one of Côte "
        "d'Ivoire's state-recognised confessions and is still concentrated exactly where he "
        "walked. "
        "**It is a child of `christianity.africaninstituted` rather than a sibling**, "
        "because that is precisely what it is: a church founded in Africa by an African, "
        "belonging to no mission denomination, and predating almost every other body on that "
        "node — Harris was preaching two decades before the Vapostori and before the "
        "Celestial Church. spec §2.4's rule is that a node earns its place by being "
        "countable somewhere, and Côte d'Ivoire counts it at régional level. "
        "**Its geography is the check, and it passes**: 2.4% in La Mé, 1.7% in "
        "Grands-Ponts, 1.6% in Agnéby-Tiassa, 1.5% in Moronou and 1.4% in Lôh-Djiboua — the "
        "southern lagoon belt — against **a printed 0.0% in seven régions: Bafing, Folon, "
        "Hambol, Kabadougou, Poro, Tchologo and Worodougou**, every one of them northern "
        "savannah, which Harris never reached. A twenty-four-fold range across the country "
        "and a clean north/south line. That is the 1913-15 itinerary, still legible in a "
        "2021 census.",
    "Méthodiste/Protestant":
        "-> christianity.protestant, the 'named no body' answer-node, and NOT "
        "`christianity.methodist`. 678,962 people, 2.32%. The cell merges a named body with "
        "an unnamed category, which is the awkward shape: the Église Méthodiste Unie Côte "
        "d'Ivoire is the country's largest mainline Protestant church and is certainly the "
        "bulk of this cell — it is the direct institutional heir of the Harris movement, "
        "which is why Methodism is unusually large in the south-east — but the cell also "
        "holds every respondent who said only *protestant*. Sending all 679,000 to "
        "`christianity.methodist` would file those people as Methodists on the strength of a "
        "slash; sending them to the answer-node asserts what the census actually "
        "established, which is that they are Protestant and not further specified. The same "
        "call as mw2018.py's `SDA/Baptist/Apostolic`, made in the direction of the less "
        "specific node because here one of the two halves is already the generic one.",
    "Autres religions chrétiennes":
        "-> christianity.other. **6,004,781 people, 20.51% — the second largest religious "
        "group in Côte d'Ivoire and the largest undivided cell on the African part of this "
        "map.** The tome's own methodology (p80) lists the modalities the census collected: "
        "*Catholique, Méthodiste/Protestante, Evangélique, Céleste, Harriste, Musulman, "
        "Animiste, Bouddhiste et Témoin de Jehova*. Four of those — **Evangélique, Céleste, "
        "Bouddhiste, Témoin de Jehova** — are given a cell in no table in the volume, so "
        "they are inside this one, and Bouddhiste is inside it despite not being Christian "
        "at all under any reading. "
        "The bulk of it is Côte d'Ivoire's very large evangelical and Pentecostal sector, "
        "which would go to `christianity.evangelical` or `christianity.pentecostal` if the "
        "office had printed it; **the Celestial Church of Christ is also in here**, and "
        "bj2013.py draws that same church by name at commune level next door. It goes to "
        "`christianity.other` — bodies the source named as Christian and did not place — "
        "rather than to any of the three, because choosing one would invent a division "
        "ANStat did not make. Per spec §3.11, and it is the biggest single thing this "
        "country cannot show.",
    "Animiste":
        "-> indigenous.african, the node Ghana added. 629,938 people, 2.15%. **Read it as a "
        "floor**, per §11b's continental rule: the box is exclusive of `Catholique` and "
        "`Musulmane`, and in Côte d'Ivoire traditional practice very commonly accompanies "
        "one of them rather than replacing it. "
        "Its geography is the north-east and it is concentrated hard: **Bounkani 24.7%** — "
        "one region an order of magnitude above the national figure — with Béré at 6.1%, "
        "Poro 7.7% and the Zanzan district 13.5%, against 0.3% in the Abidjan district. "
        "Bounkani is Lobi and Koulango country on the Burkinabè and Ghanaian border, and it "
        "is the one place in the country where the census sees the tradition clearly.",
    "Sans religion":
        "-> unaffiliated. 3,685,173 people, 12.59%, directly measured. **Its geography does "
        "not look like secularisation and that is worth saying**: it is 27.0% in La Mé, "
        "29.8% in Tonkpi, 28.9% in Poro and 26.1% in Bounkani, against **3.6% in the "
        "district of Abidjan** — the exact inverse of where a secularising urban population "
        "would be. Some unknown part of this cell is traditional practice with no church and "
        "no box, the same reading bj2013.py gives Benin's Atacora `Aucune`. Drawn as "
        "published; not corrected, because correcting it would mean inventing a magnitude "
        "(§14.4).",
    "Autres religions":
        "-> other.ci. 53,051 people, 0.18%, and a genuinely small tail rather than a hidden "
        "large group — which is unusual, and is only true because `Autres religions "
        "chrétiennes` above is absorbing everything that would normally end up here. Per "
        "source, per spec §3.11.",
}

MAP = {
    "Catholique": "christianity.catholic",
    "Méthodiste/Protestant": "christianity.protestant",
    "Harriste": "christianity.africaninstituted.harrist",
    "Autres religions chrétiennes": "christianity.other",
    "Musulmane": "islam",
    "Animiste": "indigenous.african",
    "Autres religions": "other.ci",
    "Sans religion": "unaffiliated",
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
