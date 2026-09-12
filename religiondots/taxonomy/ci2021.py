"""Côte d'Ivoire RGPH 2021 religion (ANStat tome 1) -> religiondots taxonomy.

Nine cells at région administrative, 29.28 million people in ordinary households, 33 units,
about 887,000 each — Kenya's grain (§9o). sources/ci.md is the source write-up.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

**THE COUNTRY IS DRAWN FOR ONE CELL AND IT IS `Harriste`.** 140,482 people, 0.48%, and it
adds the first node this map has ever had for an African Initiated Church named individually
by a census outside Benin. Everything else here is ordinary; that cell is not.

**AND THE SECOND LARGEST GROUP IN THE COUNTRY WAS A RESIDUAL, UNTIL A SECOND PUBLICATION
NAMED MOST OF IT.** Tome 1's `Autres religions chrétiennes` is 6,004,781 people, 20.5%, and
that volume never divides it. The *Résultats Globaux Définitifs* does, in one sentence of
prose: *"20% d'autres chrétiens, composés principalement des évangéliques (18,6%)"*. So
5,445,459 of them are **évangéliques**, and sources/ci.py §7a splits the cell on that figure.
The magnitude is the source's; the geography is not, and the `Évangélique` note below says
what that costs.

What is left after the split is 559,322 people, and the volume's methodology names three more
modalities the census collected and no table prints — `Céleste`, `Bouddhiste`, `Témoin de
Jehova`. **The Celestial Church of Christ is inside that remainder**, and bj2013.py draws
that same church by name at commune level next door in Benin. §3.9's trade, still.
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
    "Évangélique":
        "-> christianity.evangelical, DERIVED. **5,445,459 people, 18.60%, and the second "
        "largest religious group in Côte d'Ivoire.** It is not a cell in tome 1's tables at "
        "all; it is §7a's split of `Autres religions chrétiennes`, taken from ANStat's own "
        "prose in the *Résultats Globaux Définitifs*: *\"20% d'autres chrétiens, composés "
        "principalement des évangéliques (18,6%)\"*. Those percentages are of the total "
        "population — they reproduce that publication's own % column exactly — so the "
        "magnitude is the source's rather than this project's. "
        "**The GEOGRAPHY is not established and the mapping must not be read as if it "
        "were.** No ANStat publication gives évangéliques by région, so the national ratio "
        "is applied uniformly and this node inherits the residual's shape exactly. "
        "Evangelicals in Côte d'Ivoire are very likely more southern and more urban than a "
        "flat 90.7% of every région's other-Christian cell, so the per-région shares are "
        "wrong in a spatially correlated way while the national total is right. Every row "
        "is `tier=\"derived\"` and can never ring (§3.10), and note_public says so. "
        "It goes to `christianity.evangelical` — the answer-node Kenya added for KNBS's "
        "`Evangelical Churches` — rather than to `christianity.pentecostal`, because "
        "ANStat's word is *évangélique* and the Ivorian sector it names covers both the "
        "classical evangelical missions and the newer Pentecostal assemblies without "
        "separating them.",
    "Autres chrétiens, hors évangéliques":
        "-> christianity.other, and it is what is LEFT after the évangélique split: "
        "**559,322 people, 1.91%.** Small, and still doing a lot of work. The tome's "
        "methodology (p80) names four modalities the census collected and no table ever "
        "prints — *Evangélique, **Céleste**, **Bouddhiste**, **Témoin de Jehova*** — and "
        "with the évangéliques now lifted out, the other three are the bulk of what "
        "remains. **The Celestial Church of Christ is in here**, and bj2013.py draws that "
        "same church by name in every one of Benin's 77 communes, so this map can show a "
        "Céleste geography in one country and not in the country next door. Bouddhiste is "
        "in here too, despite not being Christian under any reading. "
        "**It is also where a disagreement between two ANStat publications lands.** Tome 1 "
        "puts 159,208 more people in `autres chrétiens` than the Résultats Globaux does, "
        "and the two agree on `autres chrétiens + autres religions` to the person. Because "
        "the évangélique count is subtracted from tome 1's larger cell, those disputed "
        "people fall into this remainder rather than into the évangéliques — which is where "
        "they belong if the Résultats Globaux is right that they are not Christian. Per "
        "spec §3.11.",
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
    # §7a's split of `Autres religions chrétiennes`. sources/ci.py emits these two in its
    # place; the undivided category no longer reaches the tree, and is kept below so a
    # re-run against an older ci.csv still resolves rather than failing silently.
    "Évangélique": "christianity.evangelical",
    "Autres chrétiens, hors évangéliques": "christianity.other",
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
