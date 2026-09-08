"""Central African Republic RGPH03 2003 religion (USCB tabulation) -> religiondots taxonomy.

Five categories at commune, 3.84 million people, 177 units, 21,677 people each. Shallower
than every other African country drawn here except Malawi's four, and on much better
geography than most: Kenya is 1.01 million per unit (§9o), Zimbabwe 1.52 million (§9aj),
Ethiopia 99,900 (§9u). sources/cf.md is the source write-up.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.

**THE FORM HAS NO TRADITIONAL RELIGION BOX, AND THAT IS THE FACT THIS COUNTRY TURNS ON.** The
five cells are `Catholique`, `Protestante`, `Musulmane`, `Autre réligion`, `Sans réligion`.
Ghana offers `Traditionalist`, Kenya offers `Traditionalists`, Ethiopia offers `Traditional`,
Malawi offers `Traditional`, Benin offers both `Vodoun` and `Autres traditionnelles`. CAR
offers nothing. §11b's continental rule is that an exclusive traditional box UNDERCOUNTS
because traditional practice commonly accompanies a Christian or Muslim affiliation rather
than replacing it; CAR is the sharper case, where there is no box at all and the practice has
nowhere to go but into `Autre réligion`, into `Sans réligion`, or into one of the three named
religions. **None of those three destinations can be separated, so the map cannot draw CAR's
traditional religion and does not pretend to.** `indigenous.african` gets nothing from this
country — see the `Autre réligion` note, which is the argument for why not.

**There is no non-response category either, and 1.50% of the country is missing.** The
religion universe is 3,836,736 against the RGPH03 population of 3,895,139 — 58,403 people
counted and not in this table. That is not `Sans réligion`, which is its own published cell
of 136,850. It is simply absent, and cf.py reports it rather than filling it (§3.5).
"""

EXCLUDED = {}

REVIEW = {
    "Protestante":
        "-> christianity.protestant, the 'named no body' answer-node. 2,004,583 people, "
        "**52.25% — the largest Protestant share of any country drawn here**, and the "
        "census names not one denomination. CAR's Protestants are overwhelmingly the fruit "
        "of four mission fields that divided the country between them in the 1920s: the "
        "Baptist Mid-Missions and the Swedish Örebro mission in the north and centre, the "
        "American Grace Brethren (Église des Frères) in the Bangui region, and the "
        "Africa Inland Mission and Sudan United Mission in the east. So the cell is mostly "
        "Baptist and Brethren, and the tree could hold both apart — but the census offers "
        "one word and the answer-node is what the source supports. **Note it is a WIDER "
        "category than Ghana's or Kenya's identically-spelled ones**, which run Pentecostal "
        "and Evangelical cells alongside it; a CAR `Protestante` includes all of that. Two "
        "censuses using one word for different sets is exactly why `source_category` is "
        "kept verbatim (§2.4).",
    "Catholique":
        "-> christianity.catholic, the parent, and deliberately not `.latin`. 1,122,899 "
        "people, 29.27%. CAR's Catholic church is entirely Latin-rite — one archdiocese at "
        "Bangui and eight suffragan dioceses, no Eastern Catholic jurisdiction — so `.latin` "
        "would in fact be true here in a way it is not for Ethiopia (§9u). It is still not "
        "asserted, because the census says only `Catholique` and the parent is what the "
        "source supports; the tree does not gain from a distinction the publisher did not "
        "draw. Consistent with gh2021.py, ke2019.py, mw2018.py and bj2013.py, all of which "
        "send a bare Catholic cell to the parent.",
    "Musulmane":
        "-> islam, with no branch, because the census gives none. 400,962 people, 10.45%, "
        "and **the sharpest geography on this map's CAR**: Vakaga prefecture is 87.4% "
        "Muslim and Bamingui-Bangoran 44.5%, against 2.1% in Nana-Grébizi and 2.3% in "
        "Ouham. Ouandja commune is 96.1%. The population is Sunni and largely Maliki, and "
        "it is concentrated along the Chad and Sudan borders and in the trading towns — "
        "the Runga, Goula and Sara-Kaba of the north-east, the Hausa and Bornu merchant "
        "communities, and the Peulh/Mbororo herders. `islam.sunni` would be an inference "
        "rather than a reading. **The 2003 vintage matters more for this cell than for any "
        "other**: the 2013 Séléka/anti-balaka conflict displaced a large part of the Muslim "
        "population of the west and centre, and much of it has not returned. This map shows "
        "where they were, ten years before.",
    "Sans réligion":
        "-> unaffiliated. 136,850 people, 3.57%. A directly measured 'no religion' cell, "
        "which is what the node is for. **Its geography is the argument for reading it "
        "carefully**: it peaks at 17.7% in Basse-Batouri, 17.2% in Basse-Kadéi and 16.7% in "
        "Mongoumba — Lobaye and Mambéré-Kadéï, the south-western forest — and those are the "
        "same communes where `Autre réligion` peaks. Two residuals with one geography, in "
        "the part of the country where traditional practice is strongest and where the Aka "
        "live. Some unknown part of this cell is people with a traditional affiliation and "
        "no box to put it in, rather than people with no religion. Not corrected, because "
        "correcting it would mean inventing a magnitude (§14.4) — but the map's `note_public` "
        "says so, and so does the `other.cf` node.",
    "Autre réligion":
        "-> other.cf, and this is the consequential call. 171,441 people, 4.47%. **The "
        "tempting mapping is `indigenous.african` and it is refused.** The temptation is "
        "real: CAR's form has no traditional box, the country's traditional religions are "
        "documented and living, and this residual's geography is exactly theirs — 23.7% in "
        "Topia, 21.0% in Moboma and Baleloko, 20.2% in Carnot, all in Lobaye and "
        "Mambéré-Kadéï, the forest south-west, which is the Aka homeland and the Gbaya and "
        "Ngbaka country. Against a national 4.47%, a fivefold concentration in exactly the "
        "right place is not noise. "
        "**It is still refused, for two reasons.** First, the cell is a genuine residual "
        "and demonstrably holds other things: CAR has a long-established Bahá'í community "
        "(one of the first in central Africa), Jehovah's Witnesses, the Kimbanguists who "
        "cross from the Congo, and a small Greek and Lebanese Orthodox merchant presence in "
        "Bangui. Sending the whole cell to `indigenous.african` would count all of them as "
        "traditionalists. Second, and worse, the cell is a FLOOR and not an estimate: the "
        "traditional population that answered `Catholique` or `Protestante` is invisible "
        "here, and so is whatever part of it is inside `Sans réligion`. Mapping the "
        "residual to the tradition would therefore be both an overclaim about these 171,441 "
        "people and an undercount of the thing it claims to draw — wrong in both directions "
        "at once. **The honest statement is that CAR's traditional religion is not drawn**, "
        "which is what this mapping makes the map say. Per source, per spec §3.11.",
}

MAP = {
    "Catholique": "christianity.catholic",
    "Protestante": "christianity.protestant",
    "Musulmane": "islam",
    "Autre réligion": "other.cf",
    "Sans réligion": "unaffiliated",
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
