"""Afrobarometer Tanzania religion -> religiondots taxonomy.

Eleven categories, at 30 units (the 31 regions of the 2022 census with Songwe inside Mbeya's
unit). `Christian` and `Muslim` are the card's umbrella answers with their denominational children
folded back in; since 2026-10-03 the mainland's Christians are split into five churches and
`Other Christian` from the GFS 2023 and the Afrobarometer together (`sources/tz.md` §9), and
`Christian` remains for Zanzibar only. `sources/tz.py`'s docstring has the argument
for the folding and `sources/tz.md` the acquisition record. The same construction as
`ng2022.py`, on the same instrument, and the same ruling (ask/answered/010-ng).

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Christian":
        "-> christianity, the bare branch. **Zanzibar only** since 2026-10-03: its five regions' "
        "Christians are one pooled share (Anita's ruling) from 13 respondents, and the GFS has two "
        "Christians there, so they are not split by church. On the mainland the Christians are "
        "split into the five churches below and `Other Christian`.",
    "Roman Catholic":
        "-> christianity.catholic, the bare branch, as `na2023.py` and `zm2022.py`. 27.6% of "
        "Tanzania, 43.1% of mainland Christians as drawn; the GFS 2023 has 43.2% and the "
        "Afrobarometer names 42.1% (a floor). Ruvuma 79.9% of Christians (Songea), Rukwa 68.5%, "
        "Katavi 61.2%; Dodoma 19.6%. Rank over 25 mainland units, GFS against Afrobarometer, "
        "+0.675 (null 95th +0.335). The level is the Afrobarometer's floor: the GFS's level is "
        "0.13 points of Christians under it, so Catholics take none of the `Christian only` answers.",
    "Lutheran":
        "-> christianity.lutheran (the Evangelical Lutheran Church in Tanzania, the only large "
        "Lutheran body). 8.5% of Tanzania, 13.2% of mainland Christians; GFS 14.4%, Afrobarometer "
        "floor 10.8%. Arusha 49.2% of Christians, Njombe 36.6%, Tanga 35.1%, Kilimanjaro 33.0%. "
        "Rank +0.764. Moravians have no box on the GFS card and some may answer Lutheran.",
    "Anglican":
        "-> christianity.anglican. 6.3% of Tanzania, 9.9% of mainland Christians; GFS 9.8%, "
        "floor 6.8%. Dodoma 54.6% of Christians (the Diocese of Central Tanganyika), Kigoma 22.9%, "
        "Singida 16.5%; Mtwara 11.0% (Masasi), where the GFS's 33 Christians found 4% and the "
        "Afrobarometer 16%. Rank +0.565.",
    "Pentecostal":
        "-> christianity.pentecostal. 13.9% of Tanzania, 21.7% of mainland Christians. **The "
        "largest gap between the surveys**: the GFS's card says Pentecostal/Charismatic and 21.6% "
        "of its Christians take it; the Afrobarometer names 8.6%, and its `Christian only` (21.1% "
        "of Christians) has to be 61.5% Pentecostal for the two to agree. Drawn at the GFS's level "
        "because it is the survey where almost everyone named a church. Mbeya and Songwe 34.9% of "
        "Christians (GFS 45%, Afrobarometer 24%), Kigoma 32.9%, Pwani 33.2%. Rank +0.584.",
    "Seventh Day Adventist":
        "-> christianity.adventist.sda. 3.9% of Tanzania, 6.1% of mainland Christians; GFS 6.1%, "
        "floor 4.0%. Simiyu 21.5% of Christians, Mara 19.7% (the church's Lake Victoria field). "
        "**The weakest pass**: rank +0.396 against a null 95th of +0.333 (p 0.024); the GFS's "
        "Kilimanjaro 14% is not in the Afrobarometer (3%).",
    "Other Christian":
        "-> christianity, bare. The GFS's Independent/Holiness/Evangelical (154, mostly Mwanza and "
        "Simiyu), Baptist (85), Presbyterian, Jehovah's Witness, Latter-day Saints, Methodist and "
        "other answers; the Afrobarometer's Independent, Evangelical, Baptist, Mennonite, Moravian "
        "and the rest. 5.9% of mainland Christians. Not spread into: the Afrobarometer names more "
        "of it than the GFS. Moravians (the south-west) are here or in Lutheran.",
    "Muslim":
        "-> islam, with no branch. Tanzanian Islam is mostly Sunni (Shafi'i on the coast and "
        "in Zanzibar), with Khoja Shia Ithna'ashari, Bohra and Ismaili communities in Dar es "
        "Salaam and Zanzibar town and the Qadiriyya and Shadhiliyya brotherhoods. The survey's "
        "`Sunni only` box runs from 0.4% to 4.7% of respondents between rounds and its `Muslim "
        "only` box was re-worded in round 8, so a pooled branch share would measure the "
        "questionnaire. Not drawn.",
    "None":
        "-> unaffiliated. 4.5% of the pooled survey and 5.23% as drawn, and it carries its own "
        "geography (split-half +0.852): Shinyanga 24.0%, Simiyu 23.5%, Tabora 16.8% and Geita "
        "16.5%, which is Sukuma and Nyamwezi country, and almost none on the coast or in "
        "Zanzibar. **Not §9dn's `unknown`**, and the difference is the card: Mozambique's and "
        "Laos's census boxes took traditional religion by their own wording, while this card "
        "offers `Traditional/ethnic religion` as a separate box on every drawn round "
        "(`report_card()` reads it), so `None` is what these respondents chose with the other "
        "answer in front of them. Whether some of them keep ancestral practice as well is a "
        "question the survey cannot answer and is not drawn.",
    "Traditional/ethnic religion":
        "-> indigenous.african. 0.20%, 18 respondents over five rounds, and a floor for "
        "`lr2022.py`'s reason: the card offers it as a peer of Christian and Muslim, so it "
        "cannot see anyone who is both, and see `None` above for where the rest probably went. "
        "Drawn flat: under the 1% floor, and spec §12's small-category rule sends the tail "
        "flat because the residual would draw it at 4.03x in Mbeya and Songwe, where the "
        "survey found none.",
    "Other":
        "-> other.tz. 0.93%, 87 respondents, 64 of them in rounds 4 and 7. It clears the rank "
        "test (+0.523) and is under the 1% eligibility floor, so it is drawn flat at the "
        "national share. The card attaches no specify text.",
}

MAP = {
    "Christian": "christianity",
    "Roman Catholic": "christianity.catholic",
    "Lutheran": "christianity.lutheran",
    "Anglican": "christianity.anglican",
    "Pentecostal": "christianity.pentecostal",
    "Seventh Day Adventist": "christianity.adventist.sda",
    "Other Christian": "christianity",
    "Muslim": "islam",
    "Traditional/ethnic religion": "indigenous.african",
    "Other": "other.tz",
    "None": "unaffiliated",
}

# spec §7a-i-1: every row is measured at the node it is drawn on; nothing is inferred downward.
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
