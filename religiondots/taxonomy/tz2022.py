"""Afrobarometer Tanzania religion -> religiondots taxonomy.

Five categories, at 30 units (the 31 regions of the 2022 census with Songwe inside Mbeya's unit).
`Christian` and `Muslim` are the card's umbrella answers with their denominational children
folded back in; the other three are its own boxes. `sources/tz.py`'s docstring has the argument
for the folding and `sources/tz.md` the acquisition record. The same construction as
`ng2022.py`, on the same instrument, and the same ruling (ask/answered/010-ng).

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {}

REVIEW = {
    "Christian":
        "-> christianity, the bare branch. **No Tanzanian census since 1967 has asked about "
        "religion**, so nothing has ever split it by church. What is inside it is known in "
        "outline: the Catholic Church, the Evangelical Lutheran Church in Tanzania (strongest "
        "in Kilimanjaro, Arusha and Kagera), the Anglican Church of Tanzania, the Moravians of "
        "the south-west, the Seventh-day Adventists and the Pentecostal churches. The pooled "
        "survey names all of them and cannot level them: the share answering `Christian only` "
        "runs from 6.4% to 21.3% across the drawn rounds, and round 7's card gives Methodist "
        "7.1% and Anglican 0.5% where every other round gives about 0.1% and 4.5%, which is a "
        "different card rather than a different country. Left at the parent.",
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
