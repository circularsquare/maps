"""
Kuwait, 2021 census base -> religiondots taxonomy.

Named for the population base (the 2021 census's Table 52, Kuwaitis and non-Kuwaitis by sex in
157 areas). The religion is PACI's register count of June 2014, Muslim / Christian / Other-Not
Stated by nationality group and sex, carried to the areas by each area's mix of nationality
groups (`sources/kw.py`, `sources/kw.md`). Every row is `derived` (spec §7): counted nationally,
placed by a proxy.

Categories, as `data/normalized/kw.csv` writes them:

    Kuwaiti citizens, Shia                -> islam.shia   (PACI: 99.978% Muslim), split at
    Kuwaiti citizens, Sunni               -> islam.sunni  one national share, 17.69% Shia
                                                          (Arab Barometer III, the field team's
                                                          opinion; added 2026-10-03)
    Muslim                                -> islam        non-Kuwaitis
    Christian                             -> christianity non-Kuwaitis
    Other or not stated                   -> other.kw     PACI's `Other-Not Stated`, all groups
                                                          but the Asian, and the Asian part not
                                                          resolved by the split below
    Other or not stated, <node>           -> <node>       the Asian `Other-Not Stated`, split by
                                                          the six named Asian nationalities'
                                                          Pew 2020 shares; no row rolls up
                                                          (COLUMNS, below)
"""

EXCLUDED = {}

REVIEW = {
    "Kuwaiti citizens, Shia":
        "every area's Kuwaitis at ONE NATIONAL SHARE, 17.69% Shia, on Anita's ruling of "
        "2026-10-03 (one share for the whole country; a by-governorate split not approved). PACI "
        "counted 99.978% of Kuwaitis Muslim in June 2014 (1,257,977 of 1,258,254); the 255 "
        "Christians and 22 other or not stated are folded in. The share is Arab Barometer wave "
        "III (2014, citizens, probability sample, no quota), q2005kw, the field team's opinion of "
        "the respondent's sect: weighted, Sunni 66.6%, Shia 14.3%, cannot determine 19.1% of "
        "Muslims; drawn as Shia of those placed. CHOSEN OVER the State Department's 'about 30%' "
        "of citizens (attributed to NGOs and the media, no method) and Pew 2009's 20-25% of all "
        "Muslims (ethnographic ascription), because it is the only figure measured on a sample "
        "of Kuwaitis. It may read low if the team placed Shia less readily than Sunnis; the "
        "undetermined resemble the placed mix on the items that separate the two "
        "(sources/kw.md §8), and if every one were Shia the share would be 33.4%. Reversing it "
        "is one constant, sources/kw.py SECT_SHARE.",
    "Kuwaiti citizens, Sunni":
        "the rest of the Kuwaitis at the same national share, 82.31% (see the Shia row).",
    "Christian":
        "-> christianity, the bare family node, NOT split into churches as Saudi Arabia and Oman "
        "are. There the whole Christian count is Pew's per-nationality shares; here it is PACI's "
        "own count, and PACI's 593,751 Asian Christians are about ten times what Pew's national "
        "shares give the named Asian nationalities (Indians in Kuwait are far more Christian than "
        "India), so a nationality weighting of the churches would rest on nothing.",
    "Other or not stated":
        "-> other.kw, PACI's `Other-Not Stated`. For non-Asians it stays here whole (Arab 4,307, "
        "African 6,369, European, American and Oceanian 2,205 in 2014). Not `unaffiliated`: the "
        "register offers no such answer and `not stated` is part of the cell.",
    "Other or not stated, hinduism":
        "the Asian `Other-Not Stated` split by the six Asian nationalities PACI's mid-2018 table "
        "names by sex, each weighted by its own Pew 2020 count of people who are neither Muslim "
        "nor Christian. The magnitude is PACI's; the split is a model and India's weight in it is "
        "too large (Pew's India row, applied to 811,409 Indian men, would by itself come to nearly "
        "four times PACI's whole Asian `Other` for men), so Hindus are probably over-drawn against "
        "Buddhists: the layer draws about 20,000 Buddhists where informal community estimates "
        "(State Department 2023) say 100,000. Does not roll up (COLUMNS, below).",
}

MAP = {
    "Kuwaiti citizens, Shia": "islam.shia",
    "Kuwaiti citizens, Sunni": "islam.sunni",
    "Muslim": "islam",
    "Christian": "christianity",
    "Other or not stated": "other.kw",
    "Other or not stated, hinduism": "hinduism",
    "Other or not stated, buddhism": "buddhism",
    "Other or not stated, sikhism": "sikhism",
    "Other or not stated, jainism": "jainism",
    "Other or not stated, indigenous": "indigenous",
    "Other or not stated, unaffiliated": "unaffiliated",
}

# The PACI column each derived row came out of, KEPT AS A RECORD AND NOT ATTACHED AS `roll`:
# spec §7a-i-1's roll target must be measured at the unit the dot is drawn on, and PACI counted
# these columns for the whole country only (countries/kw.py sets NOWHERE). Reviewer, 2026-10-03.
COLUMNS = {
    "Kuwaiti citizens, Shia": "islam",
    "Kuwaiti citizens, Sunni": "islam",
    "Muslim": "islam",
    "Christian": "christianity",
    "Other or not stated": "other.kw",
    "Other or not stated, hinduism": "other.kw",
    "Other or not stated, buddhism": "other.kw",
    "Other or not stated, sikhism": "other.kw",
    "Other or not stated, jainism": "other.kw",
    "Other or not stated, indigenous": "other.kw",
    "Other or not stated, unaffiliated": "other.kw",
}


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
