"""
FBoS 2007 Census of Fiji, Table P01-3 (religion) -> religiondots taxonomy.

**Branch-level mapping with a nested parent, like bs2022.py.** No leaves are created; the
category's own name travels with the row in `source_category` (spec §2.4).

    Methodist                 290,555   34.70%   -> christianity.methodist
    Hindu                     232,103   27.72%   -> hinduism
    Catholic                   76,603    9.15%   -> christianity.catholic.latin
    Moslem                     52,594    6.28%   -> islam
    Assembly of God            47,873    5.72%   -> christianity.pentecostal.trinitarian
    Seventh Day Adventist      32,370    3.87%   -> christianity.adventist
    Other Christian            17,019    2.03%   -> christianity
    Penticostal                15,326    1.83%   -> christianity.pentecostal
    Christ Mission Fellowship  14,180    1.69%   -> christianity.pentecostal
    All Nation Christian       13,294    1.59%   -> christianity.nondenominational
    Jehovah's Witnesses         8,450    1.01%   -> christianity.witnesses
    Anglican                    6,328    0.76%   -> christianity.anglican
    Latter Day Saints           5,126    0.61%   -> christianity.latterday
    Apostolic                   5,089    0.61%   -> christianity.pentecostal
    No religion                 4,249    0.51%   -> unaffiliated
    Presbyterian                2,907    0.35%   -> christianity.reformed.presbyterian
    Gospel                      2,835    0.34%   -> christianity
    Sikh                        2,548    0.30%   -> sikhism
    Baptist                     1,772    0.21%   -> christianity.baptist
    United Pentecostal          1,361    0.16%   -> christianity.pentecostal.oneness
    Church of Christ            1,356    0.16%   -> christianity.restorationist
    Other religion              1,294    0.15%   -> other.fj
    Salvation Army              1,144    0.14%   -> christianity.holiness.salvation-army
    ---------------------------------------------------------------------------------
    Christian                 543,588            NESTED UNIVERSE, EXCLUDED
    Not stated                    895    0.11%   §3.5 residual, EXCLUDED
    Total                     837,271            universe, EXCLUDED

**`Christian` MUST NOT BE DRAWN, AND THAT IS THE ONE WAY TO BREAK THIS FILE.** The eighteen
denominations sum to it exactly on all fifteen provinces (`sources/fj.py` asserts it), so
mapping the parent as well would count 543,588 people twice — 65% of the country. It is in
EXCLUDED beside `Total` for that reason and not because it is uninteresting.

**FIJI IS THE MOST RELIGIOUSLY PLURAL COUNTRY IN THE PACIFIC AND THE ONLY ONE HERE THAT NEEDS
MORE THAN A CHRISTIAN PALETTE.** 64.9% Christian, 27.7% Hindu, 6.3% Muslim — the Indo-Fijian
population brought to the cane districts under indenture from 1879 — and the two halves are
geographically separate: Methodist is **83.7% of Lau and 81.7% of Kadavu**, Hindu is **44.3%
of Macuata and 39.7% of Ba**. No other source on this map has a Hindu or Muslim plurality
outside South Asia and the Caribbean.

**AND IT NAMES SIKHS.** 2,548 people with a printed line of their own. Very few censuses
anywhere separate them; Fiji does because the Punjabi minority inside the indenture migration
was counted separately from the colonial period on.

**THE TIER IS PROVINCES BECAUSE THE CATEGORIES ARE THE POINT.** SPC's PopGIS serves the same
census at 86 tikina — 5.7x finer — with the eighteen Christian bodies collapsed into one
`Christians` column. That would delete Methodist, the largest body in the country, and leave
Fiji looking like every other Pacific census here. `sources/fj.md` §3 has the argument.
"""

EXCLUDED = {
    "Total": "the province's own population total, not a religion category",
    "Christian":
        "**A NESTED UNIVERSE, NOT A CATEGORY.** The eighteen named denominations below sum "
        "to this figure exactly on all fifteen provinces, so drawing it too would double "
        "543,588 people — 65% of Fiji. Excluded for arithmetic, not for taste.",
    "Not stated":
        "895 people, **0.11%** — and it is not a printed row. Table P01-3 states a total of "
        "837,271 and prints six top-level rows summing to 836,376; this is the difference. "
        "SPC's PopGIS identifies it independently: its `other religion` equals the printed "
        "Sikh + Other religion + exactly these 895, which is where a residual goes and not "
        "where a religion goes. Spec §3.5: marked, not filled. Fiji is 99.89% drawn.",
}

REVIEW = {
    "All Nation Christian":
        "-> christianity.nondenominational, and this is the least certain call in the file. "
        "13,294 people, 1.59%. The census truncates every label to fourteen characters, so "
        "the printed string is `All Nation Chr`; the body it most plausibly names is **All "
        "Nations Christian Fellowship**, a Suva church that describes itself as "
        "non-denominational. Two uncertainties, both stated rather than hidden: the "
        "truncation means the identification is probable and not proven, and even granted "
        "it, `non-denominational` is the church's own self-description rather than a "
        "structural fact. The alternative was the bare `christianity` family node, which "
        "would discard a distinction FBoS chose to print. Worth a second opinion.",
    "Christ Mission Fellowship":
        "-> christianity.pentecostal. 14,180 people. **Christian Mission Fellowship "
        "International**, founded in Suva in 1990 by Suliasi Kurulo and one of the largest "
        "home-grown Pentecostal denominations in the Pacific — a Fijian church that now "
        "plants congregations in a hundred countries rather than an imported mission. It "
        "sits on the Pentecostal family and not on a child because FBoS publishes nothing "
        "about its doctrine of the Godhead, which is what `trinitarian` and `oneness` "
        "divide on.",
    "Apostolic":
        "-> christianity.pentecostal. 5,089 people, and the label is genuinely ambiguous: "
        "`Apostolic` names the **Apostolic Church**, the classical Pentecostal body of Welsh "
        "origin with a long Pacific presence, but it is also how the **New Apostolic "
        "Church** — which is not Pentecostal and not evangelical — is usually printed. The "
        "Pentecostal reading is taken because the Apostolic Church's Fijian presence is "
        "documented and the New Apostolic Church's is negligible, and because the "
        "surrounding list is otherwise all Pentecostal and evangelical bodies. On the family "
        "node either way, so a wrong reading here misplaces 0.6% of Fiji by one branch "
        "rather than across the tree.",
    "Assembly of God":
        "-> christianity.pentecostal.trinitarian, following ph2020.py, nz2023.py and "
        "au2021.py. 47,873 people, **5.72%, and the third-largest body in Fiji** — bigger "
        "than the Anglicans, Presbyterians, Baptists, Adventists and Salvationists put "
        "together. The Assemblies of God is unambiguously Trinitarian Pentecostal, which is "
        "what separates it from `United Pentecostal` below.",
    "Gospel":
        "-> christianity, the bare family node. 2,835 people and the label names no body: "
        "`Gospel` in Fijian usage covers a scatter of independent evangelical congregations. "
        "Reaching for `evangelical` or `pentecostal` would invent an affiliation FBoS did "
        "not record (§14.4).",
    "Other Christian":
        "-> christianity. 17,019 people, 2.03%, and genuinely a residual WITHIN Christianity "
        "rather than a body: FBoS prints eighteen named churches and this is what did not "
        "fit any of them. It stays on the family node, where it can neither claim a "
        "denomination nor fall out of Christianity.",
    "Moslem":
        "-> islam. FBoS's spelling, kept verbatim in `source_category` per §2.4. 52,594 "
        "people, 6.28%, and NOT split: Fiji's Muslims are overwhelmingly Sunni of South "
        "Asian origin with a small Ahmadiyya community, but the census publishes one line "
        "and inventing the split would be §14.4 rule 1.",
    "Hindu":
        "-> hinduism, undivided. 232,103 people and **27.7% of the country**, the largest "
        "Hindu share on this map outside India, Nepal and Mauritius. The tree has "
        "`hinduism.tamil`, `hinduism.telugu` and `hinduism.aryasamaj` nodes and Fiji has all "
        "three communities — the Arya Samaj especially, which ran schools across the cane "
        "belt — but FBoS prints one line, so it stays on the parent.",
    "Penticostal":
        "-> christianity.pentecostal. FBoS's own misspelling of Pentecostal, kept verbatim "
        "in `source_category` because §2.4 says the source's string travels with the row. "
        "15,326 people, distinct on this form from Assembly of God, United Pentecostal, "
        "Apostolic and Christ Mission Fellowship, so it is the unaffiliated-Pentecostal "
        "remainder and belongs on the family node.",
}

MAP = {
    # the eighteen named Christian bodies
    "Anglican": "christianity.anglican",
    "Apostolic": "christianity.pentecostal",
    "Assembly of God": "christianity.pentecostal.trinitarian",
    "All Nation Christian": "christianity.nondenominational",
    "Baptist": "christianity.baptist",
    "Catholic": "christianity.catholic.latin",
    "Christ Mission Fellowship": "christianity.pentecostal",
    "Church of Christ": "christianity.restorationist",
    "Gospel": "christianity",
    "Jehovah's Witnesses": "christianity.witnesses",
    "Latter Day Saints": "christianity.latterday",
    "Methodist": "christianity.methodist",
    "Penticostal": "christianity.pentecostal",
    "Presbyterian": "christianity.reformed.presbyterian",
    "Salvation Army": "christianity.holiness.salvation-army",
    "Seventh Day Adventist": "christianity.adventist",
    "United Pentecostal": "christianity.pentecostal.oneness",
    "Other Christian": "christianity",
    # and the five that are not Christian, plus no religion
    "Hindu": "hinduism",
    "Sikh": "sikhism",
    "Moslem": "islam",
    "Other religion": "other.fj",
    "No religion": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): every category is measured at the province it is drawn on,
# so no row is `derived` and nothing ever needs to roll up.


def resolve(category):
    """religiondots branch for an FBoS category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
