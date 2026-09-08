"""
TSD 2021 Census of Tonga, General Table G 20 -> religiondots taxonomy.

**Branch-level mapping, no nested universe and no leaves.** G 20's twenty-two columns are a
flat partition of the enumerated population at every one of 156 villages, and they sum to the
printed national row exactly. The source's own string travels with the row in
`source_category` (§2.4).

    Free Wesleyan Church            33,953  34.16%  -> christianity.methodist.tongan.freewesleyan
    Latter Day Saints               19,534  19.65%  -> christianity.latterday
    Roman Catholic                  13,649  13.73%  -> christianity.catholic.latin
    Free Church of Tonga            11,244  11.31%  -> christianity.methodist.tongan.free
    Church of Tonga                  6,782   6.82%  -> christianity.methodist.tongan.tonga
    Seventh Day Adventist            2,461   2.48%  -> christianity.adventist
    Assembly of God                  2,455   2.47%  -> christianity.pentecostal.trinitarian
    Other Pentecostal                1,904   1.92%  -> christianity.pentecostal
    Tokaikolo/Maamafo'ou             1,455   1.46%  -> christianity.pentecostal.charismatic
    Constitutional Church of Tonga   1,152   1.16%  -> christianity.methodist.tongan.constitutional
    Baha'i Faith                       730   0.73%  -> bahai
    Other minor religious groups       714   0.72%  -> other.to
    Mo'ui Fo'ou 'Ia Kalaisi            688   0.69%  -> christianity.pentecostal.charismatic
    Anglican Church                    590   0.59%  -> christianity.anglican
    No Religious affiliation           574   0.58%  -> unaffiliated
    Gospel Church                      476   0.48%  -> christianity.pentecostal.trinitarian
    Jehovah's Witness                  400   0.40%  -> christianity.witnesses
    The Salvation Army                 332   0.33%  -> christianity.holiness.salvation-army
    Hinduism                            78   0.08%  -> hinduism
    Islam                               60   0.06%  -> islam
    Buddhist                            58   0.06%  -> buddhism
    -----------------------------------------------------------------------------------------
    Refuse to answer                   119   0.12%  §3.5 residual, EXCLUDED

**HALF THE COUNTRY IS METHODIST AND THE CENSUS COUNTS IT AS FOUR CHURCHES.** 53,131 people,
**53.4%**, split between the Free Wesleyan Church, the Free Church of Tonga, the Church of
Tonga and the Constitutional Church of Tonga — all four descended from the one Wesleyan
mission of 1826, all four a printed cell, and all four still here. Add the two revival
breakaways below and 55.6% of Tonga descends from that mission. Nothing else on this map
divides a single Protestant tradition this far. `christianity.methodist.tongan` was added for
them; the argument for grouping rather than collapsing is on that node.

**AND THE STATE CHURCH IS THE ONE BODY WITH NO GEOGRAPHY.** The Free Wesleyan Church is
34.1% of Tongatapu, 34.6% of Vava'u, 33.1% of Ha'apai, 36.3% of 'Eua and 30.7% of the Niuas.
Six points across the whole kingdom, while the Church of Tonga swings from 3.8% to 20.1% and
the Catholics from 5.4% to 36.3%.

**TONGA IS THE MOST LATTER-DAY-SAINT COUNTRY IN THE UN'S TABLE.** 19,534 people, **19.65%**,
which is the largest share of any of the 67 country-years in UNSD Demographic Yearbook table
28 that name a Latter Day Saints category; Samoa 2016 is second at 16.9%. It reaches 33.7% of
Hahake district in Vava'u and **59.6% of Matahau** on Tongatapu.

**THE CATHOLICS ARE THE NIUAS AND ONE OLD VILLAGE.** 13.7% nationally, but **36.3% of Ongo
Niua and 42.8% of Niuatoputapu**, the two islands 300 km north of everything else; and
**71.0% of Lapaha**, which was the seat of the Tu'i Tonga.
"""

EXCLUDED = {
    "Refuse to answer":
        "119 people, **0.12%**, printed as its own column. §3.5: marked, not filled. Tonga "
        "is 99.88% drawn, which is among the most complete countries on this map.",
}

REVIEW = {
    "Tokaikolo/Maamafo'ou":
        "-> christianity.pentecostal.charismatic, and this is the call worth a second "
        "opinion. 1,455 people, 1.5%. The **Siasi Tokaikolo 'Ia Kalaisi** began as a "
        "fellowship founded by the Reverend **Senituli Koloi in 1978**, who was a Free "
        "Wesleyan minister, and it took its congregations out of that church; it became the "
        "Tokaikolo Christian Church in 1994. So its DESCENT is Methodist and it could be "
        "argued onto `christianity.methodist.tongan` beside the other four. It is filed on "
        "the charismatic node instead because §2.1's containment is a fact about people "
        "now: Tokaikolo is a revival body in practice, and the four churches on "
        "`christianity.methodist.tongan` are the monarchy-era bodies that kept Methodist "
        "polity and divided over who governed the church. **The census's own alternative "
        "name for it, `Maama Fo'ou`, is not a Methodist name.** If this is wrong the fix is "
        "one line here and the geography does not move.",
    "Mo'ui Fo'ou 'Ia Kalaisi":
        "-> christianity.pentecostal.charismatic, following Tokaikolo, because it is a "
        "breakaway FROM Tokaikolo and not from the Free Wesleyan Church directly. 688 "
        "people, 0.7%. Tongan court records name the Mo'ui Fo'ou fellowship as the body a "
        "Tokaikolo congregation and its minister left to join. Same caveat as above.",
    "Other Pentecostal":
        "-> christianity.pentecostal, the branch itself and not a child. 1,904 people. The "
        "workbook's own legend expands the code as *\"Other Pentecostal (All Pentocostal "
        "Churches excluding AOG)\"*, so it is defined by exclusion from the Assembly of God "
        "and names nothing. Trinitarian and Oneness bodies cannot be separated inside it, "
        "and the branch is where a cell that names no body belongs.",
    "Gospel Church":
        "-> christianity.pentecostal.trinitarian. 476 people. The workbook's legend says "
        "`GOS‐Gospel Church`; **UNSD's table 28, from Tonga's own return, calls the same "
        "476 people `Full Gospel Church`**, which is the Pentecostal name and is how "
        "Australia and New Zealand's `Full Gospel` cells are filed here. The two spellings "
        "agreeing to the person is what makes the identification safe.",
    "Free Wesleyan Church":
        "-> christianity.methodist.tongan.freewesleyan. **33,953 people, 34.2%, and the "
        "church of the monarchy.** Filed under a Tongan grouping rather than on "
        "`christianity.methodist` because Tonga prints four Methodist churches and putting "
        "them on the branch would draw 53.4% of the country in one colour. See the node.",
}

COLUMNS = {
    # Every category above is measured at the village, so the roll-up target for all of them
    # is the village itself and nothing is inferred. Recorded explicitly per COMMANDS.txt's
    # check_rollup note: Tonga has no derived rows at all.
}

MAP = {
    "Free Wesleyan Church":           "christianity.methodist.tongan.freewesleyan",
    "Free Church of Tonga":           "christianity.methodist.tongan.free",
    "Church of Tonga":                "christianity.methodist.tongan.tonga",
    "Constitutional Church of Tonga": "christianity.methodist.tongan.constitutional",
    "Latter Day Saints":              "christianity.latterday",
    "Roman Catholic":                 "christianity.catholic.latin",
    "Seventh Day Adventist":          "christianity.adventist",
    "Assembly of God":                "christianity.pentecostal.trinitarian",
    "Gospel Church":                  "christianity.pentecostal.trinitarian",
    "Other Pentecostal":              "christianity.pentecostal",
    "Tokaikolo/Maamafo'ou":           "christianity.pentecostal.charismatic",
    "Mo'ui Fo'ou 'Ia Kalaisi":        "christianity.pentecostal.charismatic",
    "Anglican Church":                "christianity.anglican",
    "Jehovah's Witness":              "christianity.witnesses",
    "The Salvation Army":             "christianity.holiness.salvation-army",
    "Baha'i Faith":                   "bahai",
    "Hinduism":                       "hinduism",
    "Islam":                          "islam",
    "Buddhist":                       "buddhism",
    "No Religious affiliation":       "unaffiliated",
    "Other minor religious groups":   "other.to",
}


def resolve(category):
    """Source category -> node, or None for the excluded residual."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
