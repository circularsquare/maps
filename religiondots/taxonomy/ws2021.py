"""
SBS 2021 Census of Samoa, Table 2 -> religiondots taxonomy.

**Branch-level mapping, no nested universe and no leaves.** Table 2's 26 columns are a flat
partition of the whole enumerated population at every one of its 395 place rows, and they sum
to the printed total exactly at all of them. The source's own string travels with the row in
`source_category` (§2.4).

    CONGREGATIONAL CHRISTIAN CHURCH OF SAMOA  55,411  26.96%  -> ...congregational.cccs
    ROMAN CATHOLIC                            36,906  17.95%  -> christianity.catholic.latin
    LATTER DAY SAINTS                         36,159  17.59%  -> christianity.latterday
    METHODIST                                 24,318  11.83%  -> christianity.methodist
    ASSEMBLY OF GOD                           20,733  10.09%  -> christianity.pentecostal.trinitarian
    SEVENTH DAYS ADVENTIST                     8,070   3.93%  -> christianity.adventist
    WORSHIP CENTRE                             6,131   2.98%  -> christianity.pentecostal.charismatic
    OTHER CHURCHES                             3,889   1.89%  -> christianity.other
    ASO FITU (SISDAC)                          1,962   0.95%  -> christianity.adventist.sisdac
    VOICE OF CHRIST                            1,741   0.85%  -> christianity.pentecostal.charismatic
    JEHOVAHS WITNESS                           1,719   0.84%  -> christianity.witnesses
    FIRST FULL GOSPEL PENTECOSTAL CHURCH       1,706   0.83%  -> christianity.pentecostal.trinitarian
    NAZARENE                                   1,288   0.63%  -> christianity.holiness
    PEACE CHAPEL                                 850   0.41%  -> christianity.pentecostal.charismatic
    BAHAI                                        790   0.38%  -> bahai
    AMAZING LOVE CHRISTIAN CHURCH                684   0.33%  -> christianity.pentecostal.charismatic
    PABTISM                                      658   0.32%  -> christianity.baptist
    CHRISTIAN FELLOWSHIP                         565   0.27%  -> christianity.nondenominational
    CCCJS (EFIS)                                 482   0.23%  -> christianity.reformed.congregational
    BIBLE STUDY                                  476   0.23%  -> christianity.nondenominational
    ELIM CHURCH                                  231   0.11%  -> christianity.pentecostal.trinitarian
    POROTESANO                                   228   0.11%  -> christianity.protestant
    SAMOA EVANGELISM                             223   0.11%  -> christianity.evangelical
    ANGLICAN CHURCH                              153   0.07%  -> christianity.anglican
    NO RELIGION                                  132   0.06%  -> unaffiliated
    MUSLIM                                        52   0.03%  -> islam

**NOTHING IS EXCLUDED AND NOTHING IS DERIVED. SAMOA IS 100% DRAWN.** There is no `Not stated`
column at all — the 2016 census had one and 2021 does not — and the only residual is a named
`OTHER CHURCHES` at 1.9%, which is small enough that the country's tail is genuinely enumerated
rather than swept. That is the cleanest a source gets on this map.

**THE SAMOAN VILLAGE IS RELIGIOUSLY MONOLITHIC, AND ONLY A VILLAGE TABLE SHOWS IT.** Malua is
**424 people and 424 Congregationalists**, not one person in any other column. Amaile is 99.6%
Roman Catholic, Mulivai Safata 99.5%, Tapueleele 99.4% Latter Day Saints, Gataivai 94.4%
Methodist. Under the *fa'amatai* system the village council decides matters for the village and
the church is one of them, so what the census records at village level is closer to a single
collective answer than to 600 individual ones. **None of this is visible on the map**, which is
drawn at 25 districts because that is as fine as Samoa's geometry goes (`sources/ws_geo.py`).

**A THREE-WAY GEOGRAPHY AT THE DISTRICT SCALE, WHICH IS VISIBLE.** The Methodists are Savai'i:
20.5% of that island against 8-10% of everywhere else, **62.7% of Satupaitea** and 94.4% of
Gataivai village. The Catholics are Apia and the far ends of the country: 25.3% of the Apia
Urban Area, 35.4% of Falealupo at the western tip of Savai'i and 35.4% of Aleipata Itupa i Lalo
at the eastern end of Upolu. The Latter Day Saints run the other way from the Catholics, 21.4%
on Savai'i and 13.6% in Apia, reaching **34.6% of Vaa o Fonoti**.

**LATTER DAY SAINTS AT 17.6% IS SECOND ONLY TO TONGA.** Of the 67 country-years in UNSD table 28
that count Latter Day Saints separately, Tonga 2021 is first at 19.7% and Samoa is second.
"""

EXCLUDED = {}

REVIEW = {
    "CONGREGATIONAL CHRISTIAN CHURCH OF SAMOA":
        "-> christianity.reformed.congregational.cccs, a node added for it. **55,411 people, "
        "27.0%, the largest church in Samoa.** The Ekalesia Fa'apotopotoga Kerisiano Samoa is "
        "the London Missionary Society's Samoan church, from John Williams' landing at "
        "Sapapali'i in 1830, and it is the mother of the three LMS bodies already on this "
        "tree: `.cicc` (Cook Islands), `.ekt` (Tuvalu) and `.niue`. **`.ekt`'s own note "
        "predicted this one** — *\"Samoa's own CCCS would be a third if Samoa is drawn\"* — "
        "so this completes a set rather than opening one.",
    "ASO FITU (SISDAC)":
        "-> christianity.adventist.sisdac, a node added for it. 1,962 people, 0.95%. The "
        "Samoa Independent Seventh Day Adventist Church, a local schism from the Seventh-day "
        "Adventists that kept the sabbath and the doctrine. **No other census anywhere counts "
        "it**, which is the argument for the node; the argument against is that at Samoa's "
        "25-district grain it never exceeds 3.0% of a unit and will not show a geography. "
        "Ekalesia Niue, 981 people, is the precedent for a node this size. `Aso Fitu` is "
        "Samoan for *the seventh day*.",
    "CONGREGATIONAL CHRISTIAN CHURCH OF JESUS IN SAMOA (EFIS)":
        "-> christianity.reformed.congregational, the parent and not a child. 482 people. It "
        "is a breakaway from the CCCS and could have a node beside it, but at 0.23% and with "
        "no district reaching 2% it would be a colour nobody could find. On the parent it "
        "still reads as Congregational, which is what it is. Revisit if Samoa is ever drawn "
        "at a finer tier.",
    "BIBLE STUDY":
        "-> christianity.nondenominational. 476 people, and the shakiest cell in the file. "
        "UNSD table 28's 2016 return calls the same body **`Aoga Tusi Paia`**, Samoan for "
        "*Bible school*, so the English column heading is a translation of a Samoan name "
        "rather than a denomination anybody would recognise. Nothing published says which "
        "body it is. `christianity.nondenominational` holds an answer rather than a church "
        "(see the node), which is the honest place for a name that describes a practice.",
    "POROTESANO":
        "-> christianity.protestant. 228 people. Samoan for *Protestant*, and UNSD's 2016 "
        "return renders the same cell in English as `Protestant`. So this is an ANSWER and "
        "not a body, which is exactly what `christianity.protestant` exists for; it is "
        "deliberately not a parent of the Protestant families.",
    "PABTISM":
        "-> christianity.baptist. 658 people. The workbook's own spelling, kept in "
        "`source_category` per §2.4; UNSD's 2016 return calls the same cell `Baptist`, which "
        "is what settles the identification.",
    "OTHER CHURCHES":
        "-> christianity.other and NOT a country residual. Samoa names Muslims, Baha'is and "
        "`No religion` in columns of their own, so this cell is what was left after "
        "twenty-five named answers, and the header says **churches**. That is the opposite "
        "of Vanuatu's `Other churches` (§9bg), where volume 2 of the same census defined the "
        "cell as *88 different religions* and it had to go to `other.vu`. Here the wording "
        "and the rest of the table agree, so it stays Christian. 1.89%.",
    "WORSHIP CENTRE":
        "-> christianity.pentecostal.charismatic. 6,131 people, 3.0%, and the largest of "
        "Samoa's independent charismatic churches. Grouped there with Voice of Christ, Peace "
        "Chapel and Amazing Love: all four are founder-led Samoan congregations in the "
        "worship-and-revival stream rather than branches of a classical Pentecostal body, "
        "which is the Philippines' precedent (`ph2020.py`). **They are an Apia and North "
        "West Upolu phenomenon** — Worship Centre is 4.1% and 4.5% of those two regions "
        "against 1.0% of the Rest of Upolu.",
}

COLUMNS = {
    # Every category is measured at the village, which is finer than the unit anything is
    # drawn on, so no row is derived and nothing rolls up. Recorded explicitly per
    # COMMANDS.txt's check_rollup note.
}

MAP = {
    "CONGREGATIONAL CHRISTIAN CHURCH OF SAMOA":
        "christianity.reformed.congregational.cccs",
    "CONGREGATIONAL CHRISTIAN CHURCH OF JESUS IN SAMOA (EFIS)":
        "christianity.reformed.congregational",
    "ROMAN CATHOLIC":                                 "christianity.catholic.latin",
    "LATTER DAY SAINTS":                              "christianity.latterday",
    "METHODIST":                                      "christianity.methodist",
    "ASSEMBLY OF GOD":                                "christianity.pentecostal.trinitarian",
    "FIRST FULL GOSPEL PENTECOSTAL CHURCH IN SAMOA":  "christianity.pentecostal.trinitarian",
    "ELIM CHURCH":                                    "christianity.pentecostal.trinitarian",
    "WORSHIP CENTRE":                                 "christianity.pentecostal.charismatic",
    "VOICE OF CHRIST":                                "christianity.pentecostal.charismatic",
    "PEACE CHAPEL":                                   "christianity.pentecostal.charismatic",
    "AMAZING LOVE CHRISTIAN CHURCH":                  "christianity.pentecostal.charismatic",
    "SEVENTH DAYS ADVENTIST":                         "christianity.adventist",
    "ASO FITU (SISDAC)":                              "christianity.adventist.sisdac",
    "JEHOVAHS WITNESS":                               "christianity.witnesses",
    "NAZARENE":                                       "christianity.holiness",
    "PABTISM":                                        "christianity.baptist",
    "CHRISTIAN FELLOWSHIP":                           "christianity.nondenominational",
    "BIBLE STUDY":                                    "christianity.nondenominational",
    "POROTESANO":                                     "christianity.protestant",
    "SAMOA EVANGELISM":                               "christianity.evangelical",
    "ANGLICAN CHURCH":                                "christianity.anglican",
    "OTHER CHURCHES":                                 "christianity.other",
    "BAHAI":                                          "bahai",
    "MUSLIM":                                         "islam",
    "NO RELIGION":                                    "unaffiliated",
}


def resolve(category):
    """Source category -> node. Samoa excludes nothing, so this never returns None."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
