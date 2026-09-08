"""CSO 2022 census religion classification -> religiondots taxonomy.

Twenty-two named categories plus a non-answer, on 10 districts. Shares are of the household
population, 171,834, which is CSO's own **weighted** estimate — see `sources/lc.py` on the
23.3% undercount the census corrected before publishing.

    50.61%  Roman Catholic            -> christianity.catholic
    14.11%  None - No religion ...    -> unaffiliated                       REVIEW
    10.82%  Seventh Day Adventist     -> christianity.adventist
     9.03%  Pentecostal               -> christianity.pentecostal
     4.11%  Not reported              -> EXCLUDED
     2.24%  Other                     -> other.lc   (a NEW node)
     2.19%  Mennonite                 -> christianity.evangelical           REVIEW  <-- !!
     1.74%  Baptist                   -> christianity.baptist
     1.43%  Rastafarian               -> rastafari
     1.28%  Anglican                  -> christianity.anglican
     0.79%  Jehovah Witnesses         -> christianity.witnesses
     0.39%  Methodist                 -> christianity.methodist
     0.30%  Atheist - Do not ...      -> secular                            REVIEW
     0.18%  Nazarene                  -> christianity.holiness.nazarene
     0.17%  Islam                     -> islam
     0.16%  Universal Church          -> christianity.pentecostal.charismatic  REVIEW
     0.15%  Hindu                     -> hinduism                           REVIEW
     0.12%  Salvation Army            -> christianity.holiness.salvation-army
     0.07%  Brethren                  -> christianity.plymouth              REVIEW
     0.04%  Hinduism                  -> hinduism   (the SAME node as Hindu)  REVIEW
     0.03%  Buddhism                  -> buddhism
     0.03%  Mormon                    -> christianity.latterday
     0.02%  Bahai Faith               -> bahai

**THE `Mennonite` ROW IS THE FORM'S `Evangelical` OPTION, AND IT IS 2.2% OF THE COUNTRY.**
This is the largest single reading decision on this country and it is not a guess:
`sources/lc.py` machine-checks the evidence on every run and refuses to build without it.
See REVIEW below for the whole chain.

**SAINT LUCIA IS THE MOST CATHOLIC COUNTRY ON THIS MAP** at 50.61% — ahead of Mexico
(77.7% but a different vintage question), and far ahead of every other Caribbean source here:
Grenada 31.5%, Trinidad 21.6%, Belize 40.1%, Barbados 3.8%. **And it is falling faster than
anything else this map can see.** The census's own back-series, `Table 40` of the 2010
preliminary report, is one number a decade:

    1960  92.4%     1970  90.5%     1980  85.6%
    1991  79.0%     2001  67.5%     2010  61.1%     2022  50.6%

**Forty-two points in sixty-two years**, and still 9 points a decade at the end. What
replaced it is not irreligion alone: Seventh Day Adventists went 1.8% -> 10.8% and
Pentecostals 0.0% -> 9.0% over the same span, so about half the loss went to two churches
and half to `None`.

**THE CATHOLIC GEOGRAPHY IS THE ISLAND'S NORTH-SOUTH DIVIDE.** 71.7% in Choiseul, 70.1% in
Soufriere, 66.1% in Canaries — the south-west coast — against **44.1% in Anse La Raye, 44.8%
in Castries and 45.8% in Gros Islet**, the north-west where nearly two-thirds of Saint
Lucians now live. The Adventists run the other way, 18.6% of Anse La Raye and 16.3% of
Canaries against 5.3% in Soufriere.

**AND THE FORM SPLITS DISBELIEF FROM NON-AFFILIATION, WHICH ALMOST NOTHING ELSE HERE DOES.**
`None - No religion but believe in God` is 14.11% and `Atheist - Do not believe in God` is
0.30% — **a 47-fold gap** — so the two nodes below are carrying a distinction the census
actually measured rather than one inferred from a single box. Grenada's 2021 form
(`gd2021.py`) does the same thing and gets 5.95% against 0.05%, a 130-fold gap.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the district's own population total, not a category. Carried in lc.csv because "
        "sources/lc.py checks the categories against it.",
    "Not reported":
        "7,064 people, **4.11%**. Spec §3.5: marked, not filled. "
        "**Its geography is the one thing worth reading here**: 6.90% in Castries and "
        "4.49% in Gros Islet — the two urban districts, which together hold 53% of the "
        "country — against 0.41% in Canaries and 0.76% in Dennery. A refusal rate that "
        "tracks urbanness is the ordinary pattern and it means the drawn shares in "
        "Castries rest on 93% of its people rather than 99%. "
        "**It is NOT the census's undercount, and Saint Lucia's undercount is not missing "
        "from this map** — CSO estimated the enumeration was 23.3% short and weighted "
        "every district back up before publishing (`sources/lc.py`), which is the opposite "
        "of Barbados (`bb2010.py`), where the raw count is published and this project "
        "declines to scale it. So what is drawn here already includes CSO's estimate of "
        "the people it did not find.",
}

REVIEW = {
    "Mennonite":
        "-> christianity.evangelical, and **the source's label is not believed**. 3,760 "
        "people, **2.19% of Saint Lucia** — the largest reading decision on this country.\n"
        "**The census's own questionnaire says the option was `Evangelical`.** CSO "
        "publishes the 2022 instrument (*St Lucia Census 2022, Version 4*) on its own site, "
        "and question 1.5 offers 22 options. Table D.2's 23 rows are those 22 options **in "
        "the same order**, plus `Not reported`. Twenty-two of the twenty-three match one "
        "for one. The one that does not is **option 6: `Evangelical` on the form, "
        "`Mennonite` in the report**. `sources/lc.py` asserts exactly that — the same list, "
        "the same order, one disagreement, in that position — and refuses to build if the "
        "shape of the discrepancy changes.\n"
        "**The 2010 census agrees.** Its Table 40 has `Evangelical` at **2.2%**, the same "
        "share, in the same place in a similar list, and has no Mennonite row at all. The "
        "2022 report has no Evangelical row at all.\n"
        "**And there is no Mennonite community of 3,760 people in Saint Lucia.** The "
        "Caribbean's Mennonite populations are in Belize (`bz2022.py`), Jamaica and the "
        "Dominican Republic; Grenada's 2021 census counts 280 Mennonites, 0.26%, on a form "
        "that ALSO has a separate `EVANGELICAL` cell of 2.36% — which is what a real "
        "Mennonite count next to a real Evangelical count looks like in this region.\n"
        "**Why the node is `christianity.evangelical` and not `christianity.protestant`.** "
        "The branch node holds an ANSWER rather than a church, and the answer given was "
        "`Evangelical` — the same call `ke2019.py` makes for Kenya's `Evangelical Churches` "
        "and `gd2021.py` for Grenada's cell. Filing it as generic Protestant would discard "
        "the one thing the form actually establishes.\n"
        "**`lc.csv` still says `Mennonite`.** The normalised file records what the source "
        "printed (§12); this module is where the reading happens, which is the whole reason "
        "the two layers are separate. **Its geography fits Evangelical and not Mennonite** "
        "— 6.33% of Micoud, 5.07% of Laborie, 3.51% of Dennery, the rural south and east, "
        "against 0.93% in Castries and 0.05% in Anse La Raye — which is where Caribbean "
        "evangelical churches grow and is not a pattern any Anabaptist settlement makes.",
    "None - No religion but believe in God":
        "-> unaffiliated, NOT `secular`. 24,252 people, **14.11%**, the second largest "
        "answer in the country. branches.py draws the line at whether a POSITION is stated, "
        "and this option states the opposite of one: it is *no religion* **but believe in "
        "God**, which is affiliation lapsing rather than belief going. The form's own "
        "wording settles what `hu2022.py` and `jm2011.py` had to argue about from a bare "
        "`None`. "
        "**Its geography is the mirror of the Catholic one**: 16.24% in Castries, 16.08% in "
        "Dennery, 16.06% in Gros Islet, 16.04% in Anse La Raye — against **4.77% in "
        "Choiseul**, which is also the most Catholic district at 71.7%. Where the church "
        "held, nobody left it.",
    "Atheist - Do not believe in God":
        "-> secular, and this is the distinction branches.py exists for: `unaffiliated` is "
        "the absence of a religion, `secular` is a stated non-theistic position, and CSO "
        "put the two on the same form as options 20 and 21. 514 people, **0.30%** against "
        "the other cell's 14.11%. "
        "**A 47-fold gap is itself the finding.** ru2012.py and kz2021.py had to reason "
        "about which of the two a single Russian or Kazakh answer meant; Saint Lucia asked "
        "both questions and got a very lopsided answer, which is evidence about the "
        "Caribbean rather than about the instrument. Highest in Anse La Raye (0.58%) and "
        "Gros Islet (0.57%).",
    "Universal Church":
        "-> christianity.pentecostal.charismatic. 277 people, 0.16%. **This is the Igreja "
        "Universal do Reino de Deus** — the Brazilian neo-Pentecostal church, present "
        "across the Anglophone Caribbean as *the Universal Church of the Kingdom of God*, "
        "and `br2010.py` files the same body at the same node for its 1.87 million "
        "Brazilian members. Neo-Pentecostal, 1977, prosperity theology and no classical "
        "Pentecostal lineage, hence `charismatic` rather than `trinitarian`. "
        "**The alternative reading is rejected**: `Universal` could name the Unitarian "
        "Universalists, but nothing else on a Saint Lucian form suggests them, they have no "
        "Caribbean presence to speak of, and IURD does — the church has been in Castries "
        "since the 1990s. Its geography is urban and Dennery (0.39%), which fits a "
        "storefront church and not a liberal denomination.",
    "Hindu":
        "-> hinduism, together with the separate `Hinduism` row. See that entry.",
    "Hinduism":
        "-> hinduism, **the same node as `Hindu`**, which is option 7 on the same form. "
        "Question 1.5 offers both — `7:Hindu` and `19:Hinduism` — and they are 253 and 66 "
        "people. **This is a duplicated option, not two religions.** The form's list runs "
        "*Anglican, Baptist, Bahai Faith, Brethren, Buddhism, Evangelical, Hindu, ...* down "
        "to *Seventh-Day Adventist, Universal Church, Hinduism, Atheist, None, Other*, so "
        "`Hinduism` sits at the end of the list where late additions go, next to the "
        "non-religious options; the most likely history is that it was appended without "
        "anyone noticing option 7. "
        "**Both are drawn, at one node, and neither is dropped** — 319 people, 0.19%. The "
        "two cells even have different geographies (Hindu peaks in Gros Islet at 0.37% and "
        "Micoud at 0.29%; Hinduism in Gros Islet at 0.15%), which is what you would expect "
        "from an arbitrary split of one small population between two adjacent boxes. "
        "Merging them is not a judgement about what respondents meant — it is the only "
        "reading under which the same word does not appear twice in one legend.",
    "Brethren":
        "-> christianity.plymouth, NOT christianity.anabaptist.brethren. 122 people. The "
        "same Caribbean collision `jm2011.py`, `bs2022.py` and `bb2010.py` document: the "
        "Brethren assemblies of the Anglophone Caribbean are the **Plymouth / Christian "
        "Brethren**, out of 1820s Dublin and arriving through nineteenth-century British "
        "missions, and the Schwarzenau (German Baptist) Brethren have no Saint Lucian "
        "presence. §12: never map a category on its string alone.",
    "Other":
        "-> other.lc, a per-source residual (§3.11). 3,854 people, 2.24%. It is the form's "
        "own option 22 rather than a tabulator's leftover, and its flat geography (2.85% to "
        "1.15% across ten districts) makes it a mixture rather than a missing category by "
        "§9r's rule. Likely contents are in branches.py.",
    "Nazarene":
        "-> christianity.holiness.nazarene, and **not** anything under "
        "`christianity.anabaptist`. 310 people. The same §12 string collision `bb2010.py` "
        "documents at length for Barbados's 7,299: the U.S. Religion Census names four "
        "bodies containing the word, one Holiness (the Church of the Nazarene, out of 1908 "
        "Pilot Point) and three Anabaptist Apostolic Christian bodies with no Caribbean "
        "presence. **71% of Saint Lucia's Nazarenes are in Gros Islet** — 220 of 310, "
        "0.73% of that district against 0.18% nationally — which is a single congregation "
        "showing up in a national census.",
}

MAP = {
    "Anglican": "christianity.anglican",
    "Baptist": "christianity.baptist",
    "Bahai Faith": "bahai",
    "Brethren": "christianity.plymouth",
    "Buddhism": "buddhism",
    "Mennonite": "christianity.evangelical",     # the form says `Evangelical`. See REVIEW.
    "Hindu": "hinduism",
    "Jehovah Witnesses": "christianity.witnesses",
    "Methodist": "christianity.methodist",
    "Mormon": "christianity.latterday",
    "Islam": "islam",
    "Pentecostal": "christianity.pentecostal",
    "Nazarene": "christianity.holiness.nazarene",
    "Rastafarian": "rastafari",
    "Roman Catholic": "christianity.catholic",
    "Salvation Army": "christianity.holiness.salvation-army",
    "Seventh Day Adventist": "christianity.adventist",
    "Universal Church": "christianity.pentecostal.charismatic",
    "Hinduism": "hinduism",                      # the same node as `Hindu`. See REVIEW.
    "Atheist - Do not believe in God": "secular",
    "None - No religion but believe in God": "unaffiliated",
    "Other": "other.lc",
}


def _key(cat):
    """Normalise apostrophes, dashes and whitespace.

    CSO's questionnaire writes `Jehovah's Witnesses` and `Islam (Muslim)` where the report
    writes `Jehovah Witnesses` and `Islam`, and the two long non-religious options are
    printed across a line break. Folding here means a future switch to another release does
    not silently unmap a category (§12).
    """
    s = str(cat).replace("’", "'").replace("–", "-").replace("—", "-")
    return " ".join(s.split())


EXCLUDED = {_key(k): v for k, v in EXCLUDED.items()}
MAP = {_key(k): v for k, v in MAP.items()}
REVIEW = {_key(k): v for k, v in REVIEW.items()}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    Saint Lucia's no-religion cell is spelled `None - No religion but believe in God`, so
    unlike Belize, Trinidad, the Bahamas and Cayman it is NOT the literal string `None` and
    survives a bare `pandas.read_csv`. `_lc_counts` in countries.py still passes
    `keep_default_na=False`, for consistency and because `Not reported` would otherwise be
    the next thing to go wrong.
    """
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
