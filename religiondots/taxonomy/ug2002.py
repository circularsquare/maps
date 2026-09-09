"""Uganda 2002 census Table B7 religion -> religiondots taxonomy.

Seven categories plus the row total, on the 56 districts of 2002. The list is short for
an African census and it is short in a particular direction: five named bodies, one
residual, one no-religion cell, and nothing at all between Anglicanism and `Other`.

    41.92%  Catholic     -> christianity.catholic
    35.95%  Anglican     -> christianity.anglican
    12.10%  Moslem       -> islam
     4.62%  Pentecostal  -> christianity.pentecostal
     3.04%  Other        -> other.ug
     1.51%  SDA          -> christianity.adventist.sda
     0.87%  None         -> unaffiliated

**THE TABLE IS 2002 AND THE COUNTRY HAS MOVED, WHICH IS THE FIRST THING TO KNOW BEFORE
READING ANY OF THESE.** UBOS has published religion nationally three times since, and
between 2002 and 2024 the Pentecostal cell went 4.6% -> 11.1% -> 14.3% while Catholicism
fell 41.6% -> 39.3% -> 36.2% and Anglicanism 36.7% -> 32.0% -> 29.0%. None of that
movement is subnational anywhere: there is no religion-by-geography table for 2014 or
2024. So the shares mapped here are a 2002 picture drawn on 2002 districts, and
`countries.py` says so in `how` and in the note.

**`None` IS THE LITERAL STRING AND PANDAS WILL DELETE IT** — the same trap as `ph`, `gy`
and `zw`. Read the normalized file with `keep_default_na=False, na_values=[""]` or 212,388
irreligious Ugandans resolve to nothing and disappear without an error.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the district's own population total, not a category. Table B7's own footnote "
        "says the table `excludes population enumerated in hotels`, and Table B1 of the "
        "same annex series gives each district's full 2002 count, so the excluded people "
        "are exactly B1 minus B7: 8,952 nationally, 0.037%, spread over 55 of the 56 "
        "districts. That is the country's `gap` and it is the smallest on this map.",
}

REVIEW = {
    "Catholic":
        "-> christianity.catholic, the parent rather than `.latin`. 10,242,594 people, "
        "41.92%, and the largest answer in 29 of the 56 districts. Latin rite throughout "
        "and the census does not say so, which is the call ke2019.py, mw2018.py, "
        "bj2013.py, zw2022.py and rw2022.py all make. "
        "**Its geography is the north and it is the White Fathers' mission field**: "
        "Adjumani 82.5%, Gulu 78.2%, Nebbi 74.1%, Pader 70.9%, against 13.6% in Iganga "
        "and 14.1% in Yumbe. The whole Northern region is 58.6% Catholic and the Eastern "
        "region 29.6%, which is the sharpest regional contrast in the table apart from "
        "Islam's.",
    "Anglican":
        "-> christianity.anglican. 8,782,821 people, 35.95%, and the largest answer in 25 "
        "districts. The body is the **Church of Uganda**, a province of the Anglican "
        "Communion, and in Ugandan usage `Protestant` means it: the 2002 Population "
        "Composition report prints the identical national figure under the label "
        "`Anglican /Protestant`, which is the same cell described twice. "
        "**Nothing is inferred about other Protestants from that second label.** The form "
        "already gives the Pentecostals and the Adventists boxes of their own and puts "
        "Baptists, Presbyterians, Methodists and the Salvation Army into `Other Christian` "
        "inside `Other`, so this cell is one church rather than a family, and "
        "`christianity.protestant` would claim the opposite. "
        "**Its geography is the south-west and the Buganda cattle corridor**: Nakasongola "
        "60.8%, Ntungamo 60.6%, Kanungu 56.7%, Kabale 53.5%, against 4.9% in Kotido and "
        "8.0% in Adjumani. It is close to the mirror image of the Catholic map, which is "
        "the CMS and White Fathers spheres of the 1890s still legible a century on.",
    "Moslem":
        "-> islam, the root. 2,956,121 people, 12.10%. Overwhelmingly Sunni, and the "
        "census does not say so, so nothing below the root is claimed. "
        "**This is the sharpest single thing in the table.** Yumbe district is **76.2%** "
        "Muslim against 0.4% in Kotido and 0.4% in Pader, a range of nearly two hundred "
        "to one across 56 units. Yumbe is Aringa county in West Nile, the Nubi and Aringa "
        "heartland, and the next highest are the Busoga shore of Lake Victoria and the "
        "old Buganda trading towns: Mayuge 36.2%, Iganga 33.8%, Jinja 25.9%, Kayunga "
        "25.8%. Those two clusters have nothing to do with each other historically, which "
        "is why Islam is the one column here that a four-region table destroys.",
    "Pentecostal":
        "-> christianity.pentecostal, the parent. 1,129,647 people, 4.62% in 2002 and "
        "**14.32% by the 2024 census**, which is the largest movement in Ugandan religion "
        "in living memory and the reason this table's age matters most for this row. The "
        "2014 and 2024 censuses label the same cell `Pentecostal/Born Again/Evangelical`. "
        "**No branch node**: `christianity.pentecostal.trinitarian`'s children are all "
        "United States denominations and the census gives a family name and nothing else, "
        "the same call rw2022.py and zw2022.py make. "
        "**Its geography in 2002 is Sebei and Teso, not Kampala**: Kapchorwa 18.0%, "
        "Kaberamaido 11.8%, Soroti 11.4%, Busia 11.2%, against 0.3% across Karamoja and "
        "0.6% in Arua. Kampala is 9.0%, above the national rate but not the top of it.",
    "SDA":
        "-> christianity.adventist.sda, the child rather than the parent, **because the "
        "source names the body**. `SDA` is the Seventh Day Adventist Church and nothing "
        "else, and the 2014 and 2024 censuses spell the same cell out in full; md2024.py "
        "and ro2021.py use the parent for the opposite reason, that their category name "
        "says only `Adventist`. 367,972 people, 1.51%. "
        "**Its geography is the Rwenzori**: Bundibugyo 13.5% and Kasese 8.4%, with "
        "Kabarole 6.7% behind them, against 0.0% in Moroto. Bundibugyo is nine times the "
        "national rate and the three are contiguous, which is what a single mission field "
        "looks like rather than a scatter.",
    "Other":
        "-> other.ug. 741,589 people, 3.04%. See the node's own note: Table B7's footnote "
        "says it holds Orthodox, Bahai, Other Christian, Non-Christian **and Traditional** "
        "together, and the traditional part is most of what makes it move. Per spec §3.11 "
        "it is drawn whole. "
        "**Splitting it was considered and refused.** The 2002 Population Composition "
        "report's Table 3.6 gives four of those pieces nationally — Orthodox 35.4k, Other "
        "Christian 282.3k, Bahai 18.5k, and a further 380.4k of `Other Non-Christians, "
        "Traditional and None` — and `sources/ug.py` checks that they reconcile with this "
        "cell exactly. But applying that national split to each district would put 24% of "
        "Kotido's 167,065 into `Other Christian`, and Kotido is Karamoja, where the cell "
        "is traditional religion. §14.4's first rule forbids estimating a magnitude the "
        "source does not publish, and this is the case it was written for.",
    "None":
        "-> unaffiliated. 212,388 people, 0.87%. One cell, so nothing goes to `secular`, "
        "which needs a separately-counted atheist or humanist answer — the call ke2019.py, "
        "mw2018.py, bj2013.py, zw2022.py and rw2022.py all make. "
        "**It is not a secular-city figure and it is not really a no-religion figure "
        "either.** A third of the entire national cell is in Kotido alone, and the three "
        "Karamoja districts hold 52% of it between them: Nakapiripirit 12.0%, Kotido "
        "11.9%, Moroto 11.7%, against 0.0% in Yumbe and 0.1% in Adjumani, Moyo and Arua. "
        "Kampala is 0.27%. Read together with `Other`, which peaks in the same three "
        "districts, this is a census question that offered five named churches and Islam "
        "to a population that practises neither, and recorded the answer in two different "
        "boxes.",
}

MAP = {
    "Catholic": "christianity.catholic",
    "Anglican": "christianity.anglican",
    "Moslem": "islam",
    "Pentecostal": "christianity.pentecostal",
    "SDA": "christianity.adventist.sda",
    "Other": "other.ug",
    "None": "unaffiliated",
}

# spec §7a-i-1: the level this country MEASURED each node at, so a dot inferred below one
# rolls up to the source's own column instead of vanishing. Table B7 measures all seven
# directly, at district, so each node's target is itself. Uganda has no `derived` rows.
COLUMNS = {v: v for v in MAP.values()}


def _key(cat):
    return " ".join(str(cat).split())


_FOLDED = {_key(k): v for k, v in MAP.items()}
_FOLDED_EXCLUDED = {_key(k) for k in EXCLUDED}


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree.

    `cat` is never Python's None here: the only reason it could be is pandas having
    parsed the string `None` as NaN, and countries.py refuses to run in that case rather
    than silently mapping a missing value onto the no-religion node.
    """
    c = str(cat)
    if c in EXCLUDED or _key(c) in _FOLDED_EXCLUDED:
        return None
    if c in MAP:
        return MAP[c]
    return _FOLDED.get(_key(c))
