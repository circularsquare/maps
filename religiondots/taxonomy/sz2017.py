"""Eswatini CSO 2017 PHC religion -> religiondots taxonomy.

Twenty-one categories plus the universe total, at region. Thirteen come from Table 3.2.2 /
3.2.4, *Christians by Denomination*, and carry the `Christian: ` prefix `sources/sz.py` adds;
eight come from Table 3.2.1, the top-level religion table, and are the country's non-Christian
answers.

**THE PREFIX IS NOT DECORATION.** Both tables print a row called `Other` and both print one
called `Not Stated`, and they mean different things: 13,458 other Christian DENOMINATIONS
against 3,363 other RELIGIONS, and 13 Christians who named no denomination against 23,925
people who named no religion. Nothing in the source distinguishes them, so the normalised file
does.

**THIS IS THE FIRST COUNTRY ON THE MAP WHOSE PLURALITY RELIGION IS AN AFRICAN INSTITUTED
CHURCH.** `Zionists` are 367,290 people, 33.60% of Eswatini, and adding `Apostles` takes
`christianity.africaninstituted` to **420,690, or 38.48%** — larger than every mission
denomination in the country put together. Zimbabwe's `Apostolic Sect` is a larger share of a
larger country (40.3%) and reached the node first; what Eswatini adds is that the CSO counts
the Zionist and Apostolic churches **beside a full mission denomination list** rather than as
one undivided cell next to `Protestant`, so for the first time the African Instituted share
can be read against named Catholics, Anglicans, Lutherans, Methodists and Nazarenes in the
same table.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Total":
        "the region's own population total, not a category.",
    "Not Stated":
        "23,925 people, 2.19%, who gave no religion at all in Table 3.2.1. Not drawn, per "
        "§3.5 and tt2011.py's Trinidad precedent: a non-response is not an answer and "
        "filling it in would invent the one thing the census declined to record. It is the "
        "whole of Eswatini's `gap`. Note this is NOT the same cell as `Christian: Not "
        "Stated`, which is thirteen people who said Christian and named no denomination.",
}

REVIEW = {
    "Christian: Zionists":
        "-> christianity.africaninstituted, and **this is the category the country is worth "
        "drawing for**. 367,290 people, 33.60% of Eswatini and 37.64% of its Christians -- "
        "the single largest religious answer in the country by a factor of one and a half "
        "over the next one, and **the first time an African Instituted church is the "
        "PLURALITY religion of a country on this map**. "
        "The Swazi Zionist churches descend from John Alexander Dowie's Christian Catholic "
        "Apostolic Church in Zion, a Chicago body whose missionaries reached southern Africa "
        "in 1904, and separated from it almost at once: they are indigenous churches with "
        "prophecy, faith healing, immersion baptism in running water, white robes and drums, "
        "and they are the ordinary religion of the Swazi countryside rather than a movement "
        "at its edges. The CSO gives them ONE CELL, so the Swazi Christian Church, the "
        "Jericho and Nazarite bodies and the many independent congregations cannot be told "
        "apart here. "
        "**The oracle calls this cell `Zion Christian Church` and the volume calls it "
        "`Zionists`, and the volume is right.** The Zion Christian Church proper is Engenas "
        "Lekganyane's church at Moria in Limpopo; what Table 3.2.2 counts is the whole Swazi "
        "Zionist stream. The UNSD label is a tidy-up made when the CSO forwarded the table "
        "and it would be a real error to read it as one denomination. §2.4 -- the source's "
        "own word is kept in `source_category`. "
        "Its geography is the poor rural south: **42.35% of Shiselweni's Christians and "
        "39.90% of Lubombo's, against 33.17% in Manzini**, and Table 3.2.3 puts it at 42.03% "
        "rural against 23.07% urban, the only denomination in the country that is more "
        "common outside the towns than in them.",
    "Christian: Apostles":
        "-> christianity.africaninstituted, with `Zionists`. 53,400 people, 4.88%. **Not a "
        "free call, and it is the one most worth arguing with.** In southern African usage "
        "the *Apostolic* churches are the second great African Instituted family beside the "
        "Zionist one -- the Vapostori of Zimbabwe (zw2022.py's 6.1M `Apostolic Sect`), the "
        "African Apostolic and Jericho churches, the *emajerike* of the Swazi Middleveld -- "
        "and branches.py's own definition of this node names 'the Zionist and Apostolic "
        "churches of southern Africa' in one breath. "
        "**What could send it elsewhere is that Apostolic Faith Mission is a Pentecostal "
        "body with a real Swazi presence.** The reason it does not: the CSO prints "
        "`Pentecostal` as its own cell three rows above, at 130,794, so a respondent in the "
        "AFM had a box of their own to tick and the residual meaning of `Apostles` is the "
        "indigenous one. Its geography agrees -- 7.68% of Shiselweni's Christians and 5.59% "
        "of Manzini's against 4.20% in Hhohho, which tracks `Zionists` and is the inverse of "
        "`Pentecostal`. A source that separates the two would be a straight lookup.",
    "Christian: Evangelical":
        "-> christianity.evangelical. 248,233 people, 22.71%, the second largest answer in "
        "the country. **An ANSWER and not a church**, which is the call ke2019.py and "
        "zw2022.py both make for their own broad cells: it holds the Church of the Nazarene's "
        "wider constituency, the Evangelical Church of Eswatini, the Alliance and Free "
        "Evangelical congregations and the independent evangelical churches, and those sit in "
        "several different places on the tree. No branch is inferred because the census gives "
        "none. Highest in Shiselweni at 33.35% of Christians and lowest in Hhohho at 19.99%.",
    "Christian: Nazarene":
        "-> christianity.holiness.nazarene, and it is genuinely that denomination rather than "
        "a generic label. 44,112 people, 4.03%. The Church of the Nazarene has been the "
        "largest mission body in Swaziland since Harmon Schmelzenbach arrived in 1910; it "
        "runs the Raleigh Fitkin Memorial Hospital at Manzini and a nursing college, and it "
        "is one of the few places on earth where a Holiness denomination is a percentage "
        "point of a national population. Highest in Hhohho (5.84%) and Lubombo (5.81%) of "
        "Christians. **The node is a US-tree leaf being used by a census outside the US**, "
        "which is what it is for; nothing about it is American except where it was founded.",
    "Christian: Methodist":
        "-> christianity.methodist, the parent rather than .african. 40,452 people, 3.70%. "
        "The Methodist Church of Southern Africa, whose Swazi work runs back to 1844 and is "
        "the oldest mission presence in the country. Not the African Methodist Episcopal "
        "line, which is a separate node and which the census does not name.",
    "Christian: Roman Catholic":
        "-> christianity.catholic, the parent rather than .latin. 35,969 people, 3.29%. "
        "Latin rite throughout and the census does not say so, which is the call ke2019.py, "
        "mw2018.py, bj2013.py and zw2022.py all make. Urban and central: 5.11% of Manzini's "
        "Christians and 6.53% of urban Christians nationally, against 1.38% in Shiselweni.",
    "Christian: Jehovah Witness":
        "-> christianity.witnesses. 11,896 people, 1.09%. The CSO's spelling is singular and "
        "is kept (§2.4). The one denomination Table 3.2.4 shows as genuinely even across the "
        "country, 1.12% to 1.27% of Christians in all four regions, which the volume's own "
        "text remarks on.",
    "Christian: Seventh Day Adventist":
        "-> christianity.adventist, the parent rather than .sda, which is the convention "
        "every non-US mapping here follows for a cell printed with this name. 8,783 people, "
        "0.80%. Also close to even across the regions, 0.67% to 1.04% of Christians.",
    "Christian: Other":
        "-> christianity.other. 13,458 people, 1.23%. **Unusually small for a residual of "
        "this kind** -- Ghana's is 12.3% and Malawi's 26.6% -- because the CSO has already "
        "lifted out eleven named bodies plus the Zionists and the Apostles, so what is left "
        "is genuinely a tail. Highest in Hhohho at 2.18% of Christians.",
    "Christian: Not Stated":
        "-> christianity.other. THIRTEEN PEOPLE in the whole country, five of them in "
        "Hhohho. They said Christian and named no denomination, so they are Christians whose "
        "branch is unknown, and `christianity.other` is where an unplaceable Christian goes. "
        "**Deliberately NOT excluded**, unlike Table 3.2.1's `Not Stated`: that one is 23,925 "
        "people who gave no religion at all and is a non-response, this one is an answer with "
        "a missing second level. At thirteen people it draws nothing at 1:1,000 either way; "
        "the point is that the two cells are different and are treated differently.",
    "Traditionalist":
        "-> indigenous.african, the node Ghana added. 4,869 people, 0.45%, and **read it as a "
        "floor, harder here than almost anywhere.** Swazi ancestral practice -- the "
        "*emadloti*, consultation of an *inyanga* or *sangoma*, and the national rituals of "
        "*Incwala* and *Umhlanga* which the monarchy conducts every year and which most of "
        "the country takes part in -- is not what this box measures. The box is exclusive of "
        "the Christian ones, so it counts people who put nothing else first, and 0.45% is "
        "very far below the share of Swazis who do any of those things. The Zionist churches "
        "in particular grew out of exactly that overlap and are counted at seventy-five times "
        "the size. No child node: no source names a Swazi tradition individually (§2.4). "
        "**DERIVED, like the other seven non-Christian cells** -- see the module docstring; "
        "the census publishes this figure nationally and not by region.",
    "No religion":
        "-> unaffiliated. 80,861 people, 7.40%, the largest non-Christian answer by a factor "
        "of sixteen. One cell, so nothing goes to `secular`, which needs a separately counted "
        "atheist or humanist answer -- the call ke2019.py, mw2018.py, bj2013.py and zw2022.py "
        "all make. "
        "**Its regional geography is not known and is not claimed.** The CSO publishes this "
        "figure nationally only, so it is drawn at the national rate inside each region's own "
        "measured non-Christian total and every row of it is `derived`. It is the single "
        "biggest thing this country cannot show, and the one a reader is most likely to "
        "assume it can: Zimbabwe's `None` runs 4.5% to 13.5% across its provinces, a "
        "threefold spread, and Eswatini's variation is invisible here.",
    "Islam":
        "-> islam, with no branch, because the census gives none. 3,626 people, 0.33%. Two "
        "communities the census cannot separate: the small Swazi Muslim population and the "
        "Indian and Pakistani traders of Mbabane and Manzini. `derived`.",
    "Bahai Faith":
        "-> bahai. 430 people, 0.04%. The CSO's spelling drops the apostrophe and is kept "
        "(§2.4); the oracle writes `Baha'i`. Under one dot at 1:1,000 in three of the four "
        "regions, so it will draw sparsely and may ring (§4.3). `derived`.",
    "Hindu":
        "-> hinduism, with no branch. 244 people, 0.02%, the smallest cell in the country and "
        "the same Indian-descended community as the Muslim figure. `derived`.",
    "Judaism":
        "-> judaism, with no branch. 163 people, 0.01%. `derived`. Zimbabwe's warning does "
        "NOT transfer: 163 people is the size an actual expatriate and Israeli-connected "
        "community would be, and there is no Lemba-shaped anomaly to explain, so this is read "
        "at face value.",
    "Other":
        "-> other.sz. 3,363 people, 0.31%. A NEW single-country node, and a small one, "
        "because the CSO gives Islam, Hinduism, the Baha'i Faith and Judaism boxes of their "
        "own -- so this is a genuine tail rather than the crowded residual most censuses hand "
        "over, the same shape as `other.zw`. Per §3.11. `derived`.",
}

MAP = {
    # Table 3.2.2 / 3.2.4 — Christians by denomination, MEASURED at region.
    "Christian: Roman Catholic": "christianity.catholic",
    "Christian: Anglican": "christianity.anglican",
    "Christian: Lutheran": "christianity.lutheran",
    "Christian: Methodist": "christianity.methodist",
    "Christian: Jehovah Witness": "christianity.witnesses",
    "Christian: Evangelical": "christianity.evangelical",
    "Christian: Pentecostal": "christianity.pentecostal",
    "Christian: Zionists": "christianity.africaninstituted",
    "Christian: Apostles": "christianity.africaninstituted",
    "Christian: Nazarene": "christianity.holiness.nazarene",
    "Christian: Seventh Day Adventist": "christianity.adventist",
    "Christian: Other": "christianity.other",
    "Christian: Not Stated": "christianity.other",
    # Table 3.2.1 — the non-Christian answers, published NATIONALLY, drawn `derived`.
    "Islam": "islam",
    "Hindu": "hinduism",
    "Bahai Faith": "bahai",
    "Traditionalist": "indigenous.african",
    "Judaism": "judaism",
    "Other": "other.sz",
    "No religion": "unaffiliated",
}

# spec §7a-i-1. A `derived` row rolls up to the node its SOURCE COLUMN names at the drawn
# unit. Eswatini's eight derived rows have NO such column: what the CSO measured per region
# is the thirteen Christian denominations, and their residual is everybody else, which is not
# a religion and has no node. So there is deliberately no COLUMNS entry here and
# tools/check_rollup.py will report these rows as not rolling up. That is the honest answer
# rather than a missing one -- inventing a "non-Christian" node to roll them into would be
# asserting a category the census never used.
COLUMNS = {}


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
