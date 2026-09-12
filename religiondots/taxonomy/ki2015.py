"""
Kiribati NSO, 2015 Population Census, Report Volume 1 Table 6 -> religiondots taxonomy.

**Branch-level mapping, no nested universe and no leaves.** Table 6's fourteen columns are a
flat partition of the whole enumerated population at every one of 24 islands, and they sum to
the printed national row exactly. The source's own string travels with the row in
`source_category` (§2.4).

    Roman Catholic                63,116  57.31%  -> christianity.catholic.latin
    KPC                           34,464  31.29%  -> christianity.reformed.congregational.kpc
    Latter Day Saints              5,857   5.32%  -> christianity.latterday
    Bahai                          2,314   2.10%  -> bahai
    Seventh Day Adventist          2,064   1.87%  -> christianity.adventist
    Other                            832   0.76%  -> other.ki
    Assembly of God                  364   0.33%  -> christianity.pentecostal.trinitarian
    Jehova's Witness (Te Koaua)      352   0.32%  -> christianity.witnesses
    Church Of God                    279   0.25%  -> christianity.holiness
    All Nation                       141   0.13%  -> christianity.baptist
    Islam                            139   0.13%  -> islam
    Te Ran                            86   0.08%  -> other.ki
    Four Square                       77   0.07%  -> christianity.pentecostal.trinitarian
    No religion                       51   0.05%  -> unaffiliated

**NOTHING IS EXCLUDED. KIRIBATI IS 100% DRAWN**: the 2015 table has no refusal cell and no
not-stated cell. (The 2020 census introduced a `Not stated` of 62 people; 2015 has none.)

**THE MISSION PARTITION OF THE GILBERTS IS STILL ALMOST PERFECT AFTER 150 YEARS.** The chain
runs Catholic in the north and Protestant in the south, and the two ends invert:

    Butaritari   82.5% Catholic   13.2% KPC
    Makin        79.8%            15.9%
    Marakei      76.8%            15.6%
    Abaiang      75.3%            16.8%
    ...
    Beru         29.4%            65.3%
    Onotoa       27.1%            67.1%
    Tamana        2.0%            95.8%
    Arorae        1.4%            98.0%

**Arorae is 98.0% Protestant and Butaritari, 600 km north, is 82.5% Catholic**, and the share
falls almost monotonically down the chain between them. The Sacred Heart mission worked the
north and the Congregational missions the south, and the census still draws the line.

**SOUTH TABITEUEA IS 12.0% BAHÁ'Í**, against 2.1% nationally, which is the largest concentration
of any minority here and the sort of thing only an island-level table shows.
"""

EXCLUDED = {}

REVIEW = {
    "KPC":
        "-> christianity.reformed.congregational.kpc, a node added for it. **34,464 people, "
        "31.3%.** The Kiribati Protestant Church descends from TWO Congregational missions, "
        "the American Board from 1857 and the London Missionary Society from 1870, which is "
        "what puts it beside `.cccs`, `.cicc`, `.ekt` and `.niue` rather than on its own kind "
        "of node. **The live question is the 2014 union**: the body reconstituted as the "
        "Kiribati Uniting Church, a union of Congregationalists, Evangelicals, Anglicans and "
        "Presbyterians, and about ten thousand members refused and re-formed a separate KPC. "
        "`christianity.united` would fit the KUC half. It is not used here because **the 2015 "
        "census still prints one cell**, and 2015 is the newest year Kiribati publishes "
        "religion with a geography; the 2020 census separates them (KUC 21%, KPC 8%) and has "
        "no island table. Splitting the 2015 cell on 2020's ratio was considered and rejected: "
        "it would be a modelled split of the country's second-largest body, and §2.6 forbids "
        "using a later ratio as a magnitude for an earlier year.",
    "Te Ran":
        "-> other.ki, and this is an admitted failure to identify. 86 people in 2015 and 89 in "
        "the 2020 census, so the office treats it as a body worth its own printed cell, but "
        "**nothing published says what it is**: not the 2015 report, not the 2020 report, not "
        "the Census Atlas 2022's religion section, and not any reachable secondary source. "
        "The name is Gilbertese. It goes to the country's unclassified cell rather than to "
        "`christianity.other` because filing it as Christian would be a claim nothing supports "
        "(§14.4 rule 1). At 0.08% it draws no dot either way; if it is ever identified this is "
        "a one-line change.",
    "Church Of God":
        "-> christianity.holiness. 279 people. `Church of God` names at least two unrelated "
        "families — the Holiness body from Anderson, Indiana and the Pentecostal one from "
        "Cleveland, Tennessee — and the census does not say which. Filed on the holiness node "
        "following Antigua (`ag2001.py`) and Barbados (`bb2010.py`), which face the same "
        "ambiguous cell and resolve it the same way, so the map is at least consistent about "
        "it.",
    "All Nation":
        "-> christianity.baptist. 141 people. The report's header says only `All Nation`, "
        "which names no family; **UNSD table 28's return for the same 141 people calls it "
        "`All Nations Baptist`**, and that is what identifies it. The Yearbook doing the work "
        "the report's own header could not is the same thing that happened in Samoa (§9bk).",
    "Jehova's Witness (Te Koaua)":
        "-> christianity.witnesses. 352 people. The report gives the Gilbertese name beside "
        "the English; `te koaua` is *the truth*, which is the Witnesses' own self-designation "
        "in the language. UNSD's return lists the same 352 under `Te koaua` alone, which is "
        "how the two are known to be one cell.",
}

COLUMNS = {
    # Every category is measured at the island, which is the unit drawn, so no row is derived
    # and nothing rolls up. Recorded per COMMANDS.txt's check_rollup note.
}

MAP = {
    "Roman Catholic":              "christianity.catholic.latin",
    "KPC":                         "christianity.reformed.congregational.kpc",
    "Seventh Day Adventist":       "christianity.adventist",
    "Church Of God":               "christianity.holiness",
    "Latter Day Saints":           "christianity.latterday",
    "Assembly of God":             "christianity.pentecostal.trinitarian",
    "Four Square":                 "christianity.pentecostal.trinitarian",
    "Bahai":                       "bahai",
    "Jehova's Witness (Te Koaua)": "christianity.witnesses",
    "Islam":                       "islam",
    "Te Ran":                      "other.ki",
    "All Nation":                  "christianity.baptist",
    "No religion":                 "unaffiliated",
    "Other":                       "other.ki",
}


def resolve(category):
    """Source category -> node. Kiribati excludes nothing, so this never returns None."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
