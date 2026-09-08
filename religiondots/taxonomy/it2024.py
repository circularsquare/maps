"""ESS `rlgdnm` -> religiondots taxonomy. Italy's Italian-citizen half.

The European Social Survey asks `rlgblg` — do you consider yourself as belonging to any
particular religion or denomination — and only then `rlgdnm`, which one. So `Not applicable`
below is not a non-response: it is everyone who answered no to the first question, and in
Italy it is 23% of citizens and the second largest answer.

    1   Roman Catholic                 -> christianity.catholic.latin
    2   Protestant                     -> christianity.protestant
    3   Eastern Orthodox               -> christianity.orthodox.canonical
    4   Other Christian denomination   -> christianity
    5   Jewish                         -> judaism
    6   Islam                          -> islam.sunni
    7   Eastern religions              -> other.it
    8   Other Non-Christian religions  -> other.it
    66  Not applicable                 -> unaffiliated
    77/88/99  Refusal / DK / No answer -> excluded, spec §3.5

**ITALY IS THE ONLY LARGE ESS COUNTRY WITH NO COUNTRY-SPECIFIC DENOMINATION VARIABLE, AND
THAT IS WHY THIS FILE IS SHORT.** Rounds 6-11 carry `rlgdnade`, `rlgdnach`, `rlgdnanl`,
`rlgdnbat`, `rlgdnagr`, `rlgdnapl` and about twenty more — Germany gets its Landeskirchen,
Switzerland its cantonal churches, the Netherlands its whole Protestant taxonomy. **There is
no `rlgdnait`.** Italian respondents only ever meet the harmonised eight-code list above, so
the deepest this country can go on ESS is `christianity.catholic.latin` — no rite, no
movement, no order. Every REVIEW note below is a consequence of that one absence.

THE CATEGORY LIST IS THIN AND THE MINORITY CELLS ARE SINGLE DIGITS. Pooled over the two
NUTS 2 rounds the whole non-Catholic, non-unaffiliated slice is 79 weighted respondents; over
the three NUTS 1 rounds it is 195. Read the Catholic and unaffiliated shares; distrust
everything below one percent, and see `Jewish` below for the one that is outright wrong.
"""

EXCLUDED = {
    "Refusal": "ESS's 77. A refusal to answer, so spec §3.5 marks it rather than filling "
               "it. 13 of 7,663 citizen respondents in the NUTS 1 pool.",
    "Don't know": "ESS's 88, and the same treatment.",
    "No answer": "ESS's 99, and the same treatment.",
}

REVIEW = {
    "Roman Catholic":
        "-> christianity.catholic.latin. 74.3% of citizens and by a wide margin the largest "
        "single node Italy puts on the map. **The one thing folded into it that a better "
        "source would separate is the Arbereshe**, the Italo-Albanian Byzantine-rite "
        "Catholics of the eparchies of Lungro and Piana degli Albanesi — roughly 60,000 "
        "people in Calabria and Sicily who are in full communion with Rome but are not "
        "Latin-rite, and who therefore belong on `christianity.catholic.eastern`. ESS cannot "
        "see them: there is no Italian denomination variable and the harmonised list has one "
        "Catholic code. Recorded per §2.4 so that the day a source resolves it the fix is a "
        "lookup rather than a discovery. Not asserted now (§14.4).",
    "Eastern Orthodox":
        "-> christianity.orthodox.canonical, the BARE canonical node and deliberately not a "
        "national church. gr2024.py names the Church of Greece because a Greek national "
        "sample of Orthodox citizens is that church's faithful and almost nothing else. "
        "Italy is the opposite case: its Orthodox are split between the Romanian Orthodox "
        "Diocese of Italy — the largest, and the reason Romanian is the country's biggest "
        "foreign citizenship — the Ecumenical Patriarchate's Sacra Arcidiocesi Ortodossa "
        "d'Italia, and Serbian, Ukrainian, Russian and Bulgarian jurisdictions. Naming any "
        "one of them would invent a fact. The citizen half is in any case the small half of "
        "Italian Orthodoxy: most Orthodox residents are foreign nationals and arrive through "
        "the nationality model instead.",
    "Protestant":
        "-> christianity.protestant, and **the cell is far too small, which is diagnosis "
        "rather than noise.** Ten weighted respondents give ~71,000 Protestant citizens; "
        "CESNUR counts 313,000 Pentecostals and ~65,000 historic Protestants. The gap is "
        "almost exactly filled by `Other Christian denomination` (~297,000), which is the "
        "finding: **Italian Pentecostals and Jehovah's Witnesses do not call themselves "
        "Protestant on a survey form**, and the two cells are only meaningful added "
        "together. sources/it.md carries the arithmetic. Nothing is moved between them here "
        "— that would be inventing a magnitude to fix a label (§14.4) — but a reader of the "
        "Protestant node alone is reading about a fifth of the people it names.",
    "Other Christian denomination":
        "-> christianity, the root, following ru2012.py, mk2021.py and gr2024.py: the answer "
        "names no body and rules out the three that have codes of their own, so the parent "
        "is the honest place. **In Italy it is a large and identifiable mixture** — the "
        "Assemblee di Dio in Italia, the country's largest non-Catholic Christian body; the "
        "Jehovah's Witnesses, who are among the largest per head in Europe; the Waldensians "
        "and Methodists; and the Romanian and Filipino evangelical congregations of "
        "naturalised citizens. Identifiable, and not separable by anything ESS asks.",
    "Jewish":
        "-> judaism, **and this is the one cell on Italy that is wrong rather than merely "
        "thin.** Nine weighted respondents scale to ~64,000 Jewish citizens. The Unione "
        "delle Comunita Ebraiche Italiane has roughly 24,000 registered members and no "
        "serious estimate of Italian Jewry exceeds about 30,000. The map draws it because "
        "dropping a real community to protect an estimate is worse than drawing it with the "
        "error named (§3.9b), but it is named: sources/it.md and `note_public` both say the "
        "Jewish figure is roughly double, and it is the reason no Italian minority should be "
        "read off a single region.",
    "Islam":
        "-> islam.sunni. Italy's Muslims are overwhelmingly Sunni — the Moroccan, Albanian, "
        "Bangladeshi, Pakistani, Egyptian and Senegalese communities that dominate both the "
        "citizen and the foreign half are Sunni almost throughout. **The Shia presence is "
        "real and is not separable**: a few tens of thousands, largely Iranian, Iraqi and "
        "Lebanese, with centres in Rome and Milan, and neither ESS nor the nationality model "
        "resolves them below the national level. Said here rather than silently rolled in. "
        "This is the one minority cell whose magnitude cross-checks well — ~494,000 Muslim "
        "citizens against CESNUR's 417,900.",
    "Eastern religions":
        "-> other.it with 'Other Non-Christian religions', for gr2024.py's, fr2024.py's, "
        "hr2021.py's and ru2012.py's reason: the tree has no node for 'some Eastern "
        "religion, unspecified' and picking one would invent a fact. ESS does not offer "
        "Buddhism or Hinduism separately. See branches.py's `other.it` for what is inside "
        "it, which in Italy is unusually knowable.",
    "Not applicable":
        "-> unaffiliated. Not a missing value: it is everyone who answered NO to `rlgblg`, "
        "23.2% of Italian citizens, and the second largest answer in the country. "
        "branches.py's line is whether a POSITION is stated — `unaffiliated` is a report of "
        "not belonging, `secular` is a stated non-theistic stance — and 'I do not belong to "
        "a religion' is plainly the first. **Nothing in Italy reaches `secular`**, because "
        "ESS never offers atheist or agnostic as a denomination; gr2024.py, fr2024.py and "
        "ge2014.py all make the same call for the same reason.",
}

MAP = {
    "Roman Catholic": "christianity.catholic.latin",
    "Protestant": "christianity.protestant",
    "Eastern Orthodox": "christianity.orthodox.canonical",
    "Other Christian denomination": "christianity",
    "Jewish": "judaism",
    "Islam": "islam.sunni",
    "Eastern religions": "other.it",
    "Other Non-Christian religions": "other.it",
    "Not applicable": "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
