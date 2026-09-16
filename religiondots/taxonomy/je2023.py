"""
Jersey Opinions and Lifestyle Survey 2023 (Statistics Jersey), with the 2018 round's churches
-> religiondots taxonomy.

Survey shares laid on the end-2023 population, 104,030; one geography, the island. See
`sources/je.py` and `sources/je.md`.

    No religion                        50.00%  -> unaffiliated
    Catholic                           17.96%  -> christianity.catholic.latin       REVIEW
    Church of England                  14.01%  -> christianity.anglican             REVIEW
    Not sure                           11.00%  -> EXCLUDED (the gap)                REVIEW
    Other Christian denomination        4.31%  -> christianity                      REVIEW
    Religion other than Christianity    2.73%  -> other.je   (a NEW node)           REVIEW

**The form asks "Do you regard yourself as having a religion?" (Yes, No, Not sure) and then
"If yes, which?" as a write-in**, in 2015, 2018 and 2023 alike. Every religious category here
is the office's coding of that write-in, reported only as rounded percentages.

Catholic is the write-in `Catholic` or `Roman Catholic` (the 2015 report's words), Latin Church:
Jersey is in the Diocese of Portsmouth, and the Portuguese, Madeiran and Polish Catholics the
2015 report names are Latin rite too. Church of England is `Anglican` or `Church of England`.
Neither node is arguable; the shares under them are, which is why they are in REVIEW.

EXCLUDED holds categories that are deliberately not on the tree.
REVIEW holds calls that are defensible but arguable, with the reason.
"""

EXCLUDED = {
    "Resident population, end of 2023":
        "the universe, 104,030 (Statistics Jersey's revised estimate), not a religion "
        "category. Kept so tools/gap_share.py can confirm the not-sure share from the file.",
    "Not sure":
        "11.00% of adults, 11,443 people on the base. A real answer to the yes/no question, "
        "and not a religion. See REVIEW.",
}

REVIEW = {
    "Catholic":
        "-> christianity.catholic.latin, on a share from a DIFFERENT ROUND. 2023's report gives "
        "no church split, only that 93% of those who named a religion named Christianity. The "
        "shares under Christianity are the 2018 report's, of respondents who named a specific "
        "denomination: Catholic 50%, Church of England 39%, other 12% (sum 101, divided by "
        "101). Anita's ask 003 allows a mixed vintage where the method is sound. Two things "
        "rest on it: that the churches moved little from 2018 to 2023, and that respondents "
        "who wrote plain 'Christian' split as the named ones did (neither report says how many "
        "there were). The 2015 report had Catholic 43, Anglican 44, other 13. Built from "
        "2023's religion by place of birth and 2015's churches by place of birth, the split "
        "reads 44.3/42.4/13.3 (sources/je.md §3), so the 2018 figures lean Catholic by about "
        "5 points against that reading. The drift since 2018 has no settled direction: the "
        "share with a religion fell among the Jersey-born (39% to 30%) and British-born (51% "
        "to 38%), who are mostly Anglican, and rose among the Portugal- and Madeira-born (59% "
        "to 68%), all Catholic (2023 report Figure 9.8), which would push Catholics up, while "
        "the birthplace reading pushes them down.",

    "Church of England":
        "-> christianity.anglican, on 2018's 39%; the same reasoning as `Catholic`.",

    "Other Christian denomination":
        "-> christianity, the parent, as Gibraltar's `Other Christian` (gi2022.py). A residual "
        "of whatever write-ins were not Catholic or Anglican; no report names what is in it. "
        "2018's 12%, the same reasoning as `Catholic`.",

    "Religion other than Christianity":
        "-> other.je, a per-source residual (spec §3.11). **7% of those who named a religion in "
        "2023, 2.73% of adults.** No round prints it apart; the 2015 report names Buddhist, "
        "Hindu, Jewish, Muslim and Sikh answers, each from very small numbers of respondents. "
        "Putting all of it on one world religion's node would guess which.",

    "Not sure":
        "EXCLUDED, into `gap`, and NOT `unknown` or a node. The survey's analogue is ESS's "
        "'don't know' to 'do you belong to any religion', which fr2024, gr2024 and it2024 "
        "exclude. `unknown` (spec §6.3a-ii) is for people a source counted whose religion its "
        "answer set could not record; these respondents were offered a write-in and chose "
        "not sure. It is 11% (2015 7%, 2018 not printed in the report), so the legend's bar "
        "shows a real hole, which is the point of drawing it that way.",
}

COLUMNS = {
    # Every category is on the island, which is the unit drawn, so no row is derived and nothing
    # rolls up. Recorded per COMMANDS.txt's check_rollup note.
}

MAP = {
    "Catholic":                         "christianity.catholic.latin",
    "Church of England":                "christianity.anglican",
    "Other Christian denomination":     "christianity",
    "Religion other than Christianity": "other.je",
    "No religion":                      "unaffiliated",
}


def resolve(category):
    """Source category -> node, or None for a category deliberately not on the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
