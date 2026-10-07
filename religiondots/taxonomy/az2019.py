"""
Azerbaijan (no source asks religion with a place; `sources/az.py`) -> religiondots taxonomy.

Named for the population base, the 2019 census's existing population. Every category is a cell of
the ethnicity model in `sources/az.py`: a group's 2019 count placed on its 2009 distribution and
given a religion, with everyone outside the modelled groups on Islam. Every row is `modelled`
(spec §7b); `sources/az.md` has the construction and the witnesses.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as Mauritania, Somalia and Afghanistan. Everyone outside "
        "the eight modelled groups, plus the Muslim share Kazakhstan's census gives Russians, "
        "Ukrainians and Tatars. LiTS III (1,510 of 1,510 Muslim), DHS 2006 (99.2% of women 15-49) "
        "and EVS 2017 (99.4% of those naming a religion) agree for the majority. NO SECT: Pew 2011 "
        "(37% Shia, 16% Sunni, 45% 'just a Muslim') and CRRC 2012 are national only, and the om/sa "
        "ruling (2026-09-15) leaves an unplaced split undrawn. An ethnicity-derived Sunni layer "
        "(Lezgins, Avars, Tsakhurs) was considered and refused: it would mark the north's "
        "minorities and leave its Sunni Azerbaijanis on the parent (sources/az.md §6).",
    "Orthodox (Kazakhstan's coefficient)":
        "-> christianity.orthodox, the branch, as kz2021.py files Kazakhstan's `Православие`, "
        "whose share this is. In Azerbaijan the Russians and Ukrainians are almost all under the "
        "Russian Orthodox eparchy of Baku, but the coefficient's source does not name a church. "
        "Ismayilli's and Gadabay's Russians are largely Molokans (Ivanovka, Slavyanka), a "
        "Spiritual Christian body, and are drawn Orthodox anyway: the census does not separate "
        "them and no node exists for them (sources/az.md §6). REVIEW 2026-10-03: Armenia "
        "(am2022.py `Molokai`) files Molokans on christianity.other and says they must not be "
        "Orthodox; move these two units' share there at the next rebuild (sources/az.md §9).",
    "Non-believer (Kazakhstan's coefficient)":
        "-> secular, as kz2021.py files `Неверующие` (non-believers, a stated position, not a "
        "plain 'no religion' box). The weakest cell drawn: non-belief is attitude-shaped and "
        "Kazakhstan's held-out test missed it by -13% urban and +42% rural (spec §14.12).",
    "Georgian Orthodox":
        "-> christianity.orthodox.canonical.georgian. The census's Georgians (8,442 in 2019) are "
        "the Christian Ingiloys and Georgians of Gakh, Zagatala and Balakan and Baku's Georgians; "
        "since 2019 the census has a separate Ingiloy box, and Muslim Ingiloys have mostly been "
        "recorded as Azerbaijani. Religio-ethnic in Azerbaijan in the sense of spec §14.5.",
    "Ingiloy, religion not known":
        "-> unknown. The 1,817 who chose the 2019 census's new Ingiloy box are Georgian-speaking "
        "and either Muslim or Christian, and nothing says which; placed on the 2009 Georgians.",
    "Udi Christian":
        "-> christianity.oriental, the branch. The Udins of Nij (Gabala) and Oghuz belong to the "
        "Albanian-Udi Christian community, registered in 2003, which claims the Church of "
        "Caucasian Albania; that church was miaphysite and in communion with the Armenian "
        "Apostolic Church, and the Nij Udins were under the Armenian church until the 1990s. "
        "Oriental, but not Armenian: the community does not call itself Armenian. Muslim Udins "
        "have been recorded as Azerbaijanis since the nineteenth century, so the category is "
        "religio-ethnic (spec §14.5).",
    "Jewish":
        "-> judaism. Mountain Jews (Quba's Red Town), Ashkenazi and Georgian Jews; the census "
        "nationality is a religious community by definition (spec §14.5).",
    "Armenian Apostolic":
        "-> christianity.oriental.armenian (ask 004). The 178 Armenians the 2019 census counted "
        "outside the Armenian-held territory, placed on the 2009 Armenians outside Karabakh "
        "(Baku 104). Under one dot anywhere.",
    "Other nationality, religion not known":
        "-> unknown. The census's 'other nationalities' (5,039): Belarusians, Kazakhs, Uzbeks, "
        "Germans, Iranians and others, mixed, and printed as one row.",
}

MAP = {
    "Muslim": "islam",
    "Orthodox (Kazakhstan's coefficient)": "christianity.orthodox",
    "Non-believer (Kazakhstan's coefficient)": "secular",
    "Georgian Orthodox": "christianity.orthodox.canonical.georgian",
    "Ingiloy, religion not known": "unknown",
    "Udi Christian": "christianity.oriental",
    "Jewish": "judaism",
    "Armenian Apostolic": "christianity.oriental.armenian",
    "Other nationality, religion not known": "unknown",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`; nothing here was counted as a religion.


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
