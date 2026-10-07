"""
Tajikistan (no published source places religion; `sources/tj.py`) -> religiondots taxonomy.

Named for the population base, the 2020 census's permanent population. Every category is a cell of
the ethnicity model in `sources/tj.py`: a non-Muslim-heritage group's 2020 count (2010 scaled by the
Russians' fall to the Agency's 0.3%) placed on its 2010 regional distribution and given a religion,
with everyone else on Islam. Every row is `modelled` (spec §7b); `sources/tj.md` has the
construction and the witnesses.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as az2019.py, Mauritania and Afghanistan. Every "
        "nationality of Muslim heritage (Tajiks, Uzbeks and the Uzbek tribal rows, Kyrgyz, "
        "Turkmens, Arabs, Afghans and the rest; sources/tj.py MUSLIM_HERITAGE), plus the Muslim "
        "share Kazakhstan's census gives Russians, Tatars, Ukrainians, Belarusians and Germans. "
        "LiTS III (99.5%, sources/tj.md §3) and Pew's 2011-12 survey (98.9%) agree for the "
        "majority. NO SECT: the Pamiris of Gorno-Badakhshan are Nizari Ismailis and stay here, "
        "as cn2000.py keeps Taxkorgan's Ismaili Tajiks on islam; drawing them is ask 051 (§14). "
        "Pew's 2011-12 sect question is national only (om/sa ruling, 2026-09-15).",
    "Orthodox (Kazakhstan's coefficient)":
        "-> christianity.orthodox, the branch, as az2019.py and kz2021.py. Tajikistan's Russians "
        "are under the Russian Orthodox eparchy of Dushanbe, but the coefficient's source does "
        "not name a church. Includes Germans at Kazakhstan's own German row (89.1% Orthodox "
        "once refusals and Catholics are out); the Catholics and Protestants among them (3.6% in "
        "Kazakhstan) are dropped, about 13 people.",
    "Non-believer (Kazakhstan's coefficient)":
        "-> secular, as az2019.py and kz2021.py file `Неверующие`. The weakest cell drawn: "
        "attitude-shaped, and Kazakhstan's held-out test missed it by -13% urban and +42% rural "
        "(spec §14.12).",
    "Armenian Apostolic":
        "-> christianity.oriental.armenian (ask 004), the 2010 census's 434 Armenians scaled to "
        "2020 and placed where the Russians are. Religio-ethnic (spec §14.5). Under one dot.",
    "Georgian Orthodox":
        "-> christianity.orthodox.canonical.georgian, the census's 92 Georgians. Under one dot.",
    "Jewish":
        "-> judaism, the census's 34 Jews and 2 Central Asian (Bukharan) Jews; the nationality is "
        "a religious community by definition (spec §14.5). Under one dot.",
    "Other nationality, religion not known":
        "-> unknown. Koreans (634 in 2010), Chinese (801), Ossetians (396), Moldovans, the "
        "peoples of India and Pakistan, Karelians, English, Americans and about forty smaller "
        "rows; mixed or unknown religion, printed one nationality at a time but not "
        "classifiable without guessing.",
}

MAP = {
    "Muslim": "islam",
    "Orthodox (Kazakhstan's coefficient)": "christianity.orthodox",
    "Non-believer (Kazakhstan's coefficient)": "secular",
    "Armenian Apostolic": "christianity.oriental.armenian",
    "Georgian Orthodox": "christianity.orthodox.canonical.georgian",
    "Jewish": "judaism",
    "Other nationality, religion not known": "unknown",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`; nothing here was counted as a religion.


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
