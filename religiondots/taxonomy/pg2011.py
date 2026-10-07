"""
Papua New Guinea, 2011 census (NSO, 2011 National Report, Table 2.4 and Figure 2.1) -> religiondots
taxonomy.

The categories are the census's own, but no province's mix was printed: `sources/pg.py` fits each
province from the national shares and the one church per province the Summary Indicators name,
so every row is `modelled` (spec §7b). National shares, 2011 (churches as % of Christians, who are
95.6% of citizens):

    Roman Catholic             26.0%  -> christianity.catholic.latin
    Evangelical Lutheran       18.4%  -> christianity.lutheran
    Seventh Day Adventist      12.9%  -> christianity.adventist
    Pentecostals               10.4%  -> christianity.pentecostal
    United Church              10.3%  -> christianity.united
    Other Christian             9.7%  -> christianity.other
    Evangelical Alliance        5.9%  -> christianity.evangelical
    Anglican                    3.2%  -> christianity.anglican
    Baptist                     2.8%  -> christianity.baptist
    Salvation Army              0.4%  -> christianity.holiness.salvation-army
    Kwato Church                0.2%  -> christianity.melanesianindependent
    Non-Christian   1.4% of citizens  -> other.pg
    No Religion     0.0% of citizens  -> unaffiliated (draws nothing)
    Not Stated      3.1% of citizens  EXCLUDED, §3.5 residual
"""

EXCLUDED = {
    "Not Stated":
        "3.1% of citizens in 2011 (2.0% in 2000). §3.5: marked, not filled; it is the not-drawn "
        "part of the bar.",
}

REVIEW = {
    "Evangelical Alliance":
        "-> christianity.evangelical, as Kenya's `Evangelical Churches`. The census counts the "
        "Evangelical Alliance of PNG as one body; it is the fellowship of the evangelical "
        "mission churches of the Western and Southern Highlands, Hela, Enga and Western "
        "(the Evangelical Church of PNG, out of the Unevangelized Fields Mission and the Asia "
        "Pacific Christian Mission, and others), so it names an answer and no single "
        "denomination. The largest church in Western (37.1%) and Hela (19.7%).",
    "United Church":
        "-> christianity.united, as sb2019.py: the United Church in Papua New Guinea is the "
        "1968 union of the Methodist and London Missionary Society churches, and the Solomon "
        "Islands' United Church was part of it until 1996. Largest in Gulf, Central, NCD and "
        "Milne Bay, and in New Ireland in 2000.",
    "Kwato Church":
        "-> christianity.melanesianindependent. The Kwato church came out of Charles Abel's "
        "London Missionary Society station on Kwato Island, Milne Bay, which left the LMS in "
        "1917 and ran as the Kwato Extension Association; that is the node's definition (a "
        "church that came out of a mission and stopped belonging to it). 0.2% of Christians, "
        "about 20,000 people on the 2024 count. No source places it, so the fit spreads it at "
        "one rate like every church the Summary Indicators never name, though its home is "
        "Milne Bay.",
    "Other Christian":
        "-> christianity.other, as bb2010.py, bj2013.py, ee2021.py. 9.7% of Christians in "
        "2011. The 2022 SDES lists more bodies and puts `Other Christian Churches` at 0.89%, so "
        "most of this cell is named churches the 2011 form did not print (Jehovah's Witnesses, "
        "Latter-day Saints, the Christian Brethren and the revival churches), not an unknown "
        "tail. The largest answer in Southern Highlands (then including Hela) in 2000.",
    "Non-Christian":
        "-> other.pg, a per-source residual (§3.11). 1.4% of citizens in 2000 and 2011. No "
        "PNG publication says what is in it; the Baha'i Faith and Islam are both present and "
        "traditional belief may be answered here. Nothing splits it, so it claims nothing.",
    "Pentecostals":
        "-> christianity.pentecostal, the family node: the census's one Pentecostal cell "
        "holds the Assemblies of God, the Christian Life Centres and the rest, and does not "
        "say which.",
}

MAP = {
    "Roman Catholic": "christianity.catholic.latin",
    "Evangelical Lutheran": "christianity.lutheran",
    "Seventh Day Adventist": "christianity.adventist",
    "Pentecostals": "christianity.pentecostal",
    "United Church": "christianity.united",
    "Other Christian": "christianity.other",
    "Evangelical Alliance": "christianity.evangelical",
    "Anglican": "christianity.anglican",
    "Baptist": "christianity.baptist",
    "Salvation Army": "christianity.holiness.salvation-army",
    "Kwato Church": "christianity.melanesianindependent",
    "Non-Christian": "other.pg",
    "No Religion": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`.


def resolve(category):
    """religiondots branch for a 2011 census category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
