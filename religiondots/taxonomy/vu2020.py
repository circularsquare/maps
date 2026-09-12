"""
VNSO 2020 Census of Vanuatu, Basic Tables Vol 1, Table 3.5 -> religiondots taxonomy.

**Branch-level mapping, no nested universe and no leaves.** Table 3.5's fourteen columns are a
flat partition of the population in private households — there is no `Christian` subtotal to
double-count, which is what made Fiji dangerous (fj2007.py). The source's own string travels
with the row in `source_category` (spec §2.4).

    Presbyterian                80,060   27.23%  -> christianity.reformed.presbyterian
    Seventh Day Adventist       43,541   14.81%  -> christianity.adventist
    Catholic                    35,602   12.11%  -> christianity.catholic.latin
    Anglican                    35,339   12.02%  -> christianity.anglican
    Other churches              35,270   12.00%  -> other.vu
    Churches of Christ          14,588    4.96%  -> christianity.restorationist
    Assemblies of God (AOG)     14,450    4.92%  -> christianity.pentecostal.trinitarian
    Neil Thomas Ministry         9,515    3.24%  -> christianity.pentecostal.charismatic
    Customary beliefs            9,080    3.09%  -> indigenous.vanuatu
    Apostolic                    6,894    2.35%  -> christianity.pentecostal
    Latter Day Saints (Mormon)   5,174    1.76%  -> christianity.latterday
    No Religion/Faith            4,023    1.37%  -> unaffiliated
    -----------------------------------------------------------------------------------
    Refuse to answer               394    0.13%  §3.5 residual, EXCLUDED
    Not Stated                      34    0.01%  §3.5 residual, EXCLUDED
    Total                      293,963           universe, EXCLUDED

**NO CHURCH IN VANUATU IS ANYWHERE NEAR A MAJORITY, AND THAT IS RARE IN THE PACIFIC.** The
largest is Presbyterian at 27.2%; on this map Fiji's Methodists are 34.7%, Tonga's Free
Wesleyans and Tuvalu's Ekalesia are far higher again. Four bodies sit between 12% and 15%.

**AND THE PROVINCES DO NOT AGREE WITH EACH OTHER AT ALL.** Anglican is **77.3% of Torba and
0.2% of Tafea**; Presbyterian is 43.9% of Malampa and 0.5% of Torba. Those are mission
spheres — the Anglican Melanesian Mission worked the northern islands from the 1850s, the
Presbyterians the centre and south — and they are still legible 170 years later. Vanuatu is
drawn at 66 area councils, so this is visible at a much finer grain than the six provinces.

**CUSTOMARY BELIEFS ARE A PRINTED CATEGORY, NOT A RESIDUAL**, which is unusual and is the
reason `indigenous.vanuatu` exists. 9,080 people, counted in four consecutive censuses, and
almost entirely on Tanna.
"""

EXCLUDED = {
    "Total": "the unit's own population in private households, not a religion category.",
    "Refuse to answer":
        "394 people, **0.13%**. The census says the religion question was not compulsory, "
        "and this is what refusing it looks like. §3.5: marked, not filled.",
    "Not Stated":
        "34 people, **0.01%**, and the smallest residual on this map. Volume 2 of the same "
        "census has no `Not Stated` row at all and instead quotes 428 for refusals and "
        "non-response together, which is exactly this table's 394 + 34 — so the two volumes "
        "agree and simply cut the residual in different places. Vanuatu is 99.85% drawn.",
}

REVIEW = {
    "Other churches":
        "-> other.vu, and this is the call worth a second opinion in this file. 35,270 "
        "people, **12.0%, the third-largest cell in the country**. Volume 1 heads the column "
        "`Other churches`, which reads as Christian and would send it to the bare "
        "`christianity` node. **Volume 2 of the same census heads it `Other` and then says "
        "what is in it: \"the category 'Other' includes 88 different religions ranging from "
        "one member to more than 2,000 members.\"** Religions, not churches. Vanuatu's "
        "Baha'i, Muslim and Jehovah's Witness communities have nowhere else in this table to "
        "be, so they are in here, and nothing published says in what proportion. Calling the "
        "whole cell Christian on the strength of a column header the other volume "
        "contradicts would be §14.4 rule 1. It goes to the country residual instead, where "
        "it claims nothing.",
    "Neil Thomas Ministry / Inner Life Ministry":
        "-> christianity.pentecostal.charismatic. 9,515 people, 3.24%, and **a "
        "Vanuatu-founded church rather than an imported mission** — one of the few on this "
        "map, beside Fiji's Christian Mission Fellowship (fj2007.py). Neil Thomas Ministries "
        "grew out of an evangelistic ministry in Port Vila and is charismatic in practice "
        "and independent in structure, which is what `charismatic` is for as against the "
        "classical Pentecostal denominations under `trinitarian`. The census pairs it with "
        "`Inner Life Ministry` in one column and says nothing about the relationship between "
        "them, which is carried in `source_category` verbatim per §2.4. It has been a "
        "printed category since 1999 and has grown 6,406 -> 7,223 -> 9,515.",
    "Apostolic":
        "-> christianity.pentecostal, the family node, exactly as in fj2007.py and for the "
        "same reason. 6,894 people, 2.35%. `Apostolic` names the Apostolic Church, the "
        "classical Pentecostal body of Welsh origin with a documented Pacific presence, but "
        "it is also how the **New Apostolic Church** — neither Pentecostal nor evangelical — "
        "is usually printed. The Pentecostal reading is taken because the Apostolic Church "
        "is established in Vanuatu and the New Apostolic Church is not, and because the "
        "surrounding list is otherwise evangelical and Pentecostal bodies. On the family "
        "node either way, so a wrong reading misplaces 2.35% by one branch rather than "
        "across the tree.",
    "Assemblies of God (AOG)":
        "-> christianity.pentecostal.trinitarian, following fj2007.py, ph2020.py, nz2023.py "
        "and au2021.py. 14,450 people, 4.92%. Unambiguously Trinitarian Pentecostal.",
    "Churches of Christ":
        "-> christianity.restorationist. 14,588 people, 4.96%. The Stone-Campbell "
        "Restorationist family, as everywhere else on this map. **Its geography is one "
        "province**: 5,506 of the 14,588 are in Penama, where it is 16.1%, against 0.9% in "
        "Malampa — the Churches of Christ mission worked Pentecost and Ambae specifically.",
    "Customary beliefs":
        "-> indigenous.vanuatu, a node added for it. 9,080 people, 3.09%, **and it is a "
        "printed census category rather than an outsider's residual**, which is what "
        "separates it from most of what sits under `indigenous`. Counted in four consecutive "
        "censuses (6,484 in 1989, 10,365 in 1999, 8,600 in 2009, 9,080 in 2020). Almost "
        "entirely on Tanna, where the John Frum movement is; the census names no movement, "
        "so neither does the tree. Read as a floor — kastom is widely kept alongside church "
        "membership and a one-answer question counts those people as their church.",
    "Presbyterian":
        "-> christianity.reformed.presbyterian. 80,060 people and **27.2% of the country, "
        "the largest body in Vanuatu** — the Presbyterian Church of Vanuatu, out of the "
        "Presbyterian mission to the southern and central islands from 1848. Its share has "
        "fallen every census since 1989: 35.8% -> 27.9% -> 27.2%.",
    "Anglican":
        "-> christianity.anglican. 35,339 people, 12.0% nationally, and **the most "
        "geographically concentrated large church on this map**: 77.3% of Torba and 44.9% of "
        "Penama against 0.2% of Tafea and 0.4% of Malampa. The Anglican Melanesian Mission "
        "worked the Banks and Torres islands and northern Pentecost from the 1850s and "
        "essentially nowhere else in the group. The Church of the Province of Melanesia, "
        "which would be new to this map as a body if any source named it, is not named here.",
    "Latter Day Saints (Mormon)":
        "-> christianity.latterday. 5,174 people, 1.76%, and **new in 2020** — Volume 2's "
        "four-census table has a dash for it in 1989, 1999 and 2009, so this is the first "
        "Vanuatu census to print it as its own category.",
}

MAP = {
    "Presbyterian": "christianity.reformed.presbyterian",
    "Seventh Day Adventist (SDA)": "christianity.adventist",
    "Catholic": "christianity.catholic.latin",
    "Anglican": "christianity.anglican",
    "Churches of Christ": "christianity.restorationist",
    "Assemblies of God (AOG)": "christianity.pentecostal.trinitarian",
    "Neil Thomas Ministry / Inner Life Ministry": "christianity.pentecostal.charismatic",
    "Customary beliefs": "indigenous.vanuatu",
    "Apostolic": "christianity.pentecostal",
    "Latter Day Saints (Mormon)": "christianity.latterday",
    "Other churches": "other.vu",
    "No Religion/Faith": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): every category is measured at the area council it is drawn
# on, so no row is `derived` and nothing ever needs to roll up.


def resolve(category):
    """religiondots branch for a VNSO category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
