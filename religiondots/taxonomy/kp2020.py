"""
Pew Research Center's 2020 composition for North Korea (the World Religion Database, through Pew),
grouped in `sources/kp.py` -> religiondots taxonomy.

Named for the year of the mix. `sources/kp.py` and `sources/kp.md` have the construction: one national
mix in every province on the 2008 census, every row `modelled` (spec §7b), because no North Korean
census or survey has ever asked religion. Pew's `Other religions` arrives already split into the
WRD's three parts (Anita, ask 052, 2026-10-03).
"""

EXCLUDED = {}

REVIEW = {
    "No religion":
        "-> unaffiliated. Pew's `Religiously_unaffiliated`, 72.87%, the World Religion Database's "
        "agnostics (57.29%) and atheists (15.58%) added together (ARDA's view of the WRD, 2025). "
        "Nobody was asked: in a state that punishes religious practice this is an estimate of "
        "what people would say, not of what they believe, and the note says so.",
    "New religionists":
        "-> eastasiannew.korean.cheondogyo. The WRD's 12.88%, Pew's `Other_religions` split in the "
        "WRD's proportions; in Korea the WRD's new religionists are Cheondogyo. Drawn on Anita's "
        "ruling on ask 052 (2026-10-03), which reversed the first build's single `other.kp`. "
        "It draws about 3.0 million people, some 45 times South Korea's counted 65,964 and 200 "
        "times the 15,000 the government gave the UN Human Rights Committee in 2002; the note "
        "gives both figures and says this is the WRD's estimate. Some of the WRD's new "
        "religionists may be Jeungsanist or Daejonggyo, but nothing splits them, and Cheondogyo is "
        "the Korean new religion with a state-sanctioned body there (the Korean Chondoist "
        "Society, and the Chondoist Chongu Party).",
    "Ethnic religionists":
        "-> indigenous.korean, added for this row. The WRD's 12.28%, which in Korea is shamanism "
        "(musok). A single-country legend row, which ask 052 approved. Not on "
        "`indigenous.northeurasian`, where Mongolia's shamanists are: that node is Russia's census "
        "answer and its label does not reach Korea.",
    "Chinese folk religionists":
        "-> chinesefolk. The WRD's 0.06%, about 14,000 people; presumably North Korea's ethnic "
        "Chinese (hwagyo), which nothing confirms. About 14 dots.",
    "Buddhist":
        "-> buddhism, the bare family node. The WRD files all of them as Mahayana, which is "
        "certainly right for Korea, but the Taiwan ruling (ask 025) keeps a country on plain "
        "`buddhism` where its source names no school, and South Korea's census Buddhists are on "
        "the bare node too (`kr2015.py`).",
    "Christian":
        "-> christianity, the bare family node. Pew's 100,372 is the WRD's (the Center for the "
        "Study of Global Christianity), of whom the WRD calls 0.35 points 'Independents', the "
        "underground churches. Open Doors says 400,000 and UN estimates 200,000 to 400,000 "
        "(State Department IRF report, 2022); the government said 12,800 in 2002. Nothing "
        "splits or places any of them.",
}

MAP = {
    "No religion": "unaffiliated",
    "New religionists": "eastasiannew.korean.cheondogyo",
    "Ethnic religionists": "indigenous.korean",
    "Chinese folk religionists": "chinesefolk",
    "Buddhist": "buddhism",
    "Christian": "christianity",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for a grouped compiler family, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
