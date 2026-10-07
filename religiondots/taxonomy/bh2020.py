"""
Bahrain, 2020 census -> religiondots taxonomy.

The census's religion table (Muslim / Others by nationality and sex, national) carried to the four
governorates by their nationality groups and sex, with the foreign residents' `Others` split
through UN DESA's origins and Pew 2020, and Bahraini Muslims split Shia / Sunni by the two
endowments' mosque counts at Arab Barometer I's level (ask 055; `sources/bh.py`, `sources/bh.md`).

Categories, as `data/normalized/bh.csv` writes them:

    Bahraini, Muslim, Shia                -> islam.shia     modelled: the census's Bahraini
    Bahraini, Muslim, Sunni               -> islam.sunni    Muslims by governorate, split by the
                                                            two endowments' mosque counts at
                                                            Arab Barometer I's national level
    Bahraini, Muslim, no sect given       -> islam          derived (the survey's 3 of 435)
    Bahraini, Others                      -> other.bh       derived, unsplit
    Non-Bahraini, Muslim                  -> islam          derived
    Non-Bahraini, Others, <node>          -> <node>         modelled: the census's `Others`
                                                            split by DESA origin x Pew 2020,
                                                            Christian/Hindu by the Gulf rule
"""

EXCLUDED = {}

REVIEW = {
    "Bahraini, Muslim, Shia":
        "-> islam.shia, not islam.shia.jaafari, though Bahrain's Shia are Ja'fari: nothing measured "
        "the school, and the branch is what the survey asked. Split drawn on Anita's ruling (ask "
        "055, 2026-10-03; spec §14 was hers). Nobody counted sect: each governorate starts at its "
        "Shia share of mosques (Ja'fari Endowments 2016 against Sunni Endowments about 2022) and one "
        "logit shift (+0.055) brings the country to Arab Barometer I's 249 of 435. Drawn 79% of "
        "Bahraini Muslims in Capital, 80% Northern, 22% Muharraq, 21% Southern. Counting ma'tams "
        "with the Ja'fari mosques would give Muharraq 27% and Northern 77% at the same level "
        "(sources.md §bh-2026-10-03c).",
    "Bahraini, Muslim, Sunni":
        "-> islam.sunni, not split into schools (Maliki and Shafi'i both present; nothing counts "
        "them). The complement of the Shia row among the survey's named answers.",
    "Bahraini, Muslim, no sect given":
        "-> islam, bare: 3 of the survey's 435 answered only 'Muslim', 0.7% of Bahraini Muslims "
        "in every governorate (the unspecified-is-fine ruling: the named split, the rest on the "
        "parent).",
    "Bahraini, Others":
        "-> other.bh, 2,295 citizens (938 men, 1,357 women). The form codes Christian, Jewish and "
        "other, but the 2020 table prints only Muslim and Others, so nothing says how many are "
        "Christian; kept as counted rather than folded into Islam as Kuwait's 277 were, because "
        "here they are about two dots and a real count.",
    "Non-Bahraini, Muslim":
        "-> islam, bare. The national count by sex is the census's; the governorate split comes from "
        "each governorate's nationality groups, each group's Muslim share from its DESA origins "
        "through Pew, with the Asian and African groups' shares moved by one logit shift per sex so "
        "the national count closes (sources/bh.py). Pakistani and other Shia not split out.",
    "Non-Bahraini, Others, christianity":
        "-> christianity, bare, not split into churches (Kuwait's reason): the Gulf rule "
        "(origin_religion.gulf_christian_hindu, as the UAE and Oman) moves 102,870 Indians from "
        "Hindu to Christian so Christians / (Christians + Hindus) is Pew 2020's Bahrain 0.550, "
        "taking Christians from the 90,718 DESA's origins give through Pew to 193,586; the "
        "origins' church shares would describe the wrong people. Many of the Gulf's Indians come "
        "from Kerala, 18% Christian in 2011.",
    "Non-Bahraini, Others, hinduism":
        "158,651, 40.9% of non-Muslims, after the Gulf rule (DESA's origins through Pew alone give "
        "67.4%; Pew's Bahrain row 43.1%, the gap being the families the rule leaves alone).",
    "Non-Bahraini, Others, buddhism":
        "as DESA's Sri Lankans and Thais give through Pew, 12,404 (3.2% of non-Muslims). Until "
        "2026-10-03 raked to Pew's Bahrain row, 2,383 (0.6%), but that row is Pew's shared Gulf "
        "template and the UAE and Oman keep the origins' Buddhists (sources.md §bh-2026-10-03b).",
    "Non-Bahraini, Others, other.bh":
        "Pew `Other religions` for origins taxonomy/origin_religion.py does not resolve.",
}

MAP = {
    "Bahraini, Muslim, Shia": "islam.shia",
    "Bahraini, Muslim, Sunni": "islam.sunni",
    "Bahraini, Muslim, no sect given": "islam",
    "Bahraini, Others": "other.bh",
    "Non-Bahraini, Muslim": "islam",
    "Non-Bahraini, Others, christianity": "christianity",
    "Non-Bahraini, Others, hinduism": "hinduism",
    "Non-Bahraini, Others, buddhism": "buddhism",
    "Non-Bahraini, Others, sikhism": "sikhism",
    "Non-Bahraini, Others, jainism": "jainism",
    "Non-Bahraini, Others, unaffiliated": "unaffiliated",
    "Non-Bahraini, Others, judaism": "judaism",
    "Non-Bahraini, Others, indigenous": "indigenous",
    "Non-Bahraini, Others, indigenous.african": "indigenous.african",
    "Non-Bahraini, Others, druze": "druze",
    "Non-Bahraini, Others, other.bh": "other.bh",
}


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
