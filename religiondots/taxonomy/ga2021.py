"""Afrobarometer Gabon (rounds 6-9, 2015-2021) with the DHS 2019-21 Christian split -> religiondots
taxonomy.

Named for the last round's fieldwork year. `sources/ga.py` has the construction and `sources/ga.md`
the record: Gabonese citizens only, on the 2026 census count; five survey groups at one national
mix, with Woleu-Ntem's Christian share and Nyanga's None share their own (the standout rule); the
Christian group divided at the DHS report's national ratio. Every row is `modelled` (spec §7b).

    31.54%  Catholique                 -> christianity.catholic.latin
     9.79%  Protestante                -> christianity.protestant
    37.41%  Église de réveil           -> christianity.pentecostal
     4.13%  Autre religion chrétienne  -> christianity.other
     1.68%  Muslim                     -> islam
     1.30%  Traditional/ethnic religion -> indigenous.african
    13.02%  None                       -> unaffiliated
     1.13%  Other                      -> other.ga (NEW)
"""

EXCLUDED = {}

_SPLIT = ("The survey's Christian group (82.9% as drawn) divided at the DHS 2019-21 report's "
          "national ratio (Tableau 3.1, women 15-49 and men 15-59 combined at the 2026 census's "
          "48.8% men): Catholic 38.1%, Protestant 11.8%, revival churches 45.1%, other Christian "
          "5.0% of Christians. The survey cannot split them itself: `Christian only` runs "
          "20.9-45.1% by round and Roman Catholic 36.7-21.7% with it (sources/ga.py). The DHS "
          "ratio is of all residents, foreigners included (14.6% of its women and 19.0% of its men "
          "are `Autres nationalités`), and the same in every province. Its one check against the "
          "survey: Catholics are 29.85% of everyone in the DHS and 29.10% in the pooled survey. "
          "The fallback is the bare `christianity` node for the whole group, as Cameroon "
          "(cm2025.py) is drawn.")

REVIEW = {
    "Catholique": "-> christianity.catholic.latin. " + _SPLIT,
    "Protestante":
        "-> christianity.protestant, the family node. In Gabon the DHS's `Protestante` is above "
        "all the Église évangélique du Gabon, a Reformed church from the American Presbyterian "
        "and Paris missions, but the box does not name it, so it stays on the family. " + _SPLIT,
    "Église de réveil":
        "-> christianity.pentecostal, as Congo-Brazzaville's `Eglises de réveil` (cg2007.py): the "
        "name used on both banks of the Congo and in Gabon for the Pentecostal and charismatic "
        "revival churches. The largest single answer in the DHS (35.4% of all residents). "
        "Afrobarometer's own `Pentecostal` box is 4-12% by round; the rest of the revival "
        "churches' members most likely answer `Christian only`. " + _SPLIT,
    "Autre religion chrétienne":
        "-> christianity.other. Unnamed in the report; the survey's small boxes (Orthodox, Church "
        "of Christ, Jehovah's Witness, Adventist, Alliance Chrétienne) suggest what it holds. "
        + _SPLIT,
    "Muslim":
        "-> islam, with no branch: 82 respondents over four rounds, `Sunni only` 2 and `Shia` 1. "
        "1.7% of citizens; the DHS's 8.2% of women and 15.2% of men are mostly the foreign "
        "residents, who are not drawn.",
    "Traditional/ethnic religion":
        "-> indigenous.african. 62 respondents, 1.3%. Bwiti, Mwiri and Djembè (write-ins to "
        "the EGEP 2017 household survey's religion question) are surely wider than this, since the card counts only people who give it as their one religion. "
        "The DHS's `Traditionnelle/animiste` is 1.28% of all residents, the same order.",
    "None":
        "-> unaffiliated. The card offers `Traditional/ethnic religion` as its own box on every "
        "round, so `None` is what these respondents chose with the traditional answer in front of "
        "them (step 2 of the draft no-religion procedure; sources/ga.py::report_card). 13.0% as "
        "drawn, 27.6% in Nyanga (the standout). The DHS's `Sans religion/aucune` is 4.7% of women "
        "and 10.9% of men, lower; it is a different instrument and includes foreigners.",
    "Other":
        "-> other.ga, a per-source residual (spec §3.11): the card's `Other` (53), `Bahai` (3) and "
        "`Jewish` (1). Nobody chose `Other` in round 8, so its pooled share leans on rounds 6-7.",
}

MAP = {
    "Catholique": "christianity.catholic.latin",
    "Protestante": "christianity.protestant",
    "Église de réveil": "christianity.pentecostal",
    "Autre religion chrétienne": "christianity.other",
    "Muslim": "islam",
    "Traditional/ethnic religion": "indigenous.african",
    "None": "unaffiliated",
    "Other": "other.ga",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, a survey share on a count.


def resolve(category):
    """religiondots branch for a source category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
