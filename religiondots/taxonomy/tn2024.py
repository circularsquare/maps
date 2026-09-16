"""
Arab Barometer `Q1012` (Tunisia, waves V to VIII) -> religiondots taxonomy.

Named for the population base (RGPH 2024) and the last wave (VIII, 2023 to 2024).
`sources/tn.py` and `sources/tn.md` have the construction.

Tunisia, 11,972,169, as drawn (one national share and mix in every governorate):

    99.381%  Muslim       -> islam
     0.252%  No religion  -> unaffiliated
     0.234%  Other        -> other.tn
     0.133%  Christian    -> christianity

Every row is `modelled` (spec §7b), because no Tunisian census asks religion.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as Algeria and Morocco (taxonomy/dz2022.py, ma2024.py). "
        "The card has one Muslim box. The follow-up records a label rather than a school: in "
        "waves VII and VIII 'Just a Muslim' is 55% and 52% of Tunisian Muslims, 'Maliki' 23% "
        "and 15%, and the Afrobarometer (R5-R9, 5,959 answers) finds 97% 'Muslim only'.",
    "Christian":
        "-> christianity, the bare family node. 12 answers; the follow-up reached two of them "
        "(wave VII Catholic, wave VIII 'Just a Christian'), which names no church. Pew 2020 has "
        "0.247% Christian for Tunisia against 0.133% drawn.",
    "No religion":
        "-> unaffiliated. Wave V's `Atheist` (7 answers) and VI-1 to VIII's `No religion` (21), "
        "merged in sources/tn.py::RECODE as each card's one box for having none. Pew 2020: "
        "0.441% unaffiliated, against 0.252% drawn.",
    "Other":
        "-> other.tn, a new per-country residual node, as other.dz and other.ma. 25 answers with "
        "no text recorded: wave V `other` 6, VI-1 1, VI-2 1, VI-3 `Something else: SPECIFY` 3, "
        "VII 14. Ten of VII's fourteen are in Le Kef, Siliana and Sousse, in consecutive PSUs, "
        "where waves V to VI-3 found one non-Muslim in 580 interviews; kept as recorded "
        "(sources/tn.md §5). Waves VI to VIII have no Jewish box, so a Jewish respondent could "
        "only answer here; none of the 25 is in Médenine.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "No religion": "unaffiliated",
    "Other": "other.tn",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for an Arab Barometer answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
