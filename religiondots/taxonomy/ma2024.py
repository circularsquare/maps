"""
Arab Barometer `Q1012` (Morocco, waves V to VIII) -> religiondots taxonomy.

Named for the population base (RGPH 2024) and the last wave (VIII, December 2023 to January 2024).
`sources/ma.py` and `sources/ma.md` have the construction.

Moroccans, 36,680,178, as drawn:

    99.674%  Muslim       -> islam
     0.215%  Christian    -> christianity
     0.101%  No religion  -> unaffiliated
     0.011%  Other        -> other.ma

Foreign residents, 148,152, arrive already on nodes (`data/normalized/ma_foreign.csv`): HCP's
national nationality mix through `taxonomy/origin_religion.py`, with Muslim branches folded to
`islam` in `sources/ma.py`. Every row is `modelled` (spec §7b), because no Moroccan census asks
religion.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as Algeria (taxonomy/dz2022.py). The card has one Muslim "
        "box. The follow-up records a label rather than a school: Afrobarometer asks Moroccans "
        "(R5-R9, 5,981 answers) with Sunni, Shia and the Sufi orders on the card, and 92.8% say "
        "`Muslim only`. The foreign half's Sunni and Shia shares are folded to islam in "
        "sources/ma.py, so the country has one Muslim row and a Senegalese Muslim is not drawn "
        "as more specifically Sunni than a Moroccan one.",
    "Christian":
        "-> christianity, the bare family node. The follow-up (`Q1012A_CHRISTIAN`, waves VII and "
        "VIII) was put to 14 of the 19 Christians: 12 `Just a Christian`, 1 Catholic, 1 refused, "
        "which names no church. 12 of the 19 are in wave VIII; waves V to VII hold 7 in 7,976 "
        "answers. Pew 2020 has 0.085% Christian for the whole country, against 0.215% of Moroccans "
        "drawn here.",
    "No religion":
        "-> unaffiliated. Wave V's card offers `Atheist` and no Moroccan chose it; VI-1 to VIII "
        "offer `No religion`, 15 answers. Pew 2020: 0.131% unaffiliated for the whole country, "
        "against 0.101% of Moroccans drawn.",
    "Other":
        "-> other.ma, a new per-country residual node, as other.dz. Two answers: `Other` in wave "
        "VI-2 (Casablanca-Settat) and `Something else: SPECIFY` in VI-3 (Marrakech-Safi), merged in "
        "sources/ma.py::RECODE because VI-3's card has no plain `Other`; the file records no "
        "text. The foreign half's Pew `Other religions` share lands here too wherever "
        "origin_religion.py has no split for the nationality.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "No religion": "unaffiliated",
    "Other": "other.ma",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for an Arab Barometer answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
