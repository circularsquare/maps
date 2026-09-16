"""
Arab Barometer `Q1012` (Libya, waves V to VII) -> religiondots taxonomy.

Named for the population base (the Bureau of Statistics and Census's estimate of Libyans by region
for 2020). `sources/ly.py` and `sources/ly.md` have the construction.

Libyans, 6,872,674, as drawn (one national share and mix in every district):

    99.898%  Muslim       -> islam
     0.084%  Christian    -> christianity
     0.018%  No religion  -> unaffiliated

Every row is `modelled` (spec §7b), because no Libyan census asks religion. Non-Libyans are not in
the base and are not drawn.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, as Algeria, Morocco and Tunisia (taxonomy/dz2022.py, "
        "ma2024.py, tn2024.py). The follow-up records a label rather than a school: in wave VII "
        "'Just a Muslim' is 68% of Libyan Muslims, 'Sunni' 20% and 'Maliki' 11%. Wave V's list, "
        "logged by the interviewer, has no Maliki code and records 561 of 1,961 as 'Alawi', a "
        "branch with no Libyan community; they are read as Malikis logged on another code, as "
        "Yemen's Zaydis were (playbooks/arabbarometer.md), and nothing is drawn from it. Five "
        "answers name the Ibadi branch (VII 4, V 1 as 'Mozabite'); too few, and no rings.",
    "Christian":
        "-> christianity, the bare family node. 6 answers (VI-2 3, VI-3 2, VII 1); the follow-up "
        "reached one, 'Just a Christian', which names no church. The survey samples citizens (AB "
        "VII technical report), so these are Libyans. Pew 2020 has 0.525% Christian for everyone "
        "in Libya against 0.084% drawn for Libyans; the difference is mostly foreign workers, who "
        "are not in the base.",
    "No religion":
        "-> unaffiliated. One answer, wave V's `Atheist`, merged with VI-VII's `No religion` in "
        "sources/ly.py::RECODE as each card's one box for having none. Nobody chose it in VI or "
        "VII.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "No religion": "unaffiliated",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for an Arab Barometer answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
