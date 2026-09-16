"""
Arab Barometer `Q1012` (Algeria, waves V to VII) -> religiondots taxonomy.

Named for the last wave in the pool (VII, fielded June and July 2022), the registry convention.
`sources/dz.py` and `sources/dz.md` have the construction.

    99.57%  Muslim        -> islam
     0.23%  No religion   -> unaffiliated
     0.17%  Christian     -> christianity
     0.04%  Other         -> other.dz

Shares are as drawn: the survey's non-Muslim share for Kabylie and for the rest of Algeria, split at
the pooled national composition, on the 2008 census count of each wilaya. Every row is `modelled`
(spec §7b), because no Algerian census has asked religion.
"""

EXCLUDED = {}

REVIEW = {
    "Muslim":
        "-> islam, the bare family node, not `islam.sunni`. The card has one Muslim box. The "
        "follow-up (`Q1012A`, `Q1012A_MUSLIM`) is on file for waves V and VII, and §11af measured "
        "45-82% of North African respondents answering `Just a Muslim` on it, so it records a "
        "label and not a school; Egypt and Jordan declined it for the same reason. Algeria's "
        "Ibadis (the Mozabites of Ghardaïa) are 7 answers in this pool and no source gives their "
        "magnitude by wilaya, and Anita ruled on 2026-09-16 that the Maghreb gets no presence "
        "rings, so nothing Ibadi is drawn.",
    "No religion":
        "-> unaffiliated. Wave V's `Atheist` (12 answers) and waves VI-VII's `No religion` (7) "
        "are merged in `sources/dz.py::RECODE`: each is code 4 on its own card and no card "
        "offers both, so they are one box on two cards. Pew 2020 puts Algeria's religiously "
        "unaffiliated at 1.27% against 0.23% drawn; the answer is given to an interviewer in a "
        "country where apostasy is not recognised, so the drawn figure is a floor.",
    "Christian":
        "-> christianity, the bare family node. The card has one Christian box. The follow-up "
        "names a church for 11 of the 14 Christians in the pool (Catholic 5, Orthodox 2, Coptic "
        "1, `Just a Christian` 3), which cannot carry a split, so nothing is placed below the "
        "family. Pew 2020: 0.29%, against 0.17% drawn.",
    "Other":
        "-> other.dz, a new per-country residual node, following other.gm and other.ne. Three "
        "answers, all in wave VI-1 (Tizi Ouzou 2, Sétif 1); the file records no text for them. "
        "0.04% as drawn; Pew 2020's `Other religions` is also 0.04%.",
}

MAP = {
    "Muslim": "islam",
    "Christian": "christianity",
    "No religion": "unaffiliated",
    "Other": "other.dz",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, and the roll-up is about where a
# DERIVED row was counted. Nothing here was counted anywhere.


def resolve(category):
    """religiondots branch for an Arab Barometer answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
