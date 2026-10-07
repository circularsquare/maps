"""
EPHS 2010 (Eritrea Population and Health Survey, Table 3-1, printed nationally) -> religiondots taxonomy.

Named for the survey year. `sources/er.py` and `sources/er.md` have the construction: the survey's
weighted national shares of women and men 15-49, combined at the survey's own household sex split,
the same mix in every zoba, every row `modelled` (spec §7b), because Eritrea has never held a
census.

    57.39%  Orthodox               -> christianity.oriental.eritrean
    37.18%  Muslim                 -> islam
     4.36%  Catholic               -> christianity.catholic
     0.87%  Protestant             -> christianity.protestant
     0.20%  Traditional believer   -> indigenous.african

The card (questionnaires, PDF pp.460, 524, 555, 0-based 459, 523, 554) has these five codes and an
`OTHER (SPECIFY)` code 6 that Table 3-1 prints no row for (its CORE and ALL columns are 3 and 7
weighted women short of their totals); the card has no `no religion` code.
"""

EXCLUDED = {}

REVIEW = {
    "Orthodox":
        "-> christianity.oriental.eritrean, promoted from an ASARB leaf to a branch for this "
        "(taxonomy/branches.py). In Eritrea `Orthodox` is the Eritrean Orthodox Tewahedo Church, "
        "one of the four bodies the state registers, and there is no Eastern Orthodox community of "
        "any size to share the cell with, which is et2007's reasoning for Ethiopia and the opposite "
        "of ke2019's for Kenya. The fallback is the bare `christianity.oriental` parent.",
    "Catholic":
        "-> christianity.catholic, the family node. Most Eritrean Catholics belong to the Eritrean "
        "Catholic Church, a Ge'ez-rite Eastern Catholic church sui iuris since 2015, and a few to "
        "Latin parishes; the card has one Catholic code, so the cell is not split and does not go "
        "to `christianity.catholic.eastern` (ask 004: record what the source says).",
    "Protestant":
        "-> christianity.protestant. The Evangelical Lutheran Church of Eritrea is the one "
        "registered Protestant body; the cell also holds the Pentecostal and other churches the "
        "state has closed since it required registration in 2002, so it is not filed under `christianity.lutheran`.",
    "Muslim":
        "-> islam. The card has one Muslim code, nothing splits it, and the "
        "rulings for Oman and Saudi Arabia keep a group with no placement on one node.",
    "Traditional believer":
        "-> indigenous.african, the card's own words (`TRADITIONAL BELIEVER`), 0.2%.",
}

MAP = {
    "Orthodox": "christianity.oriental.eritrean",
    "Muslim": "islam",
    "Catholic": "christianity.catholic",
    "Protestant": "christianity.protestant",
    "Traditional believer": "indigenous.african",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, a national survey share on an estimate.


def resolve(category):
    """religiondots branch for an EPHS answer, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
