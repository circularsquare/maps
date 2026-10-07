"""
United Arab Emirates, 2024 population base -> religiondots taxonomy.

Named for the base year: each emirate's 2024 count, built in `sources/ae.py` from the emirates'
own offices and FCSC's 2024 national total. No source asks anyone in the UAE their religion;
`sources/ae.py` and `sources/ae.md` have the construction.

Emiratis, 1,519,227 (each emirate's newest count of Emiratis grown to 2024), as drawn:

    100%  Emirati citizens  -> islam

Non-Emiratis, 9,775,016, arrive already on nodes (`data/normalized/ae_foreign.csv`): one national
mix of UN DESA 2024's origins through `taxonomy/origin_religion.py`, Muslim branches folded to
`islam` in `sources/ae.py`. Every row is `modelled` (spec §7b).
"""

EXCLUDED = {}

REVIEW = {
    "Emirati citizens":
        "-> islam, the bare family node, as Saudi Arabia, Oman and Kuwait. NOBODY WAS ASKED: the "
        "2005 census form has no religion item, no emirate census since publishes a religion "
        "table that was found, and the UAE is in no Arab Barometer wave. No source counts an "
        "Emirati who is not Muslim, so every non-Muslim dot is a foreign resident. NOT SPLIT "
        "INTO SUNNI AND SHIA: asks 040 and 043's ruling (Gulf citizens on one Islam where nothing "
        "places a sect); no figure below the nation was looked for beyond that ruling.",
    "Gulf origins":
        "In sources/ae.py, not here: UN DESA's migrants from Bahrain, Kuwait, Qatar and Saudi "
        "Arabia (73,719) are drawn on islam rather than at Pew's rows for those countries, which "
        "are mostly their own foreign residents (Kuwait's Pew row is 80.2% Muslim; PACI counts "
        "Kuwaitis 99.978% Muslim). A migrant from a Gulf state is most likely its citizen. "
        "Reversing it draws 17,334 more non-Muslims, 15,433 of them on Kuwait's row.",
}

MAP = {
    "Emirati citizens": "islam",
}

# No COLUMNS dict (spec §7a-i-1): every row is `modelled`, nothing here was counted anywhere, and
# countries/ae.py writes roll=NOWHERE so the UAE empties under `inferred dots: not shown`.


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
