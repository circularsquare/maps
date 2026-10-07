"""Luxembourg -> religiondots taxonomy.

The five categories are the ones both surveys print (sources/lu.py): EVS 2020/21 in STATEC's
Regards 03/23 and TNS Ilres 2022 in AHA's press release. Everyone is drawn at the two surveys'
mean, one national mix on every commune, so every row is `modelled`.

    Catholic          -> christianity.catholic.latin
    Protestant        -> christianity.protestant
    Muslim            -> islam.sunni
    Other religion    -> other.lu
    No religion       -> unaffiliated
"""

CATHOLIC = "Catholic"
PROTESTANT = "Protestant"
MUSLIM = "Muslim"
OTHER = "Other religion"
NO_RELIGION = "No religion"

SOURCE = {CATHOLIC, PROTESTANT, MUSLIM, OTHER, NO_RELIGION}

EXCLUDED = {}

REVIEW = {
    CATHOLIC:
        "-> christianity.catholic.latin. Luxembourg is one Latin archdiocese; both surveys' word is "
        "`catholique` / `katholisch`, and EVS's card has no Eastern-rite answer.",
    PROTESTANT:
        "-> christianity.protestant. EVS's figure is its `Protestants` and `Evangelistes` bars "
        "together (STATEC: Christians 92% of those who belong, Catholics 85.3%); AHA's is "
        "`Protestanten`. The Protestant Church of Luxembourg is Lutheran and Reformed in one body, "
        "and the surveys name neither, so the Protestant parent and not a branch.",
    MUSLIM:
        "-> islam.sunni, the ESS countries' call (dk2024.py, se2024.py). The surveys name no branch; "
        "the census's largest foreign nationalities from Muslim-majority or mixed countries are "
        "Montenegro 2,862, Syria 2,688, Morocco 1,612 and Bosnia 1,573 (cens_21ctz_r3), Sunni in "
        "the main, and naturalised Luxembourgers of the same origins are not counted by nationality. "
        "Shia and Alevi residents are not separable.",
    OTHER:
        "-> other.lu. EVS's `Juifs`, `Bouddhistes` and `Autres` bars (not printed as numbers, so "
        "taken together as the remainder of those who belong) and AHA's `andere Religion`. Orthodox "
        "Christians are not named by either survey and can have answered here, so they are likely "
        "inside this node rather than on christianity.orthodox; nothing printed splits them out.",
    NO_RELIGION:
        "-> unaffiliated. EVS: does not belong to a denomination (52%); AHA: gehören keiner "
        "Religion an (41%, 23 points of them former Catholics).",
}

MAP = {
    CATHOLIC: "christianity.catholic.latin",
    PROTESTANT: "christianity.protestant",
    MUSLIM: "islam.sunni",
    OTHER: "other.lu",
    NO_RELIGION: "unaffiliated",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
