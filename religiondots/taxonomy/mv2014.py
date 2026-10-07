"""
Maldives, 2014 Population and Housing Census -> religiondots taxonomy.

The form asks religion (M7) only of foreign residents; Maldivians skip it. `sources/mv.py` and
`sources/mv.md` have the construction.

Maldivians, 338,434, as drawn:

    100%  Maldivian (not asked)  -> islam

Foreign residents, 63,637: the census's own six answers, counted in Malé and for the atolls
together (UNSD table 28, Urban and Rural, less the Maldivians), spread over the 20 atolls by sex,
kind of island and citizenship:

    39,217  Islam        -> islam
    10,163  Hindu        -> hinduism
     6,403  Christian    -> christianity
     5,337  Buddhist     -> buddhism
     1,289  Other        -> other.mv
     1,228  Not Stated   -> not drawn
"""

EXCLUDED = {
    "Not Stated":
        "1,228 foreign residents, 1.9% of them and 0.31% of the country. Leans nowhere in "
        "particular; the form has no box for no religion, so some people with none are probably "
        "here and some in Other.",
}

REVIEW = {
    "Maldivian (not asked)":
        "-> islam. NOBODY WAS ASKED: the 2014 form sends Maldivians past the religion question, "
        "and the 2008 constitution says a non-Muslim may not become a citizen (Article 9(d)); the "
        "tabulation codes every Maldivian Islam. Bare `islam` as Mauritania and the Gulf: the "
        "state school is Sunni, but no source asks or counts a school. Maldivians who are not "
        "Muslim exist (apostasy is a crime, so no one says so to a census) and are not drawn.",
    "Christian":
        "-> christianity, the bare family node: the census does not split it. Most are likely "
        "Indian (Kerala), Sri Lankan and Filipino, a Catholic and Protestant mix nothing measures.",
    "Buddhist":
        "-> buddhism, not buddhism.theravada, although most are Sri Lankan: the source names no "
        "school (ask 025's rule).",
    "Other":
        "-> other.mv, a new per-country residual. 1,289 foreign residents. With no box for no "
        "religion, it holds people with none as well as Sikhs, Jains and others; the fitted "
        "citizenship mix in sources/mv.py puts most of it on Indian and `other` citizens.",
}

MAP = {
    "Maldivian (not asked)": "islam",
    "Islam": "islam",
    "Hindu": "hinduism",
    "Christian": "christianity",
    "Buddhist": "buddhism",
    "Other": "other.mv",
}

# spec §7a-i-1: the foreign residents' answers are measured at the node they are drawn on, but
# only Malé's at the unit they are drawn on; the atolls' were counted for all atolls together.
# So countries/mv.py attaches NO `roll`, and must not attach this: rollup.py's table is per node
# for the whole country, so an identity roll would keep the atolls' 38,249 spread answers on
# screen under `inferred dots: not shown` on the strength of Malé's count. Their emptiness there
# is the honest answer (tools/check_rollup.py mv). Reviewer fafd1067-rev7, 2026-10-03.
COLUMNS = {v: v for v in MAP.values()}


def resolve(category):
    """religiondots branch for a category, or None if deliberately off the tree."""
    if category in EXCLUDED:
        return None
    return MAP.get(category)
