"""Aruba Fifth Population and Housing Census 2010, Table P-A.5 -> religiondots taxonomy.

Ten categories, 101,484 people, national. Transcribed from the census report by
`sources/terr.py`, with the census's own labels, not UNSD's. No new nodes beyond `other.aw`.

    75.3%  Roman Catholic     -> christianity.catholic
    11.7%  Other              -> other.aw
     5.5%  No religion        -> unaffiliated
     2.7%  Protestant         -> christianity.reformed.continental
     1.7%  Jehovah's witness  -> christianity.witnesses
     0.9%  Methodist          -> christianity.methodist
     0.9%  Adventist          -> christianity.adventist
     0.5%  Not reported       -> EXCLUDED
     0.4%  Anglican           -> christianity.anglican
     0.3%  Jewish             -> judaism

**UNSD'S `Pagan` IS THE CENSUS'S `No religion`**, 5,625 people in both (sources.md §11ap). The
questionnaire settles it: question 4 on the person form (report p.270) offers Roman Catholic,
Protestant (reformed), Jehovah's witness, Methodist, Adventist, Anglican, Jewish, No religion
and Other with a line to specify. There is no pagan box to have ticked.

**THE PARTITION IS ONE PERSON SHORT, IN THE CENSUS.** The ten rows sum to 101,483 against a
printed total of 101,484, and UNSD's row reproduces the same ten figures.
"""

EXCLUDED = {
    "Not reported":
        "515 people, 0.5%. `No religion` is its own row.",
}

REVIEW = {
    "Protestant":
        "-> christianity.reformed.continental. **The form's box reads `Protestant, "
        "reformed`**, which is the Protestant Church of Aruba, the Dutch Reformed "
        "congregation of the colonial period. The table shortens it to `Protestant`. Curaçao "
        "and the Caribbean Netherlands print a bare `Protestant` and are mapped literally to "
        "`christianity.protestant` (cw2023.py, bq2021.py), because their labels do not say "
        "reformed.",
    "Other":
        "-> other.aw, and it is large: **11,862 people, 11.7%**. The 2010 form named eight "
        "religions and left everything else to a write-in, which the table does not break "
        "down. Aruba's evangelical and Pentecostal churches had no box and are in here; the "
        "2000 round printed `Evangelical` separately (3,679), and 2010 did not.",
}

MAP = {
    "Roman Catholic": "christianity.catholic",
    "Protestant": "christianity.reformed.continental",
    "Jehovah's witness": "christianity.witnesses",
    "Methodist": "christianity.methodist",
    "Adventist": "christianity.adventist",
    "Anglican": "christianity.anglican",
    "Jewish": "judaism",
    "No religion": "unaffiliated",
    "Other": "other.aw",
}


def _key(cat):
    return " ".join(str(cat).split())


def resolve(cat):
    """Source category -> taxonomy node id, or None if deliberately not on the tree."""
    c = _key(cat)
    if c in EXCLUDED:
        return None
    return MAP.get(c)
