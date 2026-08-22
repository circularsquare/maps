"""
Classifying a point: is it a city, or an aggregate of cities?

Wikidata mixes them freely. Around New York you get, all with populations, all
as separate points:

    New York metropolitan area          19,940,274   Q1768043
    New York-Jersey City-Newark urban   19,426,449   Q5951278
    New York City                        8,804,190   Q515
    Kings County                          2,736,074   Q13414757
    Brooklyn                              2,736,074   Q408804

Three similar bubbles for "New York" is silly, so aggregates are hidden by
default and toggleable in settings.

THIS IS A CURATED LIST, NOT A KEYWORD MATCH, and deliberately so. Matching on
"urban area" / "metropolitan" / "greater" in the label looks tempting and is
wrong:

    Q12813115  "urban area in Sweden"    tatort -- how Swedish towns ARE counted
    Q15092344  "urban area in Norway"    same
    Q448801    "Greater district town"   a German town class, not an aggregate
    Q200250    "metropolis"              carried by New York City itself
    Q2716259   "metropolitan municipality in Turkey"   Istanbul's actual govt
    Q1530824   "metropolitan municipality in SA"       Johannesburg, Cape Town
    Q482821    "metropolitan city of South Korea"      Busan, Incheon

Every one of those is a real city that a keyword filter would delete.
"""

# Types that ARE cities but sit outside the P279* closure of human settlement
# and municipality. Found by listing every dropped type whose English label
# contains "city" -- a check worth repeating whenever the closure changes.
#
# The Chinese one is why: 396 county-level cities were fetched and 366 silently
# deleted, which is exactly the "sub-prefecture cities missing from most
# provinces" that prompted this. China ends up with 286/293 prefecture-level
# cities but only 30/394 county-level without it.
EXTRA_SETTLEMENT = {
    "Q1070990",    # county-level city of China      (366)
    "Q1044880",    # subprefecture-level city         (22)
    "Q27587207",   # city municipality                (48)
    "Q21503295",   # district with city status        (46)
    "Q102104752",  # city of Bosnia and Herzegovina    (9)
    "Q17166802",   # city municipality of Serbia       (7)
    "Q17166803",   # city municipality of Lithuania    (6)
    "Q4272761",    # administrative city of Indonesia  (5)
    "Q2734310",    # territorial demarcation of Mexico City -- CDMX boroughs
    "Q34985575",   # city district
}

# Statistical or planning aggregates: a region drawn around cities, not a place
# anyone is from. Hidden by default.
AGGREGATE = {
    "Q5951278",    # urban area of the United States
    "Q1768043",    # metropolitan statistical area
    "Q1907114",    # metropolitan area
    "Q3175539",    # functional urban area
    "Q702492",     # urban area
    "Q159313",     # urban agglomeration
    "Q245260",     # conurbation
    "Q4056979",    # agglomeration of Russia
    "Q2342385",    # urban area (INSEE statistical area, France)
    "Q1479822",    # urban unit (INSEE)
    "Q3931970",    # metropolitan region in Brazil
    "Q71144301",   # metropolitan region (Brazil)
    "Q126096593",  # polycentric metropolitan area
    "Q1132300",    # metropolitan region in Germany
    "Q11962746",   # metropolitan region of Norway
    "Q3201456",    # metropolitan area of Mexico
    "Q106464977",  # built-up area in the United Kingdom
    "Q124637560",  # built-up area subdivision
    "Q923953",     # transborder agglomeration
    "Q2826814",    # urban agglomeration in Quebec
    "Q11480694",   # urbanization promotion area
    "Q15110",      # metropolitan city of Italy (replaced provinces)
    "Q122677",     # metropolitan area (generic, if present)
}

# Administrative containers that duplicate a city one-for-one (Kings County is
# Brooklyn). Also hidden by default -- same "three bubbles" complaint.
ADMIN = {
    "Q13414757",   # county of the United States (New York County = Manhattan)
    "Q47168",      # county seat of the United States
}


def classify(types):
    """-> 'aggregate' | 'admin' | 'city'.

    An AGGREGATE marker is DECISIVE. "New York metropolitan area" carries both
    Q174844 'megacity' and Q1768043 'metropolitan statistical area'; requiring
    every type to be an aggregate left it classified as a city, which is the
    exact duplicate the toggle exists to hide. None of the aggregate markers is
    carried by a real city -- 'metropolis', 'megacity' and 'metropolitan city of
    South Korea' are deliberately NOT in the set.

    ADMIN is weaker: a county only counts as one when nothing else is claimed,
    since Manhattan is both a borough and a consolidated city-county.
    """
    ts = set(types or [])
    if not ts:
        return "city"
    if ts & AGGREGATE:
        return "aggregate"
    if ts & ADMIN and not (ts - ADMIN):
        return "admin"
    return "city"
