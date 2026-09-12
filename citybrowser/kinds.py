"""
Classifying a point: is it a city, or something else with a population on it?

Wikidata mixes them freely. Around New York you get, all with populations, all
as separate points:

    New York metropolitan area          19,940,274   Q1768043
    New York-Jersey City-Newark urban   19,426,449   Q5951278
    New York City                        8,804,190   Q515
    Kings County                          2,736,074   Q13414757
    Brooklyn                              2,736,074   Q408804

Five kinds come out of here, four of them hidden by default:

    city        the default view
    aggregate   metro / urban areas: a region drawn around cities
    rural       an AREA, not a place -- counties, upazilas, regencies
    district    a piece of one named city -- wards, arrondissements, boroughs
    admin       an administrative container that duplicates a city one-for-one

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

Every one of those is a real city that a keyword filter would delete. The sets
below were picked by reading a full type tally with labels, populations and
example members -- `tools/tally_types.py` reproduces it.
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

# Administrative containers that duplicate a city one-for-one. Weak: a container
# only counts as one when nothing else is claimed, since Manhattan is both a
# borough and a consolidated city-county.
#
# US counties used to live here and now live in RURAL, which is checked first --
# a county is an area in its own right, not a duplicate of its seat, and there
# are 2,350 of them rather than the handful this set was written for.
ADMIN = {
    "Q47168",      # county seat of the United States
}

# ---------------------------------------------------------------------------
# RURAL -- an AREA, not a place.
#
# The distinction that actually holds up is NOT "rural": it is whether the row
# is a settlement or the administrative area a settlement sits in. Los Angeles
# County is 10.0M on this map, Cook County 5.3M, Harris County 4.7M -- three of
# the biggest bubbles in North America, none of them a place anyone is from.
#
# The 10,000 population floor already does the "rural" half of the job. Every
# French commune that survives it has ten thousand people in it, which is why
# communes are NOT in here: Nice, Marseille, Nantes and Toulouse are all typed
# `commune of France` and 969 of the 1,099 carry no other type, so flagging the
# class would delete French cities rather than French countryside. The same
# argument keeps out Brazilian municipalities (plain city names: Abaetetuba,
# Criciúma) and Philippine municipalities (Bay, Alaminos, Tanza) -- in both
# countries that class IS the town layer, and no other layer is fetched.
#
# What is in here is the layer ABOVE the town: a county, an upazila, a regency,
# a Mexican municipio (908 of 939 are literally named "X Municipality"), a
# South African local municipality (236 of 237 likewise), a Polish gmina.
# ---------------------------------------------------------------------------

# One Wikidata class per state, plus Louisiana parishes and Alaska boroughs.
# Generated by tools/tally_types.py; 2,364 points, of which 2,352 carry no
# other type at all, so this is about as clean a cut as the data offers.
US_COUNTY = {
    "Q11774097",   # county of Texas  (162)
    "Q13410428",   # county of Georgia  (123)
    "Q13410447",   # county of Kentucky  (99)
    "Q13414758",   # county of North Carolina  (96)
    "Q13414759",   # county of Ohio  (88)
    "Q11812346",   # county of Illinois  (87)
    "Q13415366",   # county of Tennessee  (87)
    "Q13410438",   # county of Indiana  (86)
    "Q13410520",   # county of Missouri  (85)
    "Q13415368",   # county of Virginia  (84)
    "Q13410496",   # county of Michigan  (75)
    "Q11774149",   # county of Iowa  (74)
    "Q13410508",   # county of Mississippi  (67)
    "Q12178928",   # county of Wisconsin  (67)
    "Q12262532",   # county of Minnesota  (66)
    "Q13410400",   # county of Alabama  (65)
    "Q13410422",   # county of Florida  (65)
    "Q13414763",   # county of Pennsylvania  (64)
    "Q11774062",   # county of Arkansas  (62)
    "Q13414757",   # county of New York  (61)
    "Q13414760",   # county of Oklahoma  (57)
    "Q13410524",   # parish of Louisiana  (57)
    "Q13414765",   # county of South Carolina  (44)
    "Q13415370",   # county of West Virginia  (43)
    "Q13212489",   # county of California  (41)
    "Q13410444",   # county of Kansas  (39)
    "Q13410403",   # county of Colorado  (37)
    "Q13415369",   # county of Washington  (35)
    "Q13410433",   # county of Idaho  (29)
    "Q13414761",   # county of Oregon  (29)
    "Q13414361",   # county of Nebraska  (26)
    "Q13414755",   # county of New Mexico  (25)
    "Q13410464",   # county of Maryland  (23)
    "Q13414358",   # county of Montana  (21)
    "Q13414754",   # county of New Jersey  (21)
    "Q13415365",   # county of Utah  (20)
    "Q12037308",   # county of South Dakota  (18)
    "Q13410454",   # county of Maine  (16)
    "Q13415371",   # county of Wyoming  (15)
    "Q13217186",   # county of Arizona  (14)
    "Q13410485",   # county of Massachusetts  (14)
    "Q12178655",   # county of North Dakota  (14)
    "Q131427665",  # charter county of California  (14)
    "Q13415367",   # county of Vermont  (12)
    "Q13414753",   # county of New Hampshire  (10)
    "Q13414369",   # county of Nevada  (8)
    "Q13414354",   # county of Connecticut  (8)
    "Q1188782",    # county-equivalent  (9)
    "Q13410522",   # borough of Alaska  (7)
    "Q110266209",  # home rule county of Pennsylvania  (7)
    "Q13414764",   # county of Rhode Island  (5)
    "Q13410431",   # county of Hawaii  (4)
    "Q13410411",   # county of Delaware  (3)
    "Q139553869",  # charter county of Florida  (1)
}

# Civil townships: the sub-county survey layer, whose incorporated villages are
# separate rows. New Jersey is deliberately ABSENT -- the state has no
# unincorporated territory, so a NJ township is an ordinary municipality and
# its rows are places people are from (Weehawken, Lyndhurst, Woodbridge).
US_TOWNSHIP = {
    "Q9035798",    # township of Pennsylvania  (196)
    "Q17198545",   # township of Illinois  (169)
    "Q17201685",   # township of Indiana  (129)
    "Q17198620",   # township of Ohio  (119)
    "Q6270791",    # township of Missouri  (116)
    "Q5086782",    # charter township of Michigan  (75)
    "Q28111",      # township in the United States  (41)
    "Q17205621",   # township of Iowa  (31)
    "Q17205774",   # township of Michigan  (28)
    "Q1394476",    # civil township  (6)
    "Q17205735",   # township of Kansas  (6)
    "Q7132722",    # paper township (Ohio)  (4)
    "Q6643832",    # township of North Carolina  (4)
    "Q17351862",   # township of Utah  (2)
    "Q24027556",   # township of Arkansas  (1)
    "Q119174484",  # township of Jefferson County  (1)
    "Q15063142",   # urban township of Minnesota  (1)
    "Q19610511",   # township (generic)  (9)
}

RURAL = US_COUNTY | US_TOWNSHIP | {
    # New York State towns: the township layer, with its villages and hamlets
    # listed separately. 130 of the 152 also carry "town in the United States",
    # which is why that class is NEUTRAL rather than a settlement marker.
    "Q106071004",  # town of New York  (152)

    # Mexico. The localities (Q20202352, 498 rows) are the settlements; these
    # are the municipios around them, and 908 of 939 say so in their own name.
    "Q1952852",    # municipality of Mexico  (939)

    # Morocco. `commune of Morocco` on its own is NEUTRAL -- 359 of these 576
    # carry both, and the rural marker is the one that decides.
    "Q17318027",   # rural commune of Morocco  (576)

    # South Asia
    "Q620471",     # upazila of Bangladesh  (495)
    "Q18670606",   # tehsil of Pakistan  (11)
    "Q29467088",   # rural municipality of Nepal  (10)

    # South-east Asia
    "Q3191695",    # regency of Indonesia  (358)
    "Q3700011",    # kecamatan  (25)
    "Q15141625",   # subdistrict municipality (Thailand)  (209)
    "Q15140073",   # subdistrict administrative organization (Thailand)  (95)
    "Q2582669",    # rural district of Vietnam  (5)
    "Q7830262",    # township of Myanmar  (5)

    # Russia and Ukraine: municipal formations, not settlements. The rows named
    # "<X>sky District" are these.
    "Q13626398",   # urban okrug in Russia  (343)
    "Q634099",     # rural settlement in Russia  (258)
    "Q3350075",    # municipal okrug  (48)
    "Q2198484",    # municipal district  (45)

    # Africa
    "Q1500352",    # local municipality of South Africa  (237)
    "Q1589568",    # district municipality (South Africa)  (44)
    "Q3327871",    # district municipality  (8)
    "Q1639634",    # local government area of Nigeria  (54)
    "Q690840",     # district of Ethiopia (woreda)  (56)

    # Europe
    "Q1147395",    # district of Turkey (ilçe)  (184)
    "Q3491915",    # urban-rural municipality of Poland (gmina)  (53)
    "Q3504085",    # rural municipality of Poland  (28)
    "Q17301072",   # district municipality of Lithuania  (43)
    "Q871419",     # district of Austria (Bezirk)  (14)
    "Q180673",     # ceremonial county of England  (3)
    "Q179872",     # county of Ireland  (1)
    "Q192299",     # county of Norway  (1)

    # Americas and Oceania
    "Q13997861",   # partido of Buenos Aires Province  (124)
    "Q18810091",   # census division of Canada  (28)
    "Q85796467",   # regional district in British Columbia  (25)
    "Q55774719",   # township municipality in Ontario  (13)
    "Q14763041",   # county of Ontario  (12)
    "Q11774771",   # county of Nova Scotia  (6)
    "Q3327874",    # rural municipality of Canada  (4)
    "Q603715",     # county of New Brunswick  (1)
    "Q33127844",   # Local Government Area (Australia)  (1)

    # East Asia. County-level and prefecture-level CITIES stay cities -- see
    # EXTRA_SETTLEMENT. These are the county layer proper.
    "Q1289426",    # county of China  (4)
    "Q1336099",    # autonomous county of China  (1)
    "Q18534049",   # county of North Korea  (1)
    "Q2367508",    # township of Taiwan (rural 鄉)  (95)
    "Q17194218",   # mountain indigenous township (Taiwan)  (3)
}

# Types that do not, on their own, say "this is a settlement" -- so they must
# not veto a RURAL classification. Without this the rule fires on far less than
# it should: 130 of the 152 New York towns carry `town in the United States`,
# 359 of the 576 Moroccan rural communes also carry the plain `commune of
# Morocco`, and 58 of the 358 Indonesian regencies carry the generic
# second-level-subdivision class.
#
# A class only belongs here when carrying it ALONE would leave a row correctly
# classified as a city -- `town in the United States` alone is a New England
# town, which is a real place, and it stays one.
NEUTRAL = {
    "Q56061",      # administrative territorial entity
    "Q12357832",   # second-level administrative division
    "Q12357920",   # third-level administrative division
    "Q14757767",   # fourth-level administrative division
    "Q12479774",   # second-level administrative country subdivision (Indonesia)
    "Q15127012",   # town in the United States
    "Q2989400",    # commune of Morocco
    "Q5888666",    # home rule municipality of Pennsylvania
    "Q110266206",  # optional plan municipality of Pennsylvania
    "Q7635776",    # Sukhaphiban (former Thai sanitary district)
    "Q117286571",  # planning region of Connecticut -- replaced its counties
    "Q6501447",    # local government
    "Q19953632",   # former administrative territorial entity
    "Q131463097",  # former municipality of the United States
    "Q1550680",    # unorganized territory
    "Q253836",     # judicial district
    "Q192611",     # electoral unit
    "Q3301455",    # electoral precinct
    "Q17362920",   # Wikimedia duplicated page
}

# ---------------------------------------------------------------------------
# DISTRICT -- a piece of one named city, which already has its own bubble.
#
# Decisive rather than weak, because none of these classes is carried by a
# whole city: `ward of Japan` is only ever a ku, `borough of New York City` is
# only ever one of the five. Manhattan and Queens carry `consolidated
# city-county` alongside, and the decisive rule is what stops that from
# outvoting the borough.
#
# China is the case worth stating: a city can be a whole prefecture there, and
# county-level cities sit INSIDE prefecture-level ones, so "is a subdivision"
# cannot mean "is not a city". Both city classes are absent from this set, and
# exactly one row in the data carries a district class alongside one of them --
# Beibei District, 835k, a district of Chongqing that used to be a county-level
# city. It is a district today, so the decisive rule gets it right.
#
# NOT in here, deliberately:
#   Q1070990  county-level city of China      a city inside a city, but a CITY
#   Q748149   prefecture-level city of China  same
#   Q2911266  borough of New Jersey           an independent municipality
#   Q777120   borough of Pennsylvania         same
#   Q2327515  city district in Baden-Württemberg   a Stadtkreis IS the city
#   Q20202352 locality of Mexico              the settlement, not the division
#   Q127499753 district of El Salvador        the 2023 reform renamed towns
#
# Informal neighbourhoods are a THIRD question and are still classified as
# cities: Q123705 neighborhood (722), Q188509 suburb (925), Q2755753 area of
# London (265), Q17051044 mahalle (1,487), Q253019 Ortsteil (155). The
# per-city classes below are official subdivisions and are safe to fold in;
# those five are not, and "keep neighbourhoods as map points?" is open in
# TODO.md.
# ---------------------------------------------------------------------------
DISTRICT = {
    # Japan
    "Q137773",     # ward of Japan  (178)
    "Q5327704",    # special ward of Japan -- Tokyo's 23  (23)
    "Q65948724",   # neighborhood in Japan  (3)

    # France
    "Q702842",     # municipal arrondissement of France  (45)
    "Q87410915",   # administrative quarter of Marseille  (17)

    # Russia, Ukraine, Central Asia
    "Q15195406",   # city district in Russia  (292)
    "Q4389092",    # district of Moscow  (113)
    "Q3565075",    # raion of city in Ukraine  (51)
    "Q42619282",   # municipal okrug of Saint Petersburg  (35)
    "Q129675946",  # district of a city of republican significance (KZ)  (15)
    "Q129791543",  # district of a city of regional significance (KZ)  (2)
    "Q14242187",   # city district of Nizhny Novgorod  (3)
    "Q17309849",   # city district of Turkmenistan  (3)
    "Q15630934",   # district of Tashkent  (1)

    # Generic, and East/South-east Asia
    "Q4286337",    # city district  (158)
    "Q705296",     # district of Taiwan  (152)
    "Q15634531",   # district of Bangkok  (50)
    "Q15634883",   # district of Manila  (13)
    "Q5283507",    # district of Davao City  (11)
    "Q1065118",    # district of China  (6)
    "Q15634846",   # district of Seoul  (1)

    # Germany, Austria, Switzerland
    "Q35034452",   # locality of Berlin  (77)
    "Q821435",     # borough of Berlin  (15)
    "Q15830667",   # quarter of Hamburg  (42)
    "Q278976",     # borough of Hamburg  (7)
    "Q253270",     # borough of Munich  (25)
    "Q97312698",   # city district in Hannover  (25)
    "Q15727673",   # city district of Hanover  (13)
    "Q1852178",    # locality of Düsseldorf  (24)
    "Q79416466",   # neighborhood of Frankfurt  (17)
    "Q15632133",   # city district of Cologne  (3)
    "Q17278559",   # quarter of Bremen  (3)
    "Q17278423",   # district of Bremen  (1)
    "Q13415859",   # district of Wuppertal  (2)
    "Q79337953",   # district of Kiel  (2)
    "Q110710943",  # locality of Leipzig  (4)
    "Q79423884",   # quarter in Leipzig  (2)
    "Q110712535",  # district of Leipzig  (1)
    "Q261023",     # district of Vienna  (23)
    "Q1852119",    # district of Graz  (2)
    "Q124400427",  # district of Klagenfurt  (4)
    "Q28539166",   # quarter of Basel  (7)

    # Spain and Italy
    "Q10267336",   # neighborhood of Madrid  (112)
    "Q3032114",    # district of Madrid  (21)
    "Q75135432",   # administrative quarter in Barcelona  (43)
    "Q790344",     # district of Barcelona  (10)
    "Q8561193",    # district of Valencia  (15)
    "Q6350957",    # district of Cartagena  (5)
    "Q3927261",    # quarter of Naples  (26)
    "Q3927239",    # quarter of Bari  (9)
    "Q1584957",    # district of Palermo  (8)
    "Q3927255",    # quarter of Palermo  (4)
    "Q3927244",    # borough of Bologna  (6)
    "Q16751551",   # borough of Brescia  (1)
    "Q100701580",  # quarter of Cagliari  (1)

    # Rest of Europe
    "Q15715406",   # neighbourhood of Helsinki  (25)
    "Q28480345",   # district of Vantaa  (2)
    "Q5283513",    # district of Espoo  (1)
    "Q851110",     # district of Budapest  (23)
    "Q74728036",   # administrative district of Prague  (22)
    "Q856976",     # city district of Stockholm  (12)
    "Q86681780",   # district of The Hague  (8)
    "Q15079751",   # borough of Amsterdam  (2)
    "Q2597772",    # district of Antwerp  (5)
    "Q3394564",    # quarter of Luxembourg City  (6)
    "Q1770467",    # borough of Oslo  (1)
    "Q17233249",   # city district of Szczecin  (1)
    "Q134517639",  # district of Osijek  (4)
    "Q6988101",    # neighborhood of Kaunas  (2)
    "Q6988120",    # neighborhood of Vilnius  (1)
    "Q29463880",   # quarter of Limassol  (2)
    "Q47326153",   # quarter of Larnaca  (2)

    # Americas
    "Q408804",     # borough of New York City  (5)
    "Q61297932",   # neighborhood of Manhattan  (12)
    "Q61298024",   # neighborhood in Queens  (6)
    "Q61298320",   # neighborhood in Brooklyn  (4)
    "Q61298104",   # neighborhood in The Bronx  (2)
    "Q1969642",    # neighborhood in San Francisco  (13)
    "Q12063697",   # neighborhood of Washington, D.C.  (6)
    "Q110798863",  # neighborhood of Pittsburgh  (4)
    "Q3413329",    # neighborhood in Boston  (1)
    "Q578521",     # borough of Montreal  (3)
    "Q136997911",  # neighbourhood of Quebec City  (5)
    "Q18559008",   # district of São Paulo  (95)
    "Q19886692",   # neighbourhood of São Paulo  (2)
    "Q19658107",   # neighborhood of Brazil  (5)
    "Q20683285",   # neighbourhood in Rio de Janeiro  (4)
    "Q124072717",  # neighborhood in Fortaleza  (2)
    "Q851517",     # neighborhood of Buenos Aires  (8)

    # Middle East
    "Q28372019",   # neighborhood of Yemen  (18)
    "Q12054099",   # neighborhood of Jerusalem  (3)
}


def classify(types):
    """-> 'aggregate' | 'district' | 'rural' | 'admin' | 'city'.

    Two of the five are DECISIVE -- one matching type settles it:

    AGGREGATE, because "New York metropolitan area" carries both Q174844
    'megacity' and Q1768043 'metropolitan statistical area', and requiring
    every type to be an aggregate left it classified as a city, which is the
    exact duplicate the toggle exists to hide. None of the aggregate markers is
    carried by a real city -- 'metropolis', 'megacity' and 'metropolitan city of
    South Korea' are deliberately NOT in the set.

    DISTRICT, for the same reason: Manhattan is 'borough of New York City' and
    'consolidated city-county' at once, and the borough is the true statement
    about what the bubble duplicates.

    The other two are WEAK -- they only apply when nothing in the row argues
    otherwise. RURAL allows NEUTRAL company (a New York town also being a "town
    in the United States" does not make it a settlement); ADMIN allows none,
    since a county seat that also calls itself a city is a city.
    """
    ts = set(types or [])
    if not ts:
        return "city"
    if ts & AGGREGATE:
        return "aggregate"
    if ts & DISTRICT:
        return "district"
    if ts & RURAL and not (ts - RURAL - NEUTRAL):
        return "rural"
    if ts & ADMIN and not (ts - ADMIN):
        return "admin"
    return "city"
