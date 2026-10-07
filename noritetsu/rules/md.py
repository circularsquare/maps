"""Moldova's rules for build_model.py (build_model.country_rules lists what it reads).

OSM has six route=train relations in Moldova. Three are stale (6931 Bălți Slobozia - Ocnița,
804Ц Bender - Chișinău, Ukraine's "Чернівці - Ларга" over CFM's Lipcani track) and are left out
of the clipped extract (bymd_register.STALE_RELS). The other three are international trains,
named trains: CFR/CFM's Prietenia Chișinău - București (IRN 401/105, 402/106) and UZ's Kyiv
train 351Щ. CFM's own trains (826Г/831Г Chișinău - Ungheni, one pair a day) have no OSM
relation; the tariff sections they run over are the lines. So every route=train here is a
named train.
"""


def looks_like_service(tags, name, name_en):
    return True
