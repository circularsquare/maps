"""Sri Lanka's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines are lk_register.py's (SLR's nine lines, cut where Cyclone Ditwah's damage still
stops trains). OSM's twelve route=train relations in Sri Lanka (2026-10 extract) are those same
lines mapped as routes ("Coastal Line", "Kelani Valley Line", "Matale Line", "Up-Country"),
plus the Mihintale branch (pilgrimage specials at Poson only) and the Holcim freight line. None
is a service a rider boards as a line of its own beside the register line, so every one is a
named train: their track counts through the register lines it lies on. Every other train in
Sri Lanka (Udarata Menike, Yal Devi, the intercity and night mail trains) is a named train too.
"""


def looks_like_service(tags, name, name_en):
    return True
