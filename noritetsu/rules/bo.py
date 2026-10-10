"""Bolivia's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines (latam_register.py) are the four railway stretches FCA's and FO's weekly
trains run (Oruro - Villazón, Viacha - Charaña, Santa Cruz - Puerto Quijarro, Santa Cruz -
Yacuiba); the trains (Expreso del Sur, buscarril, Expreso Oriental, ferrobús) are named
trains over them. Cochabamba's Mi Tren lines are OSM lines (light rail, never named
trains). bo_sources.md.
"""
TWIN_ON_STATIONS = True
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    return True
