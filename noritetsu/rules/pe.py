"""Peru's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines (latam_register.py) are the railways' passenger stretches; PeruRail's, Inca
Rail's and the Tren Macho's trains over them are named trains (OSM's routes are twins of the
register lines where their stations all lie on one, else named trains). The Tacna - Arica
route is a named train in any case: nothing runs on it in 2026 (its register line is greyed).
Lima's metro lines are OSM lines. pe_sources.md.
"""
TWIN_ON_STATIONS = True
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    return True
