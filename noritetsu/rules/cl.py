"""Chile's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines (cl_register.py) are the legal lines' passenger stretches; OSM route relations
are EFE's services over them. cl_sources.md has the reasoning.
"""
import re

# Named trains, not lines:
#   - the Expreso Chillán (Alameda - Chillán with four stops, a faster variant of the
#     Santiago - Chillán service, which stays a line: five trains each way a day);
#   - excursions (OSM service=tourism, or by name): El Valdiviano, the Tren del Recuerdo, the
#     Góndola Carril on the Trasandino, the Tren Arica - Poconchile (a few Saturdays a year);
#   - STOPGAP: the Tacna - Arica (Peru's FCTA): no passenger train in 2026 (the line is being
#     rebuilt); as a named train it at least counts towards nothing, since build_model has no
#     way to grey an OSM line.
NAMED = re.compile(r"Expreso|Valdiviano|Tren del Recuerdo|G[oó]ndola Carril|Poconchile|"
                   r"Tacna", re.IGNORECASE)

# An OSM line whose stations all lie on its register match is its twin and is dropped (the
# Merval's route_master against the register line of the same name).
TWIN_ON_STATIONS = True
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    if set((tags.get("service") or "").split(";")) & {"night", "long_distance", "tourism"}:
        return True
    return bool(NAMED.search(name or ""))
