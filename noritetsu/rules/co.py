"""Colombia's rules for build_model.py (build_model.country_rules lists what it reads).

The register line (latam_register.py) is the Tren Turístico de la Sabana, Bogotá - Zipaquirá
(OSM has no passenger route for it). Medellín's metro lines and tram are OSM lines. Any OSM
train route left after the clip is a named train. co_sources.md.
"""
TWIN_ON_STATIONS = True
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    return True
