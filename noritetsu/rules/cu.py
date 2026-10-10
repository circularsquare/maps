"""Cuba's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines (latam_register.py) are the Línea Central and the three branches the four
national trains run over. The trains themselves (La Habana - Santiago, - Holguín,
- Guantánamo, - Bayamo - Manzanillo, every eight days or so) are named trains over those
lines. Every other OSM train route is clipped (latam_register.py --clip cu). cu_sources.md.
"""
TWIN_ON_STATIONS = True
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    return True
