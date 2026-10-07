"""Algeria's rules for build_model.py (build_model.country_rules lists what it reads)."""

# OSM's long-distance SNTF trains mapped with only their termini and a stop or two ("Batna -
# Alger", one or two a day over 680 km, three stops) are named trains; their track counts
# through the register lines under them. The suburban and regional routes are lines.
NAMED = {"Batna - Alger"}


def looks_like_service(tags, name, name_en):
    return (name or "").strip() in NAMED
