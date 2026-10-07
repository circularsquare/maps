"""Egypt's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

# OSM's three ENR train relations are corridors with a handful of stops ("Alexandria-Aswan
# line" 1,106 km with 13, "Ismailia-Cairo line", "Mansoura-Cairo line"), not a service a rider
# boards as a line: named trains, so their track counts through the register lines under them.
CORRIDOR = re.compile(r"\bline$", re.I)


def looks_like_service(tags, name, name_en):
    if tags.get("ref") == "LRT":
        return False
    return bool(CORRIDOR.search(name or "") or CORRIDOR.search(name_en or ""))
