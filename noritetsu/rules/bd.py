"""Bangladesh's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines are bd_register.py's (bd_lines.py). OSM's route relations in the extract
(2026-10): Dhaka Metro's MRT Line 6 (route=subway), which stays an OSM line; "Chittagong-
Dohazari", the register's own line mapped as a route; and India's Agartala Rajdhani, whose
relation reaches into the extract's border band. Every train route is a named train: Bangladesh
Railway's intercity, mail and commuter trains are each a named train over the register lines,
and their track counts through those lines.
"""


def looks_like_service(tags, name, name_en):
    if tags.get("route") in ("subway", "light_rail", "monorail", "tram") or \
            tags.get("route_master") in ("subway", "light_rail", "monorail", "tram"):
        return False
    return True
