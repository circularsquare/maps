"""New Zealand's rules for build_model.py (build_model.country_rules lists what it reads).
nz_sources.md has the reasoning."""
import re

# OSM New Zealand names many stop positions after their platform or with "Station": "Maungawhau
# 3", "Te Waihorotiu 1", "Hamilton Frankton 2", "Petone Station", "Ngauranga - Platform 1"
# (2026-10-03 extract: Auckland's stations are mapped only as stop positions, so each platform
# would be a station of its own). Rail stop names are read without what this matches. The
# Cable Car's "Kelburn - Cable Car Station" keeps its name.
PLATFORM_SUFFIX = re.compile(
    r"(?:\s+-)?(?:\s+(?:railway\s+)?station)?(?:,?\s+-?\s*(?:platform|plt)\s+\w+|\s+\d{1,2}[a-z]?)$"
    r"|(?<!Car)\s+(?:railway\s+)?station$", re.IGNORECASE)

# The networks a rider uses as lines: Auckland Transport's (Auckland One Rail), Metlink's
# (Transdev Wellington) and Te Huia (Waikato Regional Council, network "AT;BUSIT").
LINE_NETWORKS = {"AT", "Metlink", "BUSIT"}


def looks_like_service(tags, name, name_en):
    # Named trains: KiwiRail Scenic's single long-distance trains (the Northern Explorer, the
    # Coastal Pacific, the TranzAlpine) and the Capital Connection (one return trip each
    # weekday, Palmerston North - Wellington); Dunedin Railways' excursions (The Inlander /
    # Taieri Gorge, the Seasider, the Victorian) and the heritage railways OSM maps as
    # route=train (Glenbrook, Goldfields, Weka Pass, Bay of Islands, Gisborne City). Their track
    # counts through the KiwiRail register lines it lies on. Te Huia (two to four trips a day,
    # Hamilton - Auckland) and the AT and Metlink lines are lines.
    nets = set(re.split(r"\s*;\s*", tags.get("network") or ""))
    return not (nets & LINE_NETWORKS)


# Not done: OSM's Kapiti Line relation (1171504) is PTv1 (forward/backward ways, stops under
# "forward:stop" / "backward:stop" or no role), so build_model reads only two of its stops
# (Wellington, Tawa). Giving it the others as `extra_route_stops` (rules/th.py's way) was tried
# on 2026-10-03 and dropped: the relation's ways assemble into several runs out of order, so
# the stops came out in the wrong order (Paremata - Paraparaumu 26 km, Wellington - Paremata
# 22 km; 108 km for a 55 km line) and the Johnsonville and Capital Connection relations broke
# the same way. The NIMT register line carries the Kapiti Line's track either way.
