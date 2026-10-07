"""Canada's rules for build_model.py (build_model.country_rules lists what it reads)."""
from rules.shared import US_TRAIN

CA_TRAIN_NETWORKS = {"Ontario Northland", "TSH", "Keewatin Railway Company", "Rocky Mountaineer"}

# A register section's share run over by OSM passenger routes is measured on its own length,
# as in the US (rules/us.py says why): Canada's register is NARN too, read by us_register.
ROUTE_SHARE_BY_LENGTH = True


def extra_route_stops(ways, rels, stops, coords, stations, resolved, log):
    # As the US (rules/us.py): unlisted stations as stops of their own network's passing
    # routes, with Canada's station overrides (ca_register.CA_NOT_ON). adopt() points
    # us_register at Canada's files and rules for this process, as the register build does.
    import ca_register
    import us_register
    ca_register.adopt()
    return us_register.osm_extra_stops(ways, rels, stops, coords, stations, resolved, log)


def looks_like_service(tags, name, name_en):
    # VIA's long-distance and less-than-daily trains (the Canadian, the Ocean, Jasper - Prince
    # Rupert, Winnipeg - Churchill, Montréal - Jonquière/Senneterre, Sudbury - White River) and
    # the Maple Leaf are named trains; VIA Rail Corridor (Québec - Windsor) is a line, as are
    # GO, exo, UP Express, West Coast Express and the Tsal'alh Seton Train (daily). The remote
    # lines' single trains (Polar Bear Express, Tshiuetin, Keewatin Railway; 2-5 a week) and
    # the Rocky Mountaineer (seasonal cruise train) are named trains. Amtrak as in the US.
    if tags.get("service") in ("long_distance", "night", "car"):
        return True
    if "VIA Rail" in tags.get("network", ""):
        return "Corridor" not in name
    if tags.get("network") in CA_TRAIN_NETWORKS:
        return True
    return bool(US_TRAIN.search(name))
