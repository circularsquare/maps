"""The USA's rules for build_model.py (build_model.country_rules lists what it reads)."""
from rules.shared import US_TRAIN

# Same-named metro stations merge only within build_model.METRO_DUP_RADIUS_M (150 m, not 500):
# American metros name stations after the cross street, so Manhattan has a "23rd Street" on
# 8th, 7th, 6th and Park Avenues and on Broadway, 250-300 m apart and no interchange, and at
# 500 m the A/C/E, 1, F/M and PATH stations came out as one. And a metro stop node finds its
# station by name only within METRO_NAME_RADIUS_M (200 m): Park Avenue's 23rd Street has no
# station node in OSM, and its platforms' stop nodes took Broadway's "23rd Street" 300 m off.
METRO_DUP = True

# A register section's share run over by OSM passenger routes is measured on the section's
# own length: the metres of its ways a route runs over, against the section, not against all
# its ways. US OSM maps an Amtrak train both ways over ONE track of a double-track line, so
# the other track, which is this line's too, halves the share: BNSF's Emporia Subdivision at
# Kansas City (the Southwest Chief, 19 km) came out 0.45 and was dropped as unridden.
ROUTE_SHARE_BY_LENGTH = True

# Register lines keep the register's kind ("rail"), never the kind of the OSM track beside them
# (register_way_lines; needs build_model to read it, handoff_notes/boston_tucson.md). Every NARN
# line us_register keeps is a railroad's (PASSNGR R, rapid transit, is left out), and the track
# test called three of them metro or light rail from the line running alongside: the MBTA's Old
# Colony Line beside the Red Line from JFK/UMass to Braintree (so its stops were offered past
# the East Subdivision on the Kingston/Plymouth train, not on the Old Colony itself, and the
# Red Line's ways were the Old Colony's to own), NS's Amtrak Connection, CPKC's Canpa.
REGISTER_KIND_SURE = True


def extra_route_stops(ways, rels, stops, coords, stations, resolved, log):
    # Stations no route relation lists, as stops of the passing routes of their own network:
    # US route relations are often partial ("Port Washington Branch (as operated)" lists 5
    # stops). The register lines already take them (us_register.unlisted_stations); this gives
    # the OSM lines the same, with us_register's station overrides (NOT_ON) applied. Unused
    # until build_model.build() calls the hook (proposed 2026-10-03, us_sources.md).
    import us_register
    return us_register.osm_extra_stops(ways, rels, stops, coords, stations, resolved, log)


def route_runs(rid, runs, members, ways, coords, station_nodes, stations):
    # A route relation whose ways are in no order (NJ Transit's, Metro-North's and a few
    # others' old both-ways relations: the Morris & Essex Lines came out of 79 and 231 runs),
    # its runs rebuilt from its own track and stops (us_register.repair_route_runs); None, so
    # assemble's runs stand, for every other route. Unused until build_model.build() calls the
    # hook (proposed 2026-10-08, handoff_notes/njt_morristown.md).
    import us_register
    return us_register.repair_route_runs(rid, runs, members, ways, coords, station_nodes,
                                         stations)


def looks_like_service(tags, name, name_en):
    # Amtrak's long-distance trains, and its trains that run once a day each way, are named
    # trains: OSM tags most long-distance ones service=long_distance or night; the rest go
    # by name (US_TRAIN). The state corridors with several trains a day (Northeast
    # Regional, Acela, Keystone, Empire Service, Capitol Corridor, Pacific Surfliner, Gold
    # Runner, Cascades, Hiawatha, Lincoln Service, Wolverine, Downeaster, Piedmont, Hartford
    # Line, Valley Flyer...) stay lines, as do every commuter railroad's, and Brightline
    # (hourly: an interval product a rider uses as a line). The Alaska Railroad's few
    # trains (Denali Star, Coastal Classic...) are named trains.
    if tags.get("service") in ("long_distance", "night", "car"):
        return True
    if tags.get("network") in ("ARR", "VIA Rail"):
        return True
    return bool(US_TRAIN.search(name))
