"""Australia's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

# OSM Australia names stop positions, and some station nodes, after their platform: "Box Hill
# 3", "Flinders Street 10", "Central, Platform 23", "Albion station, platform 1", "Armidale
# Station, Platform 1" (1,642 rail stop names in the 2026-10-02 extract, often with no
# plain-named station node beside them, so each platform became a station: 235 such records
# in a trial build). Rail stop names are read without what this matches (never a tram stop's).
PLATFORM_SUFFIX = re.compile(r"(?:\s+(?:railway\s+)?station)?"
                             r"(?:,?\s+(?:platform|plt)\s+\w+|\s+\d{1,2}[a-z]?)$", re.IGNORECASE)

AU_LINE_NETWORKS = {"Sydney Trains", "NSW TrainLink", "PTV - Metropolitan Trains",
                    "PTV - Regional Trains", "V/Line", "Translink", "Translink Nightlink",
                    "Transperth", "Adelaide Metro"}
AU_TRAIN = re.compile(r"\bXPT\b|Xplorer|^Train \d{2,3}\b|^NSW TrainLink North Coast")


def looks_like_service(tags, name, name_en):
    # The single long-distance trains are named trains: Journey Beyond's Indian Pacific,
    # Overland and Ghan, NSW TrainLink's XPTs and Xplorers (named or numbered, "Train 34"),
    # Queensland Rail Travel's (Spirit of Queensland, Tilt Train, Spirit of the Outback,
    # Westlander, Inlander, the Kuranda Scenic Railway), Transwa's Prospector and
    # Australind. The networks a rider uses as lines stay lines: Sydney Trains, NSW
    # TrainLink's intercity lines, Metro Trains Melbourne, V/Line, Translink, Transperth,
    # Adelaide Metro. Every other route=train in OSM Australia is a tourist excursion (the
    # Vintage Rail Journeys tours, Pichi Richi, Puffing Billy), a closed or heritage railway
    # mapped as a route ("Cathkin-Alexandra"), freight ("Worsley to Hamilton") or a proposal
    # ("Melbourne Metro 2"): flagged too, so it has no percentage of its own and the totals
    # leave it out (au_register.is_passenger_route decides which of them make track count).
    # A route_master with no network but service=commuter (Transperth's Thornlie-Cockburn
    # Line) is a line.
    net = tags.get("network")
    if net not in AU_LINE_NETWORKS and not (
            not net and tags.get("service") in ("commuter", "urban")):
        return True
    return (bool(AU_TRAIN.search(name))
            or tags.get("service") in ("long_distance", "national", "high_speed"))
