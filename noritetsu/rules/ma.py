"""Morocco's rules for build_model.py (build_model.country_rules lists what it reads)."""

# OSM's ONCF routes that run over one register line and stop only at its stations (Safi -
# Ben Guerir, Nador - Taourirt, Tanger - Tanger Med) are that line's twin, however their
# length compares (the routes start and end at stations the register line also ends at).
TWIN_ON_STATIONS = True
