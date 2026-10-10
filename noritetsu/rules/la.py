"""Laos's rules for build_model.py (build_model.country_rules lists what it reads).

The Laos-China Railway and the Thanaleng line are register lines (asia_register.py). OSM's
route=train relations in Laos are trains over them (the LCR's C82/C84, Kunming - Vientiane,
SRT's 147/148 Udon Thani - Khamsavath) or the LCR line again as a route: all named trains,
their track counted through the register lines.
"""


def looks_like_service(tags, name, name_en):
    return True
