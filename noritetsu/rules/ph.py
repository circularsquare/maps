"""The Philippines' rules for build_model.py (build_model.country_rules lists what it reads).

PNR's South Main Line is register lines (asia_register.py). OSM's PNR route=train relations
(Inter-Provincial Commuter, Bicol Commuter) are its trains over those lines: named trains,
their track counted through the register lines. LRT-1, LRT-2 and MRT-3 are OSM lines
(route=light_rail / subway, so this hook is never asked about them).
"""


def looks_like_service(tags, name, name_en):
    return True
