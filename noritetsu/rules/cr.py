"""Costa Rica's rules for build_model.py (build_model.country_rules lists what it reads).

Register lines (latam_register.py) are INCOFER's three lines. OSM's three route_masters are
the same lines (the clip gives them INCOFER's names), so each is dropped as its register
line's twin once its stations all lie on it. cr_sources.md.
"""
TWIN_ON_STATIONS = True
