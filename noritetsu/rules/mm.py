"""Myanmar's rules for build_model.py (build_model.country_rules lists what it reads).

Myanma Railways' lines are register lines (asia_register.py). Its long-distance trains are one
to a few a day each way; any OSM route=train is a named train, its track
counted through the register lines (the 2026-10 extract has none). Yangon's Circular and
suburban routes are route=light_rail in OSM, so OSM lines, never asked about here.
"""


def looks_like_service(tags, name, name_en):
    return True
