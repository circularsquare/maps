"""Mongolia's rules for build_model.py (build_model.country_rules lists what it reads).

UBTZ's lines are register lines (asia_register.py). Every UBTZ train is a numbered train, one or two a
day each way (UB - Irkutsk 305/306, Darkhan - Sharyn Gol 604/605 in OSM): named trains, their track
counted through the register lines, as in Central Asia.
"""


def looks_like_service(tags, name, name_en):
    return True
