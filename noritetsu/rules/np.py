"""Nepal's rules for build_model.py (build_model.country_rules lists what it reads).

Nepal Railway Company's line is a register line (asia_register.py). Its trains are one or two a
day each way under the line's own name; any OSM route=train is a named train, its track
counted through the register lines (the 2026-10 extract has none).
"""


def looks_like_service(tags, name, name_en):
    return True
