"""Zimbabwe's rules for build_model.py (build_model.country_rules lists what it reads).

NRZ's lines are register lines (eafrica_register.py). OSM's routes over them, with no stops,
are the register lines again: left out. The routes for trains that do not run are dropped by
eafrica_register --clip (NOT_SERVICE).
"""

SKIP_ROUTES = {
    2276102,     # Bulawayo–Victoria Falls
    5419654,     # Bulawayo–Harare (the register's greyed trunk line)
    5419656,     # Harare–Mutare
}
