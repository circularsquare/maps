"""Bosnia and Herzegovina's rules for build_model.py (build_model.country_rules lists what it
reads). ŽFBH's and ŽRS's OSM routes are lines ("Banja Luka => Doboj", refs 6401;6403...); the
Croatian trains in the extract's edge follow Croatia's rule."""
from rules.hr import looks_like_service  # noqa: F401
