"""Taiwan's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

# A free-standing train number: 821 in "台灣高鐵 821". Not the 1 of "S1" or "S11", whose
# numbers are the line's name. (The same pattern as build_model.TRAIN_NUMBER.)
TRAIN_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d{1,4}(?![A-Za-z0-9])")


def looks_like_service(tags, name, name_en):
    # OSM in Taiwan maps every THSR and some TRA trains by train number, one relation
    # each ("台灣高鐵 821 南港->左營"). A line's name carries no train number.
    return bool(TRAIN_NUMBER.search(name))
