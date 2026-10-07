"""Sweden's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

from rules.shared import EU_TRAIN, FI_TRAIN

SE_TRAIN = re.compile(r"\bNattåg|\bSnälltåget\b|\bInlandståget\b", re.I)
SE_TRAIN_OPERATORS = {"IBAB", "Inlandsbanan AB", "Snälltåget"}


def looks_like_service(tags, name, name_en):
    # OSM Sweden maps its trains by Samtrafiken's timetable table: "Tåg 41: Stockholm =>
    # Sundsvall => Umeå", "Pendeltåg 43", Pågatåg "Tåg 5", Øresundståg "Train 95". Those are
    # interval products a rider uses as lines, SJ's long-distance tables included. Named
    # trains are the night trains (SJ's "Nattåg 93: Stockholm => Narvik", tagged
    # service=night; VR's "Juna PYO 276"), Snälltåget, and Inlandsbanan's summer train
    # ("Tåg 37: Mora => Östersund", operator IBAB, one a day each way).
    if set((tags.get("service") or "").split(";")) & {"night", "car"}:
        return True
    return bool(SE_TRAIN.search(name) or FI_TRAIN.search(name) or EU_TRAIN.search(name)
                or tags.get("operator") in SE_TRAIN_OPERATORS)
