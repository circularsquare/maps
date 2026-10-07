"""Croatia's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

from rules.shared import EU_TRAIN

HR_TRAIN = re.compile(r"^(?:Vlak\s+)?(?:B|IC|ICN|EC|EN)\s?\d")


def looks_like_service(tags, name, name_en):
    # OSM Croatia groups HŽPP's regional trains under their timetable line number,
    # "Vlak 23" (Vlak 2300 Kloštar => Zagreb, 2301, ...): lines. Its fast and long-distance
    # trains are mapped one train or one pair per route_master: "Vlak B 182" (brzi, Split -
    # Zagreb), "IC 58 Podravka", "ICN 52", "Vlak B 188 Dalmacija", "EuroNight Lisinski".
    return bool(HR_TRAIN.search(name) or EU_TRAIN.search(name))
