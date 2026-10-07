"""Hungary's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

# "Train EC Hornád: Budapest => Košice" is mapped the Slovak way, with "Train " in front.
HU_TRAIN = re.compile(r"^(?:Train\s+)?(?:IC|EC|EN|ICE|RJX?)(?:\s|\d|$)"
                      r"|\bEuro(?:City|Night)\b|\bRailjet\b")


def looks_like_service(tags, name, name_en):
    # OSM Hungary maps each international and InterCity train as its own relation (IC 929
    # Savaria, EC 173, EN 462, ICE 90, "Hungaria EuroCity"). S, G, Z, Sz, R/REX and the
    # InterRégió patterns (IR87 AGRIA, KISKUN IR, IR CÍVIS) run every hour or two: lines.
    return bool(HU_TRAIN.search(name))
