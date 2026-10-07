"""Finland's rules for build_model.py (build_model.country_rules lists what it reads)."""
from rules.shared import FI_TRAIN

# A route_master whose every route is a named train is one too: "Juna 7" holds the night
# trains PYO 273 and PYO 276 and says so nowhere else. Finland only: in jp and tw it would
# wrongly flag JR宝塚線・福知山線 and 內灣六家線, lines whose routes are all rapid or numbered
# services.
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    # OSM Finland maps VR's interval patterns as lines ("Juna 13: Helsinki => Oulu", the
    # commuter letters R, Z, H), and single trains by number: the night trains "Juna PYO
    # 273: Helsinki => Rovaniemi" and the Parikkala - Savonlinna "Taajamajuna 751".
    return bool(FI_TRAIN.search(name))
