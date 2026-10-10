"""Pakistan's rules for build_model.py (build_model.country_rules lists what it reads).

OSM's nine Pakistan Railways route=train relations are single named trains (Rehman Baba
Express 48DN, Bolan Mail 4DN, Sukkur Express 146DN, Khushhal Khan Khattak Express, Subak
Kharam, Chaman Mixed, Zahedan Mixed): named trains, so their track counts through the register
lines (pk_register.py) under them. Lahore's Orange Line (route=subway) stays a line.
"""


def looks_like_service(tags, name, name_en):
    if tags.get("route") in ("subway", "light_rail", "tram", "monorail"):
        return False
    return tags.get("route") == "train" or tags.get("route_master") == "train"
