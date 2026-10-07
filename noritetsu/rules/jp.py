"""Japan's rules for build_model.py (build_model.country_rules lists what it reads)."""


def looks_like_service(tags, name, name_en):
    # OpenStreetMap in Japan maps lines and named trains alike, as route relations, with no
    # tag that separates them: 東海道本線 (a line) and のぞみ (a train that runs over the
    # Tokaido Shinkansen) are the same kind of object. So Japan's lines totalled 49,500 km
    # against a real passenger network of about 27,500.
    #
    # The rule is deliberately crude: a Japanese line name almost always ends in 線, and a
    # train's does not. It misfires on operators whose line is named without 線 --
    # 嵯峨野観光鉄道 is a railway, not a train -- which is acceptable because it only sets a
    # flag and drops nothing. Operating patterns (京浜東北線, 中央線快速) are named with 線
    # and are lines to a rider, so they stay lines.
    #
    # 線 is the usual suffix, but an operating pattern introduced since the war is as
    # likely to be ライン: 上野東京ライン and 湘南新宿ライン are lines a rider rides, not
    # trains, and were being flagged as trains and sorted to the bottom of every list.
    if "線" in name or "ライン" in name:
        return False
    return "line" not in name_en.lower()
