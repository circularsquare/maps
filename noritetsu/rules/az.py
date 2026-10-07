"""Azerbaijan's rules for build_model.py (build_model.country_rules lists what it reads).

- LINES: the Absheron ring (Abşeron dairəvi dəmir yolu, Baku - Pirshagi - Sumgait and Baku -
  Khirdalan - Sumgait), ADY's regional day trains (7xx: Baku - Gazakh, two pairs a day; Baku -
  Gabala; Baku - Aghdam, weekly), the Baku metro.
- NAMED TRAINS: the Baku - Balakan night train (63/64) and the Baku - Tbilisi trains (37/38).
"""
import re

NIGHT = re.compile(r"№\s*6[34]\b|Balakən|Tbilisi|თბილის", re.I)


def looks_like_service(tags, name, name_en):
    if tags.get("service") in ("long_distance", "night", "international"):
        return True
    return bool(NIGHT.search(name or ""))
