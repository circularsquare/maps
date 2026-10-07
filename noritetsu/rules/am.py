"""Armenia's rules for build_model.py (build_model.country_rules lists what it reads).

- LINES: the South Caucasus Railway's suburban trains (65xx, 70xx: Yerevan - Araks, Yerevan -
  Yeraskh, Yerevan - Shorzha in summer), Yerevan - Gyumri (681/682 and the express 100/101,
  several a day between them), the Yerevan metro.
- NAMED TRAINS: the international trains, Yerevan - Tbilisi (371/372, every other day) and the
  summer Yerevan - Batumi (201/202).
"""
import re

INTERNATIONAL = re.compile(r"Բաթում|Batumi|ბათუმ|Թբիլիսի|Tbilisi|თბილის", re.I)


def looks_like_service(tags, name, name_en):
    if tags.get("service") in ("long_distance", "night", "international"):
        return True
    return bool(INTERNATIONAL.search(name or "") or INTERNATIONAL.search(name_en or ""))
