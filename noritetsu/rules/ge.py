"""Georgia's rules for build_model.py (build_model.country_rules lists what it reads).

Lines and named trains (caucasus_sources.md, "Lines and named trains"):
- LINES: Georgian Railway's regional trains (6xx, 63xx, 64xx: Tbilisi - Borjomi, Kutaisi -
  Batumi, Kutaisi - Sachkhere, Batumi - Ozurgeti, Kutaisi - Rioni, Tbilisi - Rustavi -
  Gardabani, Zestafoni - Khashuri) and the Tbilisi - Batumi double-deckers 801-808 (three
  pairs a day: the corridor's interval service); the metro and the funicular.
- NAMED TRAINS: the once-a-day long-distance trains (853/854 Ozurgeti every other day,
  869/870 Zugdidi, 873/874 Poti), and the international ones (Yerevan 201/202 and 371/372,
  Baku 37/38). Their track counts through the tariff sections.
"""
import re

NUMBER = re.compile(r"#\s*(\d{1,5})|№\s*(\d{1,5})|^(\d{1,5})\b")
INTERNATIONAL = re.compile(r"ბაქო|Bakı|Baku|ერევან|Երևան|Yerevan|Ереван|Сухум", re.I)


def looks_like_service(tags, name, name_en):
    if tags.get("service") in ("long_distance", "night", "international") or \
            INTERNATIONAL.search(name or ""):
        return True
    for text in (tags.get("ref") or "", name or ""):
        m = NUMBER.search(text)
        if m:
            n = int(next(g for g in m.groups() if g))
            # 813/814 is OSM's route_master of the Batumi - Ozurgeti regional 613/614
            if 801 <= n <= 808 or n in (813, 814):
                return False
            return n < 600 or 800 <= n <= 999
    return False
