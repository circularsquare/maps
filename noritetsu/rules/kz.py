"""Kazakhstan's rules for build_model.py (build_model.country_rules lists what it reads). The
same rule serves Uzbekistan, Kyrgyzstan, Tajikistan and Turkmenistan (rules/uz.py, kg.py,
tj.py, tm.py import it); casia_sources.md, "Lines and named trains", has the reasoning.

- LINES: the register's tariff sections (casia_register.py) carry every train. From OSM:
  suburban trains (пригородный / электропоезд, қала маңындағы, elektropoyezd; the CIS
  suburban numbers 6000-7999, as Bishkek-1 - Tokmok 6050/6051 and Petropavl - Petukhovo
  6975), the Almaty and Tashkent metros, Astana's LRT, trams.
- NAMED TRAINS: every other OSM route=train. In all five, passenger service is long-distance
  and regional trains numbered 1-999 that run once or twice a day, every other day or weekly
  (Talgo 1/2 Almaty - Tashkent, Afrosiyob 7xx, Moscow - Dushanbe), and the unnumbered OSM
  routes are such trains drawn without a number ("Astana - Arkalyk", "Қарағанды – Семей").
  Their track counts through the tariff sections it lies on, which is the line a rider uses.
"""
import re

SUBURBAN = re.compile(r"пригородн|электропоезд|электричк|қала маң|elektropoyezd|"
                      r"shahar atrofi|banliyö|suburban|commuter", re.I)
NUMBER = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])(\d{1,5})(?:\s?[А-ЯЁA-Z])?(?![0-9])")
URBAN = {"subway", "light_rail", "tram", "monorail", "funicular"}
# A route_master with no name of its own ("13/14") is a named train when its routes are.
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    if tags.get("route") in URBAN:
        return False
    if SUBURBAN.search(name or "") or SUBURBAN.search(tags.get("service") or "") or \
            tags.get("service") == "commuter":
        return False
    for text in (tags.get("ref") or "", name or ""):
        m = NUMBER.search(text)
        if m:
            return not 6000 <= int(m.group(1)) <= 7999
    return True
