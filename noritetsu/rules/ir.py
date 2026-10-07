"""Iran's rules for build_model.py (build_model.country_rules lists what it reads).
ir_sources.md, "Lines and named trains", has the reasoning.

What OSM Iran has (data/proc/ir, 2026-10-03; 33 route=train relations):
  - RAI's suburban and local trains (operator RAI, service=commuter, numbered 7xx and 9xx):
    Tehran - Parand, Tehran - Garmsar, Tehran - Emamzadeh (Pishva), Tehran - Firuzkuh, Tehran -
    Hashtgerd, Tehran - Qom, Tabriz - Jolfa, Tabriz - Shahid Madani University, Mashhad -
    Sarakhs, Zahedan - Khash, the Ahvaz railbus. LINES: each is the regional service of its
    corridor at fixed stops, from one pair a day (Tehran - Firuzkuh) to several (Tehran -
    Parand, 12 pairs), as Türkiye's Bölgesel trains are.
  - Tehran Metro Line 5 (Tehran - Golshahr - Hashtgerd), mapped route=train service=commuter:
    a metro line, a LINE.
  - The long-distance trains of RAI's passenger companies (operator RAJA, BonRail; "شیراز –
    تهران", "همدان – مشهد", "کرمانشاه – تهران"): each is one sleeper train a day or every
    second day between two cities, sold by its company as its own train (Raja, Fadak, Bonrail,
    Noor al Reza, Rail Seir Kosar...: 19 such trains a day each way between Tehran and Mashhad
    alone, each its own product). NAMED TRAINS: their track counts through RAI's railways.
  - Pakistan Railways' Zahedan Mixed (Quetta - Zahedan, twice a month at most): a NAMED TRAIN.
"""
import re

NAMED_OPERATORS = re.compile(r"RAJA|رجا|BonRail|بن ?ریل|Fadak|فدک|Pakistan|Noor|نور|Safir|"
                             r"سفیر|Kosar|کوثر|Saba|صبا|Joopar|جوپار|Mahtab|مهتاب|Pars|پارس",
                             re.I)
NAMED_WORDS = re.compile(r"\bMixed\b|\bExpress\b|اکسپرس", re.I)
LINE_SERVICE = {"commuter", "regional", "urban", "subway"}
NAMED_SERVICE = {"long_distance", "night", "international", "high_speed"}


def looks_like_service(tags, name, name_en):
    """Is this OSM route=train relation (or route_master) a named train rather than a line?"""
    svc = set((tags.get("service") or "").split(";"))
    if svc & LINE_SERVICE:
        return False
    text = " ".join(t for t in (name, name_en, tags.get("operator"), tags.get("network"),
                                tags.get("brand")) if t)
    if svc & NAMED_SERVICE:
        return True
    who = " ".join(t for t in (tags.get("operator"), tags.get("network"), tags.get("brand"))
                   if t)
    return bool(NAMED_OPERATORS.search(who) or re.search(r"بن ?ریل|BonRail", text)
                or NAMED_WORDS.search(text))
