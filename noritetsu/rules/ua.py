"""Ukraine's rules for build_model.py (build_model.country_rules lists what it reads).

Lines and named trains (ua_sources.md, "Lines and named trains"):
- LINES: suburban trains (приміський поїзд, електричка; numbered 6000-7999 by Ukrzaliznytsia,
  "Потяг №6503: Бахмач-Пасажирський - Сновськ"), Kyiv's city electric train (Міська
  електричка, A/Б), unnamed routes between two places ("Ковель - Рівне"), and the neighbours'
  regional trains that reach the border (MÁV's Fehérgyarmat - Záhony). Metro, light rail
  (Kyiv's and Kryvyi Rih's швидкісний трамвай), trams and the funicular are never named trains.
- NAMED TRAINS: every numbered train of Ukrzaliznytsia's long-distance and regional range (1-999:
  Інтерсіті+ 7xx, night trains, the regional 8xx, local 6xx like 687 Арциз - Березине, the
  international ones), and anything OSM tags long_distance, night, international or
  intercity. Their track counts through the register lines it lies on.
- Two OSM route=train relations that are no passenger train at all are flagged as named
  trains so no total counts them: a tariff section mapped as a route ("дільниця 40-062 ...")
  and the uranium mine branch ("Залізниця до Новокостянтинівського уранового родовища").
"""
import re

SUBURBAN = re.compile(r"приміськ|електричк|дизель-поїзд|рейкобус|міська електричка", re.I)
NUMBER = re.compile(r"(?<![0-9A-Za-zА-Яа-яІіЇїЄєҐґ])(\d{1,5})(?:\s?[А-ЯІЇЄҐA-Z])?(?![0-9])")
LONG = {"long_distance", "night", "international", "intercity", "high_speed"}
NOT_A_TRAIN = re.compile(r"^дільниця\s+\d\d-\d\d\d|уранового родовища", re.I)
KIND = re.compile(r"Інтерсіті|Intercity|Нічний поїзд|Фірмовий поїзд|Міжнародний поїзд|"
                  r"\bTLK\b|\bEC\b|\bIC\b|Zakarpatia", re.I)


def looks_like_service(tags, name, name_en):
    if NOT_A_TRAIN.search(name):
        return True
    if SUBURBAN.search(name):
        return False
    if tags.get("service") in LONG or KIND.search(name):
        return True
    for text in (tags.get("ref") or "", name):
        m = NUMBER.search(text)
        if m:
            return 1 <= int(m.group(1)) <= 999
    return False
