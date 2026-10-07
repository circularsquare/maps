"""Belarus's rules for build_model.py (build_model.country_rules lists what it reads).

Lines and named trains (by_sources.md, "Lines and named trains"). Belarusian Railway sells
everything as a "line" (лінія) of some class; the project's rule decides instead: an interval
product a rider uses as a line is a line, a train with its own number running once or a few
times a day over a long route is a named train, as in Ukraine and Russia next door.
- LINES: regional economy-class trains and Minsk's city lines (four-digit numbers: 6xxx, 7xxx;
  OSM's "Прыгарадны электрацягнік: Мінск-Пасажырскі => Маладзечна", "Электрацягнік CL: Мінск
  => Рудзенск", "Цягнік №6252: Гродна => Ліда"), unnumbered "A - B" routes, Russia's
  Smolensk - Orsha/Vitebsk and Pskov-region suburban trains that reach Belarusian stations,
  the Minsk Metro, trams, the children's railway.
- NAMED TRAINS: every train with a number of 1-999, whatever its class: long-distance and
  international (001Б «Беларусь», the Moscow «Ласточка»s, the Kaliningrad trains),
  interregional business and economy class (7xx, 6xx: Minsk - Grodno 629Б, Gomel - Grodno
  609Б), and regional business class (8xx: Minsk - Orsha 861Б-868Б, one relation per train in
  OSM); and anything OSM tags long_distance, night, international (alone) or intercity. Their
  track counts through the tariff sections it lies on.
- Stale OSM relations of trains that no longer run (Praha - Moskva, Rīga - Minsk, Krichev -
  Shesterovka) are left out of the clipped extract (bymd_register.STALE_RELS); STALE here
  keeps the first two named trains should an extract come in unclipped.
"""
import re

NUMBER = re.compile(r"(?<![0-9A-Za-zА-Яа-яІіЎўЁё])(\d{1,5})(?:\s?[А-ЯІЎЁA-Z])?(?![0-9])")
LONG = {"long_distance", "night", "international", "intercity", "high_speed",
        "international;long_distance", "long_distance;international"}
# A route_master whose own tags carry no number ("Скоростной электропоезд «Ласточка»",
# "Электрацягнік RLb") is a named train when every train under it is.
SERVICE_IF_ALL_ROUTES_ARE = True
STALE = re.compile(r"^(?:Praha\s*[—-]\s*Moskva|Rīga\s*-\s*Minsk)$", re.I)


def looks_like_service(tags, name, name_en):
    if STALE.search(name or ""):
        return True
    for text in (tags.get("ref") or "", name):
        m = NUMBER.search(text)
        if m:
            return 1 <= int(m.group(1)) <= 999
    return tags.get("service") in LONG
