"""Russia's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

RU_TRAIN = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])\d{3}\s?[А-ЯЁA-Z](?![0-9A-Za-zА-Яа-яЁё])")
# A pair numbered without its letter ("Скорый поезд 001/002 «Красная стрела»", ref 001/002),
# and long-distance trains named by their kind and no number ("Скоростной поезд «Аврора»").
RU_TRAIN_PAIR = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])\d{3}\s?[А-ЯЁA-Z]?/\d{3}(?!\d)")
RU_TRAIN_KIND = re.compile(r"(?:Скорый|Скоростной|Высокоскоростной|Пассажирский|Фирменный)"
                           r"\s+поезд\b")


# Stale routes: Ukrzaliznytsia's pre-war route relation in occupied Donetsk oblast, unnamed,
# which came back as a 3.6 km line once ru_register --clip put the disused frontline track
# there back (2026-10-04). The trains there are the register's (annex_trains.json).
SKIP_ROUTES = {3478672}


def looks_like_service(tags, name, name_en):
    # OSM Russia maps every long-distance train one relation per train, by its number of
    # three digits and a letter: "Скорый поезд 124Ы: Красноярск → Абакан", "Высокоскоростной
    # поезд 752А «Сапсан»", "Скорый электропоезд 839В «Ласточка»" (ref "124Ы"). Suburban
    # trains ("Пригородный электропоезд: Дубна => Савёловский вокзал", МЦД-2, Novosibirsk's
    # numbered "Пригородный электропоезд 6323") are lines.
    ref = tags.get("ref") or ""
    return bool(RU_TRAIN.search(name) or RU_TRAIN.search(ref)
                or RU_TRAIN_PAIR.search(name) or RU_TRAIN_PAIR.search(ref)
                or RU_TRAIN_KIND.match(name))
