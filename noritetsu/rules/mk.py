"""North Macedonia's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

from rules.shared import EU_TRAIN

# "IC 891/892", Trainkos's Pristina - Skopje international pair (suspended since 2020): one
# train, a named train. MŽ's own routes ("Скопје - Тетово - Гостивар - Кичево") are lines.
IC = re.compile(r"^IC\s?\d")


def looks_like_service(tags, name, name_en):
    return bool(EU_TRAIN.search(name) or IC.search(name))
