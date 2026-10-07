"""Belgium's rules for build_model.py (build_model.country_rules lists what it reads)."""
from rules.shared import EU_TRAIN


def looks_like_service(tags, name, name_en):
    # International and long-distance trains mapped one relation per train (EC 112, EN
    # 40467, ICE 43, Eurostar, European Sleeper, Nightjet), as France, Poland, Hungary and
    # Portugal flag theirs. The interval products that are lines to a rider, IC, IR and
    # Railjet (Swiss IC 1, ÖBB's half-hourly Railjet), are left as lines. The same rule in
    # at, be, nl, ch, cz, si, bg and sk.
    return bool(EU_TRAIN.search(name))
