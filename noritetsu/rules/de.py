"""Germany's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

from rules.shared import EU_TRAIN

# Germany: "ICE 10", "IC 26.1", "ICE 42/ICE 47" are DB's interval lines; "ICE 1001",
# "IC Łużyce" single trains.
DE_LINE = re.compile(r"^(?:ICE|IC)\s?\d{1,2}(?:\.\d)?(?!\d)")
DE_TRAIN = re.compile(r"^(?:ICE|IC)\s?\d{3,5}\b|^IC\s+[A-ZÀ-ŽŁ][a-ząćęłńóśźż]"
                      r"|^Leo Express\b|^KD Premium\b")


def looks_like_service(tags, name, name_en):
    # OSM Germany maps DB Fernverkehr's ICE and IC by DB's own line number, not by train:
    # 27 "ICE 10"-style and 20 "IC 26"-style route_masters, each an hourly or two-hourly
    # interval product a rider uses as a line, as Swiss IC 1 and ÖBB's Railjet are: lines.
    # Single trains are named trains: EC, EN, NJ, European Sleeper, Eurostar, TGV
    # (EU_TRAIN), an ICE or IC by a train number of 3-5 digits, PKP Intercity's named IC
    # (IC Łużyce), Leo Express and KD Premium.
    if DE_LINE.match(name):
        return False
    return bool(EU_TRAIN.search(name) or DE_TRAIN.search(name))
