"""Serbia's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

from rules.shared import EU_TRAIN

# Named trains carry their name in „quotes“: IR 1130/1131 „Тара“ (Subotica - Bar), IR 432/433
# „Ловћен“ (Zemun - Bar). IC „Соко“ is the Belgrade - Novi Sad - Subotica high-speed service,
# nine trains each way: a line.
NAMED = re.compile(r"[„“\"«].+[“”\"»]|^(?:Ловћен|Тара|Lovćen|Tara)$")   # route_masters: bare names
# OSM maps Srbijavoz's InterRegio trains one relation per train ("IR 610: Ужице => Земун",
# one daily pair) or one per group ("IR: Београд центар => Суботица", refs 642;646;650;654).
# A single IR train is a long-distance train of its own; a group is a line riders use.
IR_SINGLE = re.compile(r"^IR\s+\d+\b")


def looks_like_service(tags, name, name_en):
    if EU_TRAIN.search(name):
        return True
    if NAMED.search(name) and "Соко" not in name:
        return True
    return bool(IR_SINGLE.search(name)) and ";" not in (tags.get("ref") or "")
