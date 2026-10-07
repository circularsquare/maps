"""Guam, 2020 Island Areas Census, table PCT25: language spoken at home -> node.

The 12 rows of PCT25 (sources/gu_census.py). One answer per person aged 5+, so "Speak only
English" is English only; a Chamorro-and-English home is under Chamorro.

Calls:
  - "Speak Philippine languages" (33,297, a quarter of Guam) sits on the Philippine group: the
    table names no language, and Guam's Filipinos speak Tagalog, Ilocano, Cebuano, Visayan and
    others. No 2020 Island Areas table splits it (the detailed cross-tabulations use the same
    groups), and the Island Areas have no public microdata. So it draws as "language not named".
  - "Speak other Pacific Island languages" (3,243; Pohnpeian, Yapese, Kosraean, Marshallese and
    the like in practice) on `pacific_other`, not on Austronesian: the Bureau's Pacific Island
    group also holds Papuan languages (tree.d/gu.txt). Kept off `other`, per the brief's rule
    for indigenous remainders.
  - "Speak Chinese" (2,114) on the Sinitic group: variety not named, as us2024 does.
  - "Speak other Asian languages" (487) and "Speak other languages" (1,907) on `other`: the
    first spans several families (Vietnamese, Thai, Hindi...), as us2024 maps "Other Languages
    of Asia".
"""

NAMES = {
    "Speak only English": "indoeuropean.germanic.english",
    "Speak Chamorro": "austronesian.chamorro",
    "Speak Carolinian": "austronesian.oceanic.carolinian",
    "Speak Palauan": "austronesian.palauan",
    "Speak Chuukese": "austronesian.oceanic.chuukese",
    "Speak Philippine languages": "austronesian.philippine",
    "Speak other Pacific Island languages": "pacific_other",
    "Speak Chinese": "sinotibetan.sinitic",
    "Speak Japanese": "japonic.japanese",
    "Speak Korean": "koreanic.korean",
    "Speak other Asian languages": "other",
    "Speak other languages": "other",
}


def resolve(label):
    return NAMES[label]
