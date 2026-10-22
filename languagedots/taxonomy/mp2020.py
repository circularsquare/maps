"""Northern Mariana Islands, 2020 Island Areas Census, table PCT24: language spoken at home -> node.

The 12 rows of PCT24 (sources/mp_census.py) are the same twelve as Guam's PCT25, so the mapping is
Guam's (taxonomy/gu2020.py) and its calls stand: Philippine languages on the Philippine group
(the table names no language), other Pacific Island languages on `pacific_other` (in the CNMI
mostly Yapese, Pohnpeian, Kosraean, Marshallese and the other Micronesian languages, but the
Bureau's group also files Papuan languages), Chinese on the Sinitic group, other Asian languages
and other languages on `other`. One answer per person aged 5+, so "Speak only English" is English
only.

Carolinian (1,851 people, 4.2%) is a language of the islands themselves here, Chamorro (9,579)
the other.
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
