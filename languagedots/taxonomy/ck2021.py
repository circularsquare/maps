"""Cook Islands, 2021 census Q18, language spoken at home (sources/ck_census.py, sources/ck.md).

Two yes/no answers per person aged 5+: Cook Islands Maori, and "any other language(s)", which the
census does not name. A person who answered yes to both is shared half and half (spec §3.6).

  - "Cook Islands Maori": its node. The question does not separate the island varieties (Rarotongan,
    Aitutaki, Penrhyn, Manihiki-Rakahanga...), and Pukapukan, a Samoic language, may be inside it
    on Pukapuka (386 of 386 people there answered yes).
  - "Other language": English. Not named by the census; English is the second official language
    and the language of schooling, and the ethnicity table shows 2,138 residents of other ethnic
    origin (Fijian, other Pacific, Indonesian, Filipino and others) whose languages would be
    the rest. A judgement call recorded in sources/ck.md; a bare `other` would put 59% of people on
    the unnamed remainder.
"""
NAMES = {
    "Cook Islands Maori": "austronesian.oceanic.cook_islands_maori",
    "Other language": "indoeuropean.germanic.english",
}


def resolve(label):
    return NAMES[label]
