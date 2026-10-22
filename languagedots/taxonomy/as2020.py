"""American Samoa, 2020 Island Areas Census, table PCT24: language spoken at home -> node.

The 6 rows of PCT24 (sources/as_census.py). One answer per person aged 5+ in households, so
"Speak only English" is English only; a Samoan-and-English home is under Samoan.

Calls:
  - "Speak other Pacific Island languages" (1,798) on `pacific_other`, as Guam (taxonomy/gu2020.py):
    in American Samoa mostly Polynesian languages (Tuvaluan, Tokelauan, Niuean, Fijian...), but the
    Bureau's group also files Papuan languages, so no Austronesian node holds everything in it.
  - "Speak Asian languages" (945) on `other`, as Guam's "other Asian languages": the group spans
    several families (Chinese, Korean, Filipino languages...) and the table does not split it.
  - "Speak other languages" (239) on `other`.
"""

NAMES = {
    "Speak only English": "indoeuropean.germanic.english",
    "Speak Samoan": "austronesian.oceanic.samoan",
    "Speak Tongan": "austronesian.oceanic.tongan",
    "Speak other Pacific Island languages": "pacific_other",
    "Speak Asian languages": "other",
    "Speak other languages": "other",
}


def resolve(label):
    return NAMES[label]
