"""New Zealand, Census 2023, languages spoken -> node. Keyed by the categories
sources/nz_census.py writes, which are persons already shared across the languages they named.

NAMED AS LANGUAGES, level 1 (SA1 layer): English, Maori, Samoan and New Zealand Sign Language,
each on the node other countries use (Maori is au.txt's, Samoan fi.txt's); NZSL is new in
tree.d/nz.txt.

NAMED AS LANGUAGES, from Aotearoa Data Explorer's CEN23_ECI_011 at SA2 (the split of level 1's
Other; sources/nz.md), each on the node other countries use:
  - "Northern Chinese" is Stats NZ's name for Mandarin and the northern varieties, on hk.txt's
    Mandarin.
  - "Yue" is Stats NZ's name for Cantonese and the other Yue varieties; no `yue` group exists
    (hk.txt has Cantonese and Sze Yap as siblings), and in New Zealand it is overwhelmingly
    Cantonese, so it sits on Cantonese.
  - "Panjabi" on tree.txt's Punjabi (a spelling variant).

NAMED AS A GROUP:
  - "Sinitic not further defined" is people who answered "Chinese" without a variety, on the
    Sinitic group `sinotibetan.sinitic` ("Chinese"), drawn washed out as "language not named".
  - "Other" is every language but the fifteen above that a person named: Korean, Japanese, Dutch,
    Fijian, Cook Islands Maori, Gujarati, Russian, Arabic and a hundred more. It spans a dozen
    families, so the narrowest node holding it is the root `other` (spec §3.2). Where an SA2's
    split has a confidential cell (124 small SA2s, 21 people's worth) Other stays whole and also
    holds the eleven there.

NOT DRAWN: "None (eg too young to talk)", 104,847 people, the gap. "Not elsewhere included" is
zero in 2023 (Stats NZ imputed missing answers).
"""
NAMES = {
    "English": "indoeuropean.germanic.english",
    "Māori": "austronesian.oceanic.maori",
    "Samoan": "austronesian.oceanic.samoan",
    "New Zealand Sign Language": "signlanguage.nzsl",
    "Northern Chinese": "sinotibetan.sinitic.mandarin",
    "Hindi": "indoeuropean.indoaryan.central.hindi",
    "Tagalog": "austronesian.philippine.tagalog",
    "Sinitic not further defined": "sinotibetan.sinitic",
    "Yue": "sinotibetan.sinitic.cantonese",
    "French": "indoeuropean.romance.french",
    "Panjabi": "indoeuropean.indoaryan.northwestern.punjabi",
    "Afrikaans": "indoeuropean.germanic.continental.afrikaans",
    "Spanish": "indoeuropean.romance.spanish",
    "German": "indoeuropean.germanic.continental.german",
    "Tongan": "austronesian.oceanic.tongan",
    "Other": "other",
}
EXTRA_NODES = []


def resolve(name):
    return NAMES.get(name)
