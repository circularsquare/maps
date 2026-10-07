"""Singapore Census of Population 2020, language most frequently spoken at home -> node.

Keyed by the labels sources/sg_census.py writes. Seven groups by planning area (CT/17596);
"Chinese Dialects" opened nationally into four (Table 41), which countries/sg.py shares out by
that national mix. So every label below is one the census prints.

  * Mandarin is the census's Mandarin: Standard Mandarin, on `mandarin` as everywhere else.
  * Hokkien is the Southern Min of Quanzhou, Zhangzhou and Xiamen, on `min_nan`, as Hong Kong's
    "Fukien" is. Teochew is named apart from it, so it has its own node beside `min_nan`
    (hk2021.py gives the reasoning and Glottolog's placement, chao1238 under Min Nan).
  * Cantonese: `cantonese`.
  * "Other Chinese Dialects" (17,359 nationally: Hakka, Hainanese, Foochow, Henghua,
    Shanghainese and the rest; the census lists no members) is the unnamed Sinitic remainder:
    on `sinitic`, drawn as "Chinese, language not named". It is the census's own label, not
    something this map folded.
  * "Chinese Dialects" itself never reaches resolve(): countries/sg.py replaces it by the four
    labels above before mapping, and resolve() refuses it so a missed split cannot slip onto
    `sinitic` silently.
  * Malay: `malay`. The census's Malay is the language, whatever the speaker's Malay sub-group
    (Javanese and Boyanese descent included; the ethnic dialect-group tables are separate).
  * Tamil: `tamil`.
  * "Other Indian Languages" (23,818: Malayalam, Hindi, Punjabi, Bengali, Gujarati, Telugu,
    Urdu and others, unlisted) crosses Indo-Aryan and Dravidian, so nothing narrower than
    `other` is sure to hold it: the UK's "Any other South Asian language" and the US's "India
    N.E.C." are on `other` for the same reason.
  * "Other Languages" (26,592: residents only, so mostly Filipino, Japanese, Korean, Thai,
    Arabic, European languages...) is on `other`.
  * Not drawn: non-residents, residents under 5, people who live alone or only with unrelated
    people, and people unable to speak; none is inside the question.
"""
ST = "sinotibetan.sinitic"

NAMES = {
    "English": "indoeuropean.germanic.english",
    "Mandarin": f"{ST}.mandarin",
    "Hokkien": f"{ST}.min_nan",
    "Teochew": f"{ST}.teochew",
    "Cantonese": f"{ST}.cantonese",
    "Other Chinese Dialects": ST,
    "Malay": "austronesian.malayic.malay",
    "Tamil": "dravidian.southern.tamil",
    "Other Indian Languages": "other",
    "Other Languages": "other",
}


def resolve(label):
    return NAMES[label]
