"""Hong Kong 2021 Population Census, usual spoken language -> node. Keyed by the census's labels.

The census's 15-group classification (IDDS grouping LANG1_5_15G), published by district; the
five-group subtotals ("Other Chinese dialects", "Other languages") that the small-area release
carries are opened into it in countries/hk.py, so every label below is one the census prints.

  * Cantonese is the census's Cantonese (Yue as spoken in Hong Kong and Guangzhou). Sze Yap is
    the Siyi Yue of Taishan, Kaiping, Enping and Xinhui (the census's own gloss: "San Wui, Hoi
    Ping, Yan Ping, Toi Shan"); Glottolog files it under Yue beside Cantonese, and it gets a node
    of its own because the census names it.
  * Putonghua is Standard Mandarin, drawn on `mandarin` as everywhere else on the map.
  * Fukien is the census's name for Hokkien, the Southern Min of Quanzhou, Zhangzhou and Xiamen
    (Hong Kong's Fujianese came mostly from Jinjiang and Shishi), so it is drawn on `min_nan`.
    Chiu Chau is Teochew, also Southern Min; the census names it apart from Fukien, so it is a
    node of its own beside `min_nan` (Glottolog has Chaozhou, chao1238, as a dialect of Min Nan;
    Canada's census lumps the two, Hong Kong's does not). It sits beside `min_nan` rather than
    under it because a child would turn `min_nan` into a group node and wash out every Hokkien
    dot already drawn in other countries.
  * Shanghainese is drawn on `wu`, whose label already says so.
  * "Other Chinese dialects" is the census's unnamed Sinitic remainder (Hainanese, Fuzhou, the
    Hunan and Sichuan varieties...): on `sinitic`, drawn as "Chinese, language not named".
  * "Filipino (Tagalog)" goes on `tagalog`, as the UK's "Tagalog or Filipino" does.
  * "Others" is everything else: the IDDS code list behind it (41, 43, 46-49, 51-54, 59-67, 69,
    92) spans Urdu, Nepali, Hindi, Punjabi, Korean, Vietnamese, French and more, so the narrowest
    node holding it is the root `other`.
  * Not drawn: children under 5 and mute persons, whom the question does not cover.
"""
ST = "sinotibetan.sinitic"

NAMES = {
    "Cantonese": f"{ST}.cantonese",
    "Putonghua": f"{ST}.mandarin",
    "Hakka": f"{ST}.hakka",
    "Chiu Chau": f"{ST}.teochew",
    "Fukien": f"{ST}.min_nan",
    "Sze Yap": f"{ST}.siyi",
    "Shanghainese": f"{ST}.wu",
    "Other Chinese dialects": ST,
    "English": "indoeuropean.germanic.english",
    "Filipino (Tagalog)": "austronesian.philippine.tagalog",
    "Indonesian (Bahasa Indonesia)": "austronesian.malayic.indonesian",
    "Japanese": "japonic.japanese",
    "Thai": "kradai.thai",
    "Others": "other",
}


def resolve(label):
    return NAMES[label]
