"""Macau 2021 Population Census, usual language -> node. Keyed by the census database's English
labels (DSEC, dimension `family_language`, "Usual language" / 日常用語 / Língua corrente).

  * Cantonese is the census's 廣州話, Yue as spoken in Macau and Guangzhou.
  * Mandarin (普通話, Putonghua) goes on `mandarin`, as everywhere else on the map.
  * "Other Chinese dialects" (其他中國方言) is the census's unnamed Sinitic remainder (Hokkien,
    Hakka, Teochew and the rest; the database's 2011 and 2016 cuts have the same seven groups and
    name none of them), so it goes on `sinitic`, drawn as "Chinese, language not named".
  * Tagalog (菲律賓語, Tagalo) goes on `tagalog`.
  * "Others" (其他) is every other language, Indonesian, Vietnamese, Thai, Burmese, Nepali and
    the European languages among them, with nothing finer published: on the root `other`.
  * Not drawn: "Not applicable" (不適用), children under 3, whom the question does not cover.
"""
ST = "sinotibetan.sinitic"

NAMES = {
    "Cantonese": f"{ST}.cantonese",
    "Mandarin": f"{ST}.mandarin",
    "Other Chinese dialects": ST,
    "Portuguese": "indoeuropean.romance.portuguese",
    "English": "indoeuropean.germanic.english",
    "Tagalog": "austronesian.philippine.tagalog",
    "Others": "other",
}


def resolve(label):
    return NAMES[label]
