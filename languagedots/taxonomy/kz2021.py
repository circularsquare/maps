"""Kazakhstan 2021 census, native language (sources/kz_census.py) -> node.

Keyed by data/normalized/kz.csv's `source_category`, the census engine's `Родной язык` field:
17 named languages and "Другой язык".

CALLS.
  Турецкий: Turkish. Kazakhstan's Turks are Meskhetian (Ahiska) Turks deported from Georgia in
    1944, as in Kyrgyzstan; the census says Turkish. 88,815 people.
  Курдский: the existing `kurdish` leaf. Soviet Kurds speak Kurmanji; no country splits it.
  Дунганский: Dungan, Kyrgyzstan's node (tree.d/kg.txt says why it is a sibling of Mandarin).
  Другой язык: `other`. It is the unnamed remainder of every language outside the 17: by the
    engine's nationality cut it holds Armenian, Moldovan, Ingush, Bashkir, Karakalpak, Georgian,
    Greek and more, and 46,865 Ukrainians and 26,349 Kazakhs whose language it does not name.
    No single narrower node holds all of that. 231,751 people, 1.21%.
"""
SL = "indoeuropean.slavic"

NAMES = {
    "Казахский": "turkic.kazakh",
    "Русский": f"{SL}.east.russian",
    "Узбекский": "turkic.uzbek",
    "Уйгурский": "turkic.uyghur",
    "Татарский": "turkic.tatar",
    "Азербайджанский": "turkic.azerbaijani",
    "Турецкий": "turkic.turkish",
    "Немецкий": "indoeuropean.germanic.continental.german",
    "Украинский": f"{SL}.east.ukrainian",
    "Дунганский": "sinotibetan.sinitic.dungan",
    "Корейский": "koreanic.korean",
    "Таджикский": "indoeuropean.iranian.tajik",
    "Белорусский": f"{SL}.east.belarusian",
    "Курдский": "indoeuropean.iranian.kurdish",
    "Чеченский": "nakhdaghestanian.nakh.chechen",
    "Кыргызский": "turkic.kyrgyz",
    "Польский": f"{SL}.west.polish",
    "Другой язык": "other",
}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"kz2021: unmapped label {label!r}")
    return NAMES[label]
