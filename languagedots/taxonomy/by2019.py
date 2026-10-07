"""Belarus, 2019 census, native language (sources/by_census.py, Belstat cube tb503) -> node.

Keyed by data/normalized/by.csv's `source_category`: Belstat's language code list `tb016` as
printed, lower case. The list is the Soviet-lineage one Russia's census also uses, so a label
Russia's mapping already has (taxonomy/ru2021.py, capitalised there) goes to the same node; the
labels below are the ones Russia's 2021 table did not print, or printed differently.

CALLS.
  афганский "Afghan" (308 native): no language, the Afghans' own; `other.afghan`, the leaf Tajikistan's
    census label already uses (Dari, Pashto or another).
  австралийский "Australian" (10): a family name, not a language, and nothing says which; a leaf
    `other.australian`, as Afghan, rather than the `australian` root, which would draw it as
    "Australian Indigenous languages, language not named" over Belarus.
  гэльский "Gaelic" (4): Scottish Gaelic. The list prints ирландский (Irish) separately.
  орокский "Orok" (3): the Russian name of Uilta; `tungusic.uilta`.
  цахрский "Tsakhur" (1): a spelling of Tsakhur (Russia's list writes цахурский).
  талышсккий (37): Talysh, typo in Belstat's code list.
  мокшанский "Moksha" (94): Moksha (Russia's list writes мокша-мордовский).
  тазский диалект "Taz dialect" (1): the Taz of Primorye, whose speech is a northern Chinese
    variety; a leaf under Sinitic.
  Pamir languages (Shughni 1, Bartangi 1, Oroshori 1, Wakhi 1, Yazghulami 4, Bajui and Ishkashimi
    at home only): each its own leaf, flat under Iranian, as Russia's Iranian leaves are. Glottolog
    makes Bartangi and Bajui dialects of Shughni; the census prints each, so each is a leaf.
  чуванский "Chuvan" (1): the extinct Yukaghir language of the Chuvans (Glottolog chuv1256 in
    Yukaghir); `yukaghir.chuvan`.
  барабинский "Barabin" (4): Baraba Tatar, a Siberian Tatar dialect; its own leaf under Turkic.
  нагайбакский "Nagaybak" (1): the Nagaybaks' Tatar dialect; its own leaf under Turkic.
  хемшильский "Hemshin" (home only): Homshetsi, the Hemshin's Armenian dialect.
  другой язык "other language" (542): `other`.
  язык не указан в переписном листе "not stated" (221,588, 2.35%): None, the gap.
Every other tiny label (Aleut 45, Koryak 11, Nivkh 1...) is drawn as printed: the census coded
it, and a coding slip cannot be told from a migrant.
"""
from ru2021 import NAMES as _RU

IR = "indoeuropean.iranian"

NAMES = {
    "афганский": "other.afghan",
    "австралийский": "other.australian",
    "белуджский": f"{IR}.balochi",
    "тиндинский": "nakhdaghestanian.avarandic.tindi",
    "ливский": "uralic.livonian",
    "баскский": "isolate.basque",
    "барабинский": "turkic.barabin",
    "гэльский": "indoeuropean.celtic.scottishgaelic",
    "язгулямский": f"{IR}.yazghulami",
    "орокский": "tungusic.uilta",
    "черногорский": "indoeuropean.slavic.south.montenegrin",
    "чулымский": "turkic.chulym",
    "шугнанский": f"{IR}.shughni",
    "бартаганский": f"{IR}.bartangi",
    "ваханский": f"{IR}.wakhi",
    "ишикашимский": f"{IR}.ishkashimi",
    "баджувский": f"{IR}.bajui",
    "орошорский": f"{IR}.oroshori",
    "чуванский": "yukaghir.chuvan",
    "цахрский": "nakhdaghestanian.lezgic.tsakhur",
    "сванский": "kartvelian.svan",
    "нагайбакский": "turkic.nagaybak",
    "тазский диалект": "sinotibetan.sinitic.taz",
    "бацбийский": "nakhdaghestanian.nakh.bats",
    "хемшильский": "indoeuropean.armenian.homshetsi",
    "крымско-татарский": "turkic.crimean_tatar",
    "мокшанский": "uralic.moksha",
    "талышсккий": f"{IR}.talysh",
    "голландский": "indoeuropean.germanic.continental.dutch",
    "сербский": "indoeuropean.slavic.south.serbian",
    "хорватский": "indoeuropean.slavic.south.croatian",
    "другой язык": "other",
}
GAP = "язык не указан в переписном листе"
# labels on the code list that carry nobody in 2019 and name no one language
NOT_LANGUAGES = {"два и более языка", "своей национальности"}


def resolve(label):
    if label == GAP:
        return None
    if label in NOT_LANGUAGES:
        raise KeyError(f"by2019: {label!r} has people now; it names no language, decide how")
    if label in NAMES:
        return NAMES[label]
    cap = label[:1].upper() + label[1:]
    if cap in _RU and _RU[cap] is not None:
        return _RU[cap]
    raise KeyError(f"by2019: unmapped label {label!r}")
