"""Russia, All-Russian Population Census 2020 (held 2021), Volume 5 Table 6, native language
(sources/ru_census.py) -> node.

Keyed by data/normalized/ru.csv's `source_category`: the census's language group as printed
(Russian, stripped). 176 language labels reach the 83 subjects drawn, plus "not stated". Rosstat's own companion,
"Spisok yazykov Rossii v itogakh VPN-2020" (Institute of Linguistics RAS, Koryakov and Davidyuk;
data/raw/ru/Tom5_Spisok_yazykov.doc), says what each group holds; the calls below lean on it.

CALLS.
  Мордовский "Mordvin" (274,876): beside Erzya-Mordvin and Moksha-Mordvin, the census prints people
    who said only "Mordvin". The Institute makes it a macrogroup because the census cannot tell
    which. A named label, so a leaf `uralic.mordvin`, not a group over Erzya and Moksha.
  Марийский "Mari" (318,495): likewise beside Hill Mari and Meadow-Eastern Mari; the existing leaf
    `uralic.mari`. Hill Mari and Meadow-Eastern Mari are leaves of their own.
  Адыгский "Adygsky" (6,923): the generic Circassian self-name, which the Institute splits between
    Adyghe and Kabardino-Cherkess. A leaf `abkhazadyghe.circassian`.
  Кабардино-черкесский "Kabardino-Cherkess": Kabardian (`abkhazadyghe.kabardian`); Cherkess is the
    same language under the Karachay-Cherkessia name.
  Дагестанский "Dagestani" (34,244): no such language; the Institute splits it over Avar, Aghul,
    Dargwa, Kumyk, Lak, Lezgian, Rutul and others, Turkic and Daghestanian both, so it sits on a
    leaf under `other` (az.txt's "Jewish" precedent), not guessed into one.
  Еврейский "Jewish" (3,675): Yiddish or Juhuri per the Institute; `other.jewish`, the node
    Azerbaijan's census label already uses.
  Татский "Tat" (720): in Russia mostly Juhuri, the Mountain Jews' language, which Soviet usage
    called Tat; the label is Tat and Juhuri sits in the Tat group, so `indoeuropean.iranian.tat`.
  Тюркский "Turkic" (946): the Institute: a name the Meskhetian Turks use for their language
    (Ahiska). `turkic.ahiska`. Турецкий "Turkish" stays Turkish, though many of its 115,838 are
    Meskhetian Turks too: the label is what they said.
  Булгарский "Bulgar" (157): the census's label; mostly a name for Tatar per the Institute. Drawn
    as its own leaf under Turkic rather than moved into Tatar.
  Китайский "Chinese": `sinotibetan.sinitic`, drawn unwashed (build.py UNWASHED), as in au, ca, cz.
  Цыганский "Gypsy/Romani": `indoeuropean.indoaryan.romani.romani` (ua2001's leaf).
  Эскимосский "Eskimo" (816) and Юитский "Yuit" (1): Siberian Yupik under two names; two leaves.
  Старославянский "Old Church Slavonic" (71): Church Slavonic, directly under Slavic (tree.d/ru.txt says why).
  Указавшие другие ответы "gave other answers" (34,340 nationally): answers the census could not
    code as a language: `other`.
  Не указавшие родной язык "native language not stated": None, the gap.
  Габонский "Gabonese" (2) appears on the federation sheet only, never in a subject; not mapped.
"""
IA = "indoeuropean.indoaryan"
IR = "indoeuropean.iranian"
SL = "indoeuropean.slavic"
RO = "indoeuropean.romance"
GE = "indoeuropean.germanic"
ND = "nakhdaghestanian"
CK = "chukotkokamchatkan"

NAMES = {
    # East Slavic and other Indo-European
    "Русский": f"{SL}.east.russian",
    "Украинский": f"{SL}.east.ukrainian",
    "Белорусский": f"{SL}.east.belarusian",
    "Русинский": f"{SL}.east.rusyn",
    "Польский": f"{SL}.west.polish",
    "Чешский": f"{SL}.west.czech",
    "Словацкий": f"{SL}.west.slovak",
    "Болгарский": f"{SL}.south.bulgarian",
    "Македонский": f"{SL}.south.macedonian",
    "Словенский": f"{SL}.south.slovenian",
    "Сербскохорватский": f"{SL}.south.serbocroatian",
    "Старославянский": f"{SL}.church_slavonic",
    "Литовский": "indoeuropean.baltic.lithuanian",
    "Латышский": "indoeuropean.baltic.latvian",
    "Армянский": "indoeuropean.armenian.armenian",
    "Греческий": "indoeuropean.hellenic.greek",
    "Албанский": "indoeuropean.albanian.albanian",
    "Немецкий": f"{GE}.continental.german",
    "Идиш": f"{GE}.continental.yiddish",
    "Нидерландский": f"{GE}.continental.dutch",
    "Английский": f"{GE}.english",
    "Датский": f"{GE}.north.danish",
    "Норвежский": f"{GE}.north.norwegian",
    "Шведский": f"{GE}.north.swedish",
    "Исландский": f"{GE}.north.icelandic",
    "Ирландский": "indoeuropean.celtic.irish",
    "Молдавский": f"{RO}.moldovan",
    "Румынский": f"{RO}.romanian",
    "Французский": f"{RO}.french",
    "Испанский": f"{RO}.spanish",
    "Итальянский": f"{RO}.italian",
    "Португальский": f"{RO}.portuguese",
    "Латинский": f"{RO}.latin",
    "Осетинский": f"{IR}.ossetian",
    "Таджикский": f"{IR}.tajik",
    "Персидский": f"{IR}.persian",
    "Дари": f"{IR}.dari",
    "Пушту": f"{IR}.pashto",
    "Курдский": f"{IR}.kurdish",
    "Талышский": f"{IR}.talysh",
    "Татский": f"{IR}.tat",
    "Цыганский": f"{IA}.romani.romani",
    "Хинди": f"{IA}.central.hindi",
    "Урду": f"{IA}.central.urdu",
    "Бенгали": f"{IA}.eastern.bengali",
    # Turkic
    "Татарский": "turkic.tatar",
    "Башкирский": "turkic.bashkir",
    "Чувашский": "turkic.chuvash",
    "Кумыкский": "turkic.kumyk",
    "Карачаево-балкарский": "turkic.karachay_balkar",
    "Ногайский": "turkic.nogai",
    "Ногайско-карагашский": "turkic.karagash",
    "Юртовско-татарский": "turkic.yurt_tatar",
    "Якутский": "turkic.yakut",
    "Долганский": "turkic.dolgan",
    "Тувинский": "turkic.tuvan",
    "Тофаларский": "turkic.tofa",
    "Алтайский": "turkic.altai",
    "Тубаларский": "turkic.tubalar",
    "Кумандинский": "turkic.kumandy",
    "Челканский": "turkic.chelkan",
    "Телеутский": "turkic.teleut",
    "Хакасский": "turkic.khakas",
    "Шорский": "turkic.shor",
    "Чулымско-тюркский": "turkic.chulym",
    "Казахский": "turkic.kazakh",
    "Киргизский": "turkic.kyrgyz",
    "Узбекский": "turkic.uzbek",
    "Уйгурский": "turkic.uyghur",
    "Туркменский": "turkic.turkmen",
    "Каракалпакский": "turkic.karakalpak",
    "Азербайджанский": "turkic.azerbaijani",
    "Турецкий": "turkic.turkish",
    "Тюркский": "turkic.ahiska",
    "Гагаузский": "turkic.gagauz",
    "Крымскотатарский": "turkic.crimean_tatar",
    "Караимский": "turkic.karaim",
    "Булгарский": "turkic.bulgar",
    # Uralic
    "Марийский": "uralic.mari",
    "Горномарийский": "uralic.hill_mari",
    "Лугово-восточный марийский": "uralic.meadow_mari",
    "Удмуртский": "uralic.udmurt",
    "Коми": "uralic.komi",
    "Коми-пермяцкий": "uralic.komi_permyak",
    "Мордовский": "uralic.mordvin",
    "Эрзя-мордовский": "uralic.erzya",
    "Мокша-мордовский": "uralic.moksha",
    "Карельский": "uralic.karelian",
    "Вепсский": "uralic.veps",
    "Водский": "uralic.votic",
    "Ижорский": "uralic.izhorian",
    "Финский": "uralic.finnish",
    "Эстонский": "uralic.estonian",
    "Венгерский": "uralic.hungarian",
    "Саамский": "uralic.saami",
    "Хантыйский": "uralic.khanty",
    "Мансийский": "uralic.mansi",
    "Ненецкий": "uralic.nenets",
    "Энецкий": "uralic.enets",
    "Нганасанский": "uralic.nganasan",
    "Селькупский": "uralic.selkup",
    # Mongolic, Tungusic
    "Бурятский": "mongolic.buryat",
    "Калмыцкий": "mongolic.kalmyk",
    "Монгольский": "mongolic.mongolian",
    "Эвенкийский": "tungusic.evenki",
    "Эвенский": "tungusic.even",
    "Нанайский": "tungusic.nanai",
    "Ульчский": "tungusic.ulch",
    "Удэгейский": "tungusic.udege",
    "Орочский": "tungusic.oroch",
    "Негидальский": "tungusic.negidal",
    "Уйльта": "tungusic.uilta",
    # Nakh-Daghestanian
    "Чеченский": f"{ND}.nakh.chechen",
    "Ингушский": f"{ND}.nakh.ingush",
    "Аварский": f"{ND}.avarandic.avar",
    "Андийский": f"{ND}.avarandic.andi",
    "Ахвахский": f"{ND}.avarandic.akhvakh",
    "Багвалинский": f"{ND}.avarandic.bagvalal",
    "Ботлихский": f"{ND}.avarandic.botlikh",
    "Годоберинский": f"{ND}.avarandic.godoberi",
    "Каратинский": f"{ND}.avarandic.karata",
    "Тиндальский": f"{ND}.avarandic.tindi",
    "Чамалинский": f"{ND}.avarandic.chamalal",
    "Цезский": f"{ND}.tsezic.tsez",
    "Гинухский": f"{ND}.tsezic.hinukh",
    "Гунзибский": f"{ND}.tsezic.hunzib",
    "Бежтинский": f"{ND}.tsezic.bezhta",
    "Хваршинский": f"{ND}.tsezic.khvarshi",
    "Даргинский": f"{ND}.dargwa",
    "Лакский": f"{ND}.lak",
    "Лезгинский": f"{ND}.lezgic.lezgian",
    "Табасаранский": f"{ND}.lezgic.tabasaran",
    "Агульский": f"{ND}.lezgic.agul",
    "Рутульский": f"{ND}.lezgic.rutul",
    "Цахурский": f"{ND}.lezgic.tsakhur",
    "Арчинский": f"{ND}.lezgic.archi",
    "Удинский": f"{ND}.lezgic.udi",
    # Northwest Caucasian, Kartvelian
    "Кабардино-черкесский": "abkhazadyghe.kabardian",
    "Адыгейский": "abkhazadyghe.adyghe",
    "Адыгский": "abkhazadyghe.circassian",
    "Абазинский": "abkhazadyghe.abaza",
    "Абхазский": "abkhazadyghe.abkhaz",
    "Грузинский": "kartvelian.georgian",
    "Мегрельский": "kartvelian.mingrelian",
    # Paleo-Siberian families and isolates, Eskimo-Aleut
    "Чукотский": f"{CK}.chukchi",
    "Корякский": f"{CK}.koryak",
    "Ительменский": f"{CK}.itelmen",
    "Алюторский": f"{CK}.alutor",
    "Керекский": f"{CK}.kerek",
    "Юкагирский": "yukaghir.yukaghir",
    "Кетский": "yeniseian.ket",
    "Югский": "yeniseian.yugh",
    "Нивхский": "isolate.nivkh",
    "Эскимосский": "eskimoaleut.yupik",
    "Юитский": "eskimoaleut.yuit",
    "Алеутский": "eskimoaleut.aleut",
    # the rest of the world
    "Арабский": "afroasiatic.arabic",
    "Иврит": "afroasiatic.hebrew",
    "Ассирийский": "afroasiatic.assyrian",
    "Амхарский": "afroasiatic.ethiosemitic.amharic",
    "Китайский": "sinotibetan.sinitic",
    "Дунганский": "sinotibetan.sinitic.dungan",
    "Тибетский": "sinotibetan.tibetic.tibetan",
    "Бирманский": "sinotibetan.burmish.burmese",
    "Корейский": "koreanic.korean",
    "Японский": "japonic.japanese",
    "Вьетнамский": "austroasiatic.vietnamese",
    "Тайский": "kradai.thai",
    "Индонезийский": "austronesian.malayic.indonesian",
    "Малайский": "austronesian.malayic.malay",
    "Маори": "austronesian.oceanic.maori",
    "Тамильский": "dravidian.southern.tamil",
    "Суахили": "nigercongo.bantu.swahili",
    # not languages, or not one
    "Русский жестовый язык глухих": "signlanguage.rsl",
    "Еврейский": "other.jewish",
    "Дагестанский": "other.dagestani",
    "Указавшие другие ответы": "other",
}
GAP = "Не указавшие родной язык"


def resolve(label):
    if label == GAP:
        return None
    if label not in NAMES:
        raise KeyError(f"ru2021: unmapped label {label!r}")
    return NAMES[label]
