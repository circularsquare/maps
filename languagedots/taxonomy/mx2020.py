"""Mexico, Censo 2020: INEGI's indigenous-language label (cube "Habla indigena y lengua (INALI)")
-> node. Labels are exactly as sources/mx_censo.py writes them into data/normalized/mx.csv, typographic
apostrophes included (INEGI writes K’iche’, Q’anjob’al and Q’eqchi’ with U+2019, Ch'ol and Qato'k
with a plain one).

Each of INALI's 68 agrupaciones is one census label and one node (spec §3.1), although several are
whole clusters of mutually unintelligible languages to Glottolog (Zapoteco, Mixteco, Chinanteco,
Mixe, Náhuatl, Otomí): the census does not split them, so neither does the map.

Remainders (spec §3.2), each on the narrowest node holding everything the census filed there:
  "Popoluca insuficientemente especificado"   Mixe-Zoque: in Veracruz "Popoluca" names Sierra
      Popoluca and Texistepequeño (Zoquean) and Oluteco and Sayulteco (Mixean). 55% are in
      Veracruz, the rest scattered with migrants.
  "Tepehuano insuficientemente especificado"  the Tepehuan group (northern or southern).
  "Chontal insuficientemente especificado"    Chontal de Oaxaca is Tequistlatecan and Chontal de
      Tabasco is Mayan, so the two share no family: americas_other. Not split by place, because
      the 1,704 people are not where either Chontal is spoken (Quintana Roo, Chiapas, México,
      Campeche, Veracruz lead; Tabasco and Oaxaca hold few).
  "Otras lenguas indígenas de América"       americas_other: indigenous languages of other
      countries of the Americas outside INALI's catalogue.
  "No especificado"                          americas_other: speaks an indigenous language, which
      was not given. As Colombia's "indígena sin información" (co2018). Almost all are Mexican
      languages; no Mexican-only node exists to hold them.
"""

SPANISH = "indoeuropean.romance.spanish"
AM = "americas_other"

NAMES = {
    # Mayan
    "Huasteco": "mayan.huasteco",
    "Maya": "mayan.yucatecan.maya",
    "Lacandón": "mayan.yucatecan.lacandon",
    "Ch'ol": "mayan.cholan_tzeltalan.chol",
    "Chontal de Tabasco": "mayan.cholan_tzeltalan.chontal_tabasco",
    "Tseltal": "mayan.cholan_tzeltalan.tseltal",
    "Tsotsil": "mayan.cholan_tzeltalan.tsotsil",
    "Tojolabal": "mayan.qanjobalan.tojolabal",
    "Q’anjob’al": "mayan.qanjobalan.qanjobal",
    "Chuj": "mayan.qanjobalan.chuj",
    "Akateko": "mayan.qanjobalan.akateko",
    "Jakalteko": "mayan.qanjobalan.jakalteko",
    "Qato'k": "mayan.qanjobalan.qatok",
    "Mam": "mayan.mamean.mam",
    "Teko": "mayan.mamean.teko",
    "Awakateko": "mayan.mamean.awakateko",
    "Ixil": "mayan.mamean.ixil",
    "K’iche’": "mayan.kichean.kiche",
    "Kaqchikel": "mayan.kichean.kaqchikel",
    "Q’eqchi’": "mayan.kichean.qeqchi",
    # Oto-Manguean
    "Otomí": "otomanguean.otopamean.otomi",
    "Mazahua": "otomanguean.otopamean.mazahua",
    "Matlatzinca": "otomanguean.otopamean.matlatzinca",
    "Tlahuica": "otomanguean.otopamean.tlahuica",
    "Pame": "otomanguean.otopamean.pame",
    "Chichimeco Jonaz": "otomanguean.otopamean.chichimeco",
    "Mazateco": "otomanguean.popolocan.mazateco",
    "Popoloca": "otomanguean.popolocan.popoloca",
    "Chocholteco": "otomanguean.popolocan.chocholteco",
    "Ixcateco": "otomanguean.popolocan.ixcateco",
    "Zapoteco": "otomanguean.zapotecan.zapoteco",
    "Chatino": "otomanguean.zapotecan.chatino",
    "Mixteco": "otomanguean.mixtecan.mixteco",
    "Triqui": "otomanguean.mixtecan.triqui",
    "Cuicateco": "otomanguean.mixtecan.cuicateco",
    "Chinanteco": "otomanguean.chinanteco",
    "Amuzgo": "otomanguean.amuzgo",
    "Tlapaneco": "otomanguean.tlapaneco",
    # Uto-Aztecan
    "Náhuatl": "utoaztecan.nahuatl",
    "Cora": "utoaztecan.corachol.cora",
    "Huichol": "utoaztecan.corachol.huichol",
    "Tarahumara": "utoaztecan.taracahitan.tarahumara",
    "Guarijío": "utoaztecan.taracahitan.guarijio",
    "Mayo": "utoaztecan.taracahitan.mayo",
    "Yaqui": "utoaztecan.taracahitan.yaqui",
    "Tepehuano del norte": "utoaztecan.tepiman.tepehuan.north",
    "Tepehuano del sur": "utoaztecan.tepiman.tepehuan.south",
    "Tepehuano insuficientemente especificado": "utoaztecan.tepiman.tepehuan",
    "Pima": "utoaztecan.tepiman.pima",
    "Pápago": "utoaztecan.tepiman.papago",
    # Mixe-Zoque
    "Mixe": "mixezoque.mixean.mixe",
    "Oluteco": "mixezoque.mixean.oluteco",
    "Sayulteco": "mixezoque.mixean.sayulteco",
    "Zoque": "mixezoque.zoquean.zoque",
    "Popoluca de la Sierra": "mixezoque.zoquean.popoluca_sierra",
    "Texistepequeño": "mixezoque.zoquean.texistepequeno",
    "Ayapaneco": "mixezoque.zoquean.ayapaneco",
    "Popoluca insuficientemente especificado": "mixezoque",
    # Totonac-Tepehua
    "Totonaco": "totonacan.totonaco",
    "Tepehua": "totonacan.tepehua",
    # Cochimí-Yuman
    "Paipai": "yuman.paipai",
    "Kiliwa": "yuman.kiliwa",
    "Kumiai": "yuman.kumiai",
    "Cucapá": "yuman.cucapa",
    # the rest
    "Chontal de Oaxaca": "tequistlatecan.chontal_oaxaca",
    "Tarasco": "isolate.purepecha",
    "Huave": "isolate.huave",
    "Seri": "isolate.seri",
    "Kickapoo": "algic.kickapoo",
    "Chontal insuficientemente especificado": AM,
    "Otras lenguas indígenas de América": AM,
    "No especificado": AM,
}

EXTRA_NODES = [SPANISH]


def resolve(label):
    return NAMES[label]
