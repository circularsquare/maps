"""Colombia, CNPV 2018: "habla la lengua nativa de su pueblo" -> node, keyed by the census code.

THE CENSUS NAMES A PEOPLE, NOT A LANGUAGE. A yes to PA_HABLA_LENG means the person speaks the
native language of the pueblo they gave in PA11_COD_ETNIA, so each pueblo code maps to that
people's language: one node per code (spec §3.1), even where Glottolog would call two codes one
language (Embera and its four named varieties; Amorúa, Wipiwi and the other Guahibo peoples).
Labels in sources/co_cnpv.py's co_pueblos.csv are DANE's: the code, the name, then its synonyms.
Families from Glottolog's classification (data/raw/glottolog), checked per language.

Where a language is already a node because Brazil has it, the code reuses that node:
Wayuu, Curripaco (Kuripako), Baniba (Baniwa), Guariquema (= Guarequena, Warekena), Tariano,
Tikuna, Puinave, Hupdu (Hup), Juhup (Yuhup), Cocama (Kokama), Yeral (Ñengatú = Nheengatu),
Murui (Witóto), Miraña, and the Tucanoan languages of the Vaupés (Tukano, Cubeo, Desano,
Guanano/Wanano, Piratapuyo, Siriano, Tuyuka, Bara, Barasano, Makuna, Carapana, Tanimuka).
Kichwa (840) reuses Brazil's "Quíchua" (Ecuadorian Kichwa); Otavaleño (830) is its own node.

THE PEOPLES WHOSE LANGUAGE IS NO LONGER SPOKEN. Some pueblos answered yes although Glottolog
lists their ancestral language as extinct (AES 6) or has no entry for it at all: Zenú 40,777 of
307,034, Pastos 14,039 of 163,682, Pijao 6,822, Yanacona 4,503, Mokaná 2,870, Muisca 1,665,
Totoró 1,453, Quillacinga 1,023, Kankuamo 753, and a few hundred across a dozen more. These are
drawn as the census recorded them, each on a node of its own, as Brazil's fragment draws the
revived languages of the Northeast (Tupinambá, Pankararú) under `unclassified`. Where Glottolog
classifies the extinct language they go under its family (Muisca = Chibcha, Kankuamo and Tairona
Chibchan; Coconuco and the Coconuco-area peoples Polindara, Ambaló and Quizgo Barbacoan, beside
Namtrik and Totoró; Dujos = Tama, Tucanoan; Betoye = Betoi-Jirara and Andakíes = Andaquí,
isolates); otherwise `unclassified` (Zenú, Pasto, Pijao, Yanacona, Mokaná, Quillacinga, Guane,
Nutabe, Guanaco, Chitarero, Quimbaya, Calima, Panche, Cañamomo Lomaprieta, Yarí). They are
most likely heritage or revival speakers; sources/co.md and note_public say so.

Unsure placements, small: Je'eruriwa (15) under Tucanoan (an Apaporis people listed beside the
Tanimuka and Letuama); Judpa (17) its own Naduhup node beside Hup and Yuhup; Maku (435, 24),
a generic name for the Naduhup and Kakua-Nukak peoples, on the `nadahup` group itself.

Remainders (spec §3.2): "Indígenas Ecuador / Perú / Venezuela / México / Brasil / Panamá /
Bolivia" and "Maya (Guatemala)" (an indigenous language of another country, not named) and
999 "Indígena sin información" (speaks their people's language, people not given) on
`americas_other`, the narrowest node holding every indigenous language of the Americas.

The other ethnic groups: Rrom (1001) speak Romanés, the Vlax Romani of Colombia's Kalderash
vitsas; Raizales (1002) San Andrés Creole; Palenqueros (1003) Palenquero.
Not drawn: 1010, people who do not speak their own people's language but do speak another
native language the census does not name (17,988), and 1011, no answer (15,354); `gap`.
Code 0, everyone else, is Spanish, tier `derived` (spec §3.5).
"""

SPANISH = "indoeuropean.romance.spanish"
AM = "americas_other"

CODES = {
    0: SPANISH,
    10: "arawakan.achagua",
    20: "guahiboan.amorua",
    21: "guahiboan.wipiwi",
    25: "guahiboan.yamalero",
    26: "isolate.pume",
    30: "isolate.andoque",
    40: "chibchan.arhuaco",
    50: "chibchan.wiwa",
    60: "tucanoan.bara",
    70: "tucanoan.barasana",
    80: "chibchan.bari",
    90: "isolate.betoi",
    100: "boran.bora",
    110: "arawakan.cabiyari",
    130: "tucanoan.karapana",
    140: "cariban.carijona",
    150: "chibchan.ette_taara",
    160: "guahiboan.chiricoa",
    170: "tupian.tupiguarani.kokama",
    180: "barbacoan.coconuco",
    190: "tucanoan.koreguaje",
    200: "unclassified.pijao",
    210: "barbacoan.awa_pit",
    220: "tucanoan.kubeo",
    230: "guahiboan.cuiba",
    240: "chibchan.tule",
    250: "arawakan.kuripako",
    251: "arawakan.baniwa",
    252: "arawakan.warekena",       # Guariquema = Guarequena
    260: "tucanoan.desana",
    270: "tucanoan.dujos",
    280: "chocoan.embera.embera",
    281: "chocoan.embera.katio",
    282: "chocoan.embera.chami",
    283: "chocoan.embera.eperara",
    284: "chocoan.embera.dobida",
    285: "unclassified.nutabe",
    290: "barbacoan.namtrik",
    291: "barbacoan.ambalo",
    292: "barbacoan.quizgo",
    300: "unclassified.guanaco",
    310: "tucanoan.wanano",
    320: "guahiboan.jiw",
    330: "unclassified.canamomo",
    340: "quechuan.inga",
    350: "isolate.kamentsa",
    360: "isolate.cofan",
    370: "chibchan.kogui",
    380: "tucanoan.letuama",
    390: "tucanoan.makaguaje",
    400: "guahiboan.hitnu",
    401: "guahiboan.macaguane",
    410: "tucanoan.makuna",
    430: "nadahup.nukak",
    431: "nadahup.kakua",
    432: "nadahup.hupd_ah",
    433: "nadahup.yuhupdeh",
    434: "nadahup.judpa",
    435: "nadahup",                  # "Maku": the generic name, no one language
    440: "guahiboan.masiguare",
    450: "arawakan.matapi",
    455: "tucanoan.jeeruriwa",
    460: "boran.miranha",
    470: "chibchan.muisca",
    480: "witotoan.nonuya",
    490: "witotoan.ocaina",
    500: "isolate.nasa",
    501: "barbacoan.polindara",
    505: "isolate.andaqui",
    510: "arawakan.piapoco",
    520: "saliban.piaroa",
    530: "tucanoan.piratapuia",
    540: "tucanoan.pisamira",
    550: "nadahup.puinave",
    560: "unclassified.pasto",
    565: "unclassified.quillacinga",
    570: "saliban.saliba",
    580: "guahiboan.sikuani",
    585: "guahiboan.mapayerri",
    590: "tucanoan.siona",
    600: "tucanoan.siriano",
    610: "tucanoan.taiwano",
    620: "tucanoan.tanimuka",
    621: "isolate.tinigua",
    630: "arawakan.tariana",
    640: "tucanoan.tatuyo",
    650: "barbacoan.totoro",
    660: "isolate.tikuna",
    670: "guahiboan.tsiripu",
    680: "tucanoan.tukano",
    690: "chibchan.uwa",
    700: "tucanoan.tuyuca",
    710: "chocoan.wounaan",
    720: "arawakan.wayuu",
    730: "witotoan.witoto",           # Murui (Huitoto, Uitoto, Minika...)
    731: "boran.muinane",
    732: "unclassified.yari",
    740: "pebayaguan.yagua",
    750: "unclassified.yanacona",
    760: "tucanoan.yauna",
    770: "arawakan.yukuna",
    780: "cariban.yukpa",
    790: "tucanoan.yuruti",
    800: "unclassified.zenu",
    810: "unclassified.guane",
    820: "unclassified.mokana",
    830: "quechuan.otavaleno",
    840: "quechuan.quichua",         # Kichwa, Brazil's node
    850: "chibchan.kankuamo",
    855: "chibchan.tairona",
    860: "unclassified.chitarero",
    870: "unclassified.quimbaya",
    880: "unclassified.calima",
    890: "unclassified.panche",
    900: AM, 910: AM, 920: AM, 930: AM, 940: AM, 950: AM, 960: AM, 970: AM,
    941: "tupian.tupiguarani.nheengatu",   # Yeral (Ñengantu)
    999: AM,
    1001: "indoeuropean.indoaryan.romani.vlax",
    1002: "creole.english_based.san_andres",
    1003: "creole.spanish_based.palenquero",
    1010: None,
    1011: None,
}


def resolve(code):
    return CODES[int(code)]
