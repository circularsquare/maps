"""Venezuela, Censo 2011: the indigenous pueblo (PERSONA.CUALINDIGE) -> the node of its language.

THE CENSUS NAMES A PEOPLE, NOT A LANGUAGE, and its language answers are not published by place
(sources/ve.md; ask 007). Anita's ruling on ask 007 (2026-10-05) is a model: each pueblo's people
are drawn on the language of that pueblo, scaled within each state to INE's published share of
indigenous people who speak their pueblo's language (countries/ve.py). So each code below says
which language a SPEAKER of that pueblo would speak; how many speak it is the model's business.

ONE NODE PER CENSUS LABEL (spec §3.1), except spelling variants and exonym/endonym pairs of one
people, which merge here with a comment. INE's table 22 groups the same pairs (sources/ve_censo.py
PUBLISHED); the merges below follow it except where the census prints a separate people
(Amorúa beside the Jivi, Shiriana beside the Yanomami), which keep a node of their own.

WHERE A LANGUAGE IS NO LONGER SPOKEN, the pueblo is drawn as Spanish (EXTINCT below), the rule
ask 007 set out: Glottolog's agglomerated endangerment status (data/raw/glottolog, `aes`) at 5
(moribund) or 6 (extinct), or no Glottolog entry and generally described as extinct. Without it a
flat state rate would invent speakers of dead languages (about 26,000 Añú in Zulia alone). The
ask listed eleven pueblos; Glottolog also puts four small living-name pueblos at aes 5 (Yabarana,
Mapoyo, Sáliba, Arutani: 1,227 people) and the same rule takes them too.

Families from Glottolog's classification, checked per language; nodes Brazil or Colombia already
define are reused (Wayuu, Warao, Pemon's Arekuna, Kamarakoto and Taurepang, Eñepa, Ye'kwana,
Kariña as Brazil's Galibi Kali'na, Sanumá, Yanomami, Xiriana, Piapoco, Puinave, Kuripako,
Warekena, Wapixana, Makuxi, Akawaio, Kubeo, Wanano, Tukano, Nheengatu, Sikuani, Amorúa, Cuiba,
Pumé, Piaroa, Barí, Yukpa, U'wa, Inga; Wichí from Argentina).
"""

SPANISH = "indoeuropean.romance.spanish"
AM = "americas_other"

# Pueblos whose language is gone: drawn as Spanish, never as speakers. Glottolog aes in brackets.
EXTINCT = {
    10: "Añú (Paraujano, aes 5)", 11: "Paraujano (aes 5)",
    50: "Baré (aes 6)",
    70: "Chaima (aes 6)",
    120: "Mapoyo (aes 5)", 121: "Wanai = Mapoyo (aes 5)",
    200: "Sapé (aes 6)",
    201: "Arutani (aes 5)", 219: "Uruak = Arutani (aes 5)",
    260: "Yavarana (Yabarana, aes 5)",
    380: "Sáliva (Sáliba, aes 5)",
    500: "Guaiquerí (no Glottolog entry; extinct)", 501: "Waikerí (no entry; extinct)",
    510: "Caquetío (no entry; extinct)", 511: "Kaketío (no entry; extinct)",
    520: "Ayaman (Jirajaran, no own entry; extinct)",
    530: "Timotocuica (Timote-Cuica, aes 6)", 539: "Timote (aes 6)",
    540: "Gayón (Jirajaran, aes 6)",
    714: "Píritu (no entry; extinct)",
    715: "Jirajara (Jirajaran, aes 6)",
    717: "Cumanagoto (aes 6)", 718: "Kumanagoto (aes 6)",
}

CODES = {
    # ---- Arawakan ----
    40: "arawakan.baniva",            # Baniva of Maroa (Glottolog Baniva de Maroa), not Brazil's
                                      # Baniwa of the Içana
    150: "arawakan.piapoco",          # Chase: the Piapoco (Dzase), INE's table 22 groups them
    151: "arawakan.piapoco",
    230: "arawakan.warekena",
    240: "arawakan.wayuu",            # Guajiro: the Spanish name for the Wayuu
    241: "arawakan.wayuu",
    350: "arawakan.wapixana",
    370: "arawakan.kuripako",         # Curripaco / Kurripako: spellings
    371: "arawakan.kuripako",
    900: "arawakan.lokono",           # Arawako: the Spanish name for the Lokono
    901: "arawakan.lokono",
    # ---- Cariban ----
    30: "cariban.akawaio",
    31: "cariban.akawaio",            # Kapón: the self-name the Akawaio share with the Patamona;
                                      # one person, filed with Akawayo by INE
    80: "cariban.enepa",              # Eñepa / Panare: endonym and exonym
    81: "cariban.enepa",
    110: "cariban.galibi_kali_na",    # Kariña = Galibi Carib (Kali'na)
    140: "cariban.pemon",             # Pemón without a sub-people, beside the three that are named
    141: "cariban.arekuna",
    142: "cariban.kamarakoto",
    143: "cariban.taurepang",
    270: "cariban.ye_kwana",          # Makiritare: the old exonym for the Ye'kwana
    271: "cariban.ye_kwana",
    280: "cariban.yukpa",
    281: "cariban.japreria",
    310: "cariban.makuxi",
    # ---- Guahiboan ----
    90: "guahiboan.sikuani",          # Guajibo, Sikwani, Jiwi: the Jivi (Hiwi) are the people
    92: "guahiboan.sikuani",          # Colombia calls Sikuani; one language
    99: "guahiboan.sikuani",
    91: "guahiboan.amorua",           # a separate label, Colombia's node
    181: "guahiboan.cuiba",           # Kuiva / Cuiba: spellings
    182: "guahiboan.cuiba",
    # ---- Saliban ----
    160: "saliban.piaroa",            # Wótüja: the Piaroa's own name
    161: "saliban.piaroa",
    162: "saliban.mako",
    # ---- Yanomaman ----
    190: "yanomaman.sanuma",          # Sanema / Sanüma: spellings
    191: "yanomaman.sanuma",
    250: "yanomaman.yanomami",
    251: "yanomaman.xiriana",         # Shiriana: the Ninam of the Paragua and Uraricoera, Brazil's
                                      # Xiriana; a separate census label (INE footnote 2)
    # ---- isolates (Glottolog: no family) ----
    100: "isolate.jodi",              # Hoti / Jodi: spellings
    101: "isolate.jodi",
    180: "isolate.pume",              # Pumé / Yaruro: endonym and exonym
    189: "isolate.pume",
    202: "isolate.warao",
    # ---- others ----
    60: "chibchan.bari",
    991: "chibchan.uwa",              # Tunebo: Colombia's U'wa (Tunebo)
    130: "tupian.tupiguarani.nheengatu",   # Ñengatú / Yeral
    131: "tupian.tupiguarani.nheengatu",
    170: "nadahup.puinave",
    290: "tucanoan.kubeo",
    300: "tucanoan.wanano",           # Guanano
    340: "tucanoan.tukano",
    320: "matacoan.wichi",            # Matako: the old name for the Wichí (Argentina's node)
    560: "quechuan.inga",
    570: "quechuan.quechua",          # Kechwa, 20 people, no country given
    # ---- remainders ----
    999: AM,                          # Otro: another pueblo, not named
    998: None,                        # No declarado: indigenous, pueblo not given; `gap`
    1001: SPANISH,                    # born in Venezuela, not indigenous (tier derived)
    1002: SPANISH,                    # born abroad, never asked (tier derived)
}
CODES.update({c: SPANISH for c in EXTINCT})


def resolve(code):
    return CODES[int(code)]
