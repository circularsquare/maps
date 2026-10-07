"""Peru, Censo 2017, variable C5P11 "Idioma o lengua con el que aprendio hablar" (mother tongue,
age 3 and over): census label -> node. Labels exactly as sources/pe_censo.py writes them into
data/normalized/pe.csv.

The first 14 labels are the form's precoded boxes; the 31 after "No escucha, ni habla" are INEI's
coding of the "other native language, specify" write-ins, so every one is a language the census
names and gets a node (spec 3.1). Same-language reuses of other fragments' nodes:
  Cashinahua -> panoan.kaxinawa      Glottolog cash1254, Brazil's Kaxinawa
  Yaminahua -> panoan.yaminawa       yami1256
  Matses -> panoan.matses            mats1244
  Kukama kukamiria -> ...kokama      coca1259, Cocama-Cocamilla
  Murui-Muinani -> witotoan.witoto   Murui and Minica Huitoto, the Witoto spoken in Peru; Brazil's
                                     Witoto is the same people's language across the border
Remainders:
  "Otra lengua nativa u originaria"  americas_other: native languages INEI left uncoded (1,060
                                     people), any family.
  "Otra lengua extranjera"           other.
Not drawn (gap): "No escucha, ni habla" (does not hear or speak), "No sabe / No responde".
"""

NAMES = {
    "Castellano": "indoeuropean.romance.spanish",
    "Portugués": "indoeuropean.romance.portuguese",
    "Otra lengua extranjera": "other",
    "Lengua de señas peruanas": "signlanguage.lsp",
    "Otra lengua nativa u originaria": "americas_other",
    # Quechuan, Aymaran
    "Quechua": "quechuan.quechua",
    "Kichwa": "quechuan.kichwa",
    "Aimara": "aymaran.aymara",
    "Jaqaru": "aymaran.jaqaru",
    "Cauqui": "aymaran.cauqui",
    # Arawakan
    "Ashaninka": "arawakan.ashaninka",
    "Matsigenka/Machiguenga": "arawakan.matsigenka",
    "Nomatsigenga": "arawakan.nomatsigenga",
    "Kakinte": "arawakan.kakinte",
    "Yine": "arawakan.yine",
    "Yanesha": "arawakan.yanesha",
    # Chicham (Jivaroan)
    "Awajún / Aguaruna": "chicham.awajun",
    "Wampis": "chicham.wampis",
    "Achuar": "chicham.achuar",
    # Cahuapanan
    "Shawi/Chayahuita": "cahuapanan.shawi",
    "Shiwilu": "cahuapanan.shiwilu",
    # Panoan
    "Shipibo - Konibo": "panoan.shipibo",
    "Kakataibo": "panoan.kakataibo",
    "Matses": "panoan.matses",
    "Yaminahua": "panoan.yaminawa",
    "Amahuaca": "panoan.amahuaca",
    "Nahua": "panoan.nahua",
    "Capanahua": "panoan.capanahua",
    "Sharanahua": "panoan.sharanahua",
    "Cashinahua": "panoan.kaxinawa",
    "Isconahua": "panoan.isconahua",
    # Tacanan, Harakmbut, Zaparoan
    "Ese Eja": "tacanan.ese_eja",
    "Harakbut": "harakmbut.harakbut",
    "Arabela": "zaparoan.arabela",
    # Tupian, Tucanoan, Peba-Yaguan, Witotoan
    "Kukama kukamiria": "tupian.tupiguarani.kokama",
    "Omagua": "tupian.tupiguarani.omagua",
    "Secoya": "tucanoan.secoya",
    "Maijuna": "tucanoan.maijuna",
    "Yagua": "pebayaguan.yagua",
    "Murui-Muinani": "witotoan.witoto",
    "Ocaina": "witotoan.ocaina",
    # isolates
    "Tikuna": "isolate.tikuna",
    "Urarina": "isolate.urarina",
    "Kandozi-Chapra": "isolate.kandozi",
    # not drawn
    "No escucha, ni habla": None,
    "No sabe / No responde": None,
}


def resolve(label):
    return NAMES[label]
