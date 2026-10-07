"""Argentina, Censo 2022: "habla y/o entiende la lengua de ese pueblo indigena u originario"
-> node, keyed by INDEC's P23 pueblo code (sources/ar_censo.py).

THE CENSUS NAMES A PEOPLE, NOT A LANGUAGE. A yes to P24 means the person speaks or understands
the language of the pueblo they gave in P23, so each pueblo maps to that people's language, as
in Colombia (taxonomy/co2018.py): one node per pueblo, labelled with the people's name where no
established language name exists. Families from Glottolog, checked per language
(data/raw/glottolog/languages.csv).

INDEC'S OWN SUBDIVISIONS. P23 codes some pueblos as subdivisions of another: 141-150 under
Diaguita, 171-173 under Guarani, 241-243 under Kolla, 281-285 under Mapuche (140/141, 170/171,
240/241 and 280/281 are the same label twice). A subdivision is drawn on its pueblo's node,
because the subdivision names a community, not a language: Diaguita Amaicha, Calchaqui,
Ingamana, Quilmes, Tolombon, Colastine, Capayan and Cacano on Diaguita; Kolla Diaguita on
Kolla; Pehuenche, Picunche and Puelche on Mapuche (Glottolog lists Pehuenche and Picunche as
Mapudungun dialects, and INDEC files this Puelche among the Mapuche, apart from 200 Gunun A
Kuna, the Puelche language of Patagonia). Exceptions, where the subdivision is or names a
different language: Huiliche (282) is Huilliche, its own Araucanian language in Glottolog;
Diaguita Quechua (144) and Kolla Quechua (243) name Quechua and sit under Quechuan; Ava Guarani
(172) is Eastern Bolivian Guarani (Chiriguano) and Tupi Guarani (173) its own node beside it.
Aoniken (30) is the Tehuelche people's own name for themselves and their language
(Aonikenk), so it is drawn with Tehuelche (430), the one merge of two separate codes.

GUARANI (171) DEPENDS ON PLACE. In Salta and Jujuy the Guarani pueblo is the Chiriguano (Ava)
of the Bermejo and San Francisco valleys, whose language is Eastern Bolivian Guarani: 7,568
and 1,439 speakers there sit beside 3,141 and 740 Ava Guarani. Everywhere else, 38,563
speakers, 22,709 of them in Buenos Aires province, 4,017 in the city, 3,899 in Corrientes and
3,731 in Misiones, it is Paraguayan Guarani, the language of Paraguay and of Corrientes and
Misiones (and of most of Argentina's Paraguayan-born). So 171 is split by province in resolve(), as India's Pahari is by state.

THE PEOPLES WHOSE LANGUAGE IS NO LONGER SPOKEN, or was never written down. Many answered yes:
Diaguita 14,947 (all subdivisions), Omaguaca 4,954, Tonokote 4,834, Huarpe 2,034, Atacama
1,828, Tehuelche 1,598, Comechingon 1,369, Ranquel 980, Charrua 775 and a few hundred each
across a dozen more. They are drawn as the census recorded them, each on a node of its own
under its Glottolog family where Glottolog classifies the language (Atacama = Kunza, isolate;
Huarpe = Huarpean; Charrua, Chana = Charruan; Avipon = Abipon, Guaicuruan; Lule, Vilela,
isolates; Chane = Chane, which Glottolog files as a dialect of Terena, Arawakan), otherwise
`unclassified` (Diaguita, whose language Glottolog lists as Calchaqui, unclassifiable;
Comechingon, Sanaviron, Tonocote, Querandi; the Quebrada de Humahuaca and Puna peoples
Omaguaca, Ocloya, Tilian, Tastil, Toara, Fiscara, Chicha, Churumata and Jujuies; Corundi,
Iogys, Wayteca). They are most likely heritage or revival speakers, or speakers of another
indigenous language: in Santiago del Estero (Tonokote 4,464 of 4,834, Diaguita Cacano 1,434,
Lule Vilela, Vilela, Sanaviron) that is very probably Santiago del Estero Quichua, but the
census does not say so and the map does not either. sources/ar.md and note_public say so.

Kolla (241): the Kolla of the Jujuy and Salta Puna and valleys; their language in Argentina
is Quechua (South Bolivian Quechua, the language of the Puna), and the node sits under
Quechuan as "Kolla", the census's word, not "Quechua". Compound pueblos that name two
peoples with different languages, Kolla Atacameno (250) and Lule Vilela (270), sit under
`unclassified` on nodes of their own; Mapuche Tehuelche (290) under Araucanian, since the
Mapuche-Tehuelche communities of Chubut and Rio Negro speak Mapudungun (Tehuelche has a
handful of speakers).

Remainders (spec §3.2): 990 "Sin informacion" (self-identified indigenous, speaks the
language of their pueblo, pueblo not recorded; 82,156) on `americas_other`. 190 Guaycuru
(159), a family name and not one people, on `guaicuruan` itself, as Colombia's "Maku".
Not drawn: code 2, indigenous people with no answer on P24 (155,542), in `gap`.
Codes 0 (not indigenous), 1 (indigenous, does not speak their people's language) and 3
(collective dwellings and the street, not asked) are Spanish, tier `derived` (spec §3.5).
"""

SPANISH = "indoeuropean.romance.spanish"
AM = "americas_other"
GN = "tupian.tupiguarani.guarani"
PARAGUAYAN = f"{GN}.paraguayan"
CHIRIGUANO = f"{GN}.chiriguano"
NOA_GUARANI = {"66", "38"}          # Salta, Jujuy: provincia part of the departamento code

CODES = {
    0: SPANISH,
    1: SPANISH,
    2: None,                                   # P24 ignorado: not drawn, `gap`
    3: SPANISH,
    10: "isolate.kunza",                       # Atacama
    20: "unclassified",                        # Alakaluf (Kawesqar): no speakers in 2022
    30: "chonan.tehuelche",                    # Aoniken = Aonikenk, the Tehuelche's own name
    40: "guaicuruan.abipon",                   # Avipon
    50: "aymaran.aymara",
    60: "charruan.chana",
    70: "arawakan.chane",
    80: "charruan.charrua",
    90: "unclassified.chicha",
    100: "matacoan.chorote",
    110: "matacoan.nivacle",
    120: "unclassified.comechingon",
    130: "unclassified.corundi",
    140: "unclassified.diaguita",
    141: "unclassified.diaguita",
    142: "unclassified.diaguita",              # Diaguita Amaicha
    143: "unclassified.diaguita",              # Diaguita Calchaqui
    144: "quechuan.diaguita_quechua",
    145: "unclassified.diaguita",              # Diaguita Ingamana
    146: "unclassified.diaguita",              # Diaguita Quilmes
    147: "unclassified.diaguita",              # Diaguita Tolombon
    148: "unclassified.diaguita",              # Diaguita Colastine
    149: "unclassified.diaguita",              # Diaguita Capayan
    150: "unclassified.diaguita",              # Diaguita Cacano
    160: "unclassified.fiscara",
    170: None,                                 # Guarani: by province, see resolve()
    171: None,
    172: CHIRIGUANO,                           # Ava Guarani
    173: f"{GN}.tupi_guarani",
    180: "tupian.tupiguarani.guarayu",
    190: "guaicuruan",                         # Guaycuru: the family's name, no one language
    200: "isolate.gununa_kune",                # Gunun A Kuna
    210: "huarpean.huarpe",
    220: "unclassified.iogys",
    230: CHIRIGUANO,                           # Isoceno: Glottolog's Izoceno dialect of it
    240: "quechuan.kolla",
    241: "quechuan.kolla",
    242: "quechuan.kolla",                     # Kolla Diaguita
    243: "quechuan.kolla_quechua",
    250: "unclassified.kolla_atacameno",
    260: "isolate.lule",
    270: "unclassified.lule_vilela",
    280: "araucanian.mapuche",
    281: "araucanian.mapuche",
    282: "araucanian.huilliche",               # Huiliche
    283: "araucanian.mapuche",                 # Pehuenche
    284: "araucanian.mapuche",                 # Picunche
    285: "araucanian.mapuche",                 # Puelche, as INDEC files it, among the Mapuche
    290: "araucanian.mapuche_tehuelche",
    300: f"{GN}.mbya",
    310: "guaicuruan.mocovi",
    320: "unclassified.ocloya",
    330: "unclassified.omaguaca",
    340: "guaicuruan.pilaga",
    350: "guaicuruan.toba",
    360: "quechuan.quechua",
    370: "unclassified.querandi",
    380: "araucanian.ranquel",
    390: "unclassified.sanaviron",
    400: "chonan.selknam",
    410: f"{GN}.tapiete",
    420: "unclassified.tastil",
    430: "chonan.tehuelche",
    440: "unclassified.tilian",
    450: "unclassified.toara",
    460: "unclassified.tonokote",
    470: "isolate.vilela",
    480: "matacoan.weenhayek",
    490: "matacoan.wichi",
    500: "isolate.yagan",
    510: "unclassified.wayteca",
    520: "chonan.haush",
    530: "matacoan.maka",
    540: "charruan.minuan",
    550: "unclassified",                       # Ansilta: no speakers in 2022
    560: "unclassified.churumata",
    570: "unclassified.jujuies",
    580: "unclassified",                       # Michilingue: no speakers in 2022
    970: AM,                                   # No codificables: no speakers in 2022
    980: AM,                                   # No corresponde: no speakers in 2022
    990: AM,                                   # Sin informacion
}

EXTRA_NODES = [PARAGUAYAN]


def resolve(code, geo_id=None):
    """Node for a pueblo code; `geo_id` is the five-digit departamento code."""
    code = int(code)
    if code in (170, 171):
        if geo_id is None:
            raise ValueError("Guarani (171) needs the departamento to resolve")
        return CHIRIGUANO if str(geo_id)[:2] in NOA_GUARANI else PARAGUAYAN
    return CODES[code]
