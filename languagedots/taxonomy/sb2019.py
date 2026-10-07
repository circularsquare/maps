"""Solomon Islands, 2019 census, first language learnt as a child (National Report Vol 1, Tables
9.6.1-9.6.3) -> node. Keyed by the labels sources/sb_census.py writes, which are the report's
own spellings.

EVERY LABEL THE REPORT PRINTS GETS A NODE, including the ones its commentary says are dialects,
lineage names or extinct varieties (Mae, Ghoighoi, Laghu, Dororo, Guliguli, Kazukuru): the
census counted people giving those answers, and the map draws what the source names (spec 3.1).
Node labels add the usual name where the census's differs:
  * Tairaha -> Bauro (Tairaha): Bauro's other name (Wikipedia, Bauro language);
  * Laube -> Lavukaleve (Laube): the report says Laube is another name for Lavukaleve. Only the
    36 who said "Laube" are here; most Lavukaleve speakers are in the remainder;
  * Asumnoa -> Asumboa, Baenggu -> Baegu, Mono -> Mono-Alu, Duke -> Duke (Nduke), Aiwoo ->
    Äiwoo, Tanibili -> Tanimbili, Zazao -> Zazao (Kilokaka): spellings or Glottolog's name;
  * RenBell -> Rennell-Bellona (9.6.1 prints the same 4,438 under that name).
Tauma is kept as printed; the report cannot verify it, and sources/sb_model.py anchors it on
Taumako in the Duff Islands, a Polynesian outlier, so it sits under Oceanic.

THE REMAINDER, 90,859 people (14.4% of those aged 5+), is everyone whose answer the report does
not print: English, "Other", and every local language too small or unremarkable for its three
tables (Tikopia, Natügu, Bughotu, Kokota, Savosavo, Luangiua, Fataleka...). It mixes local
languages with English and foreign ones and nothing published separates them, so by spec 3.2 it
sits on the narrowest node holding all of them, which is `other`. (The rule to keep indigenous
remainders off `other` applies where a census lets the two be told apart; this one does not.)
"""
NAMES = {
    "Pidgin": "creole.english_based.pijin",
    "Kiribati": "austronesian.oceanic.gilbertese",
    # Northwest Solomonic
    "Babatana": "austronesian.oceanic.nw_solomonic.babatana",
    "Varisi": "austronesian.oceanic.nw_solomonic.varisi",
    "Vaghua": "austronesian.oceanic.nw_solomonic.vaghua",
    "Ririo": "austronesian.oceanic.nw_solomonic.ririo",
    "Mono": "austronesian.oceanic.nw_solomonic.mono_alu",
    "Roviana": "austronesian.oceanic.nw_solomonic.roviana",
    "Marovo": "austronesian.oceanic.nw_solomonic.marovo",
    "Vangunu": "austronesian.oceanic.nw_solomonic.vangunu",
    "Ughele": "austronesian.oceanic.nw_solomonic.ughele",
    "Simbo": "austronesian.oceanic.nw_solomonic.simbo",
    "Lungga": "austronesian.oceanic.nw_solomonic.lungga",
    "Duke": "austronesian.oceanic.nw_solomonic.nduke",
    "Kazukuru": "austronesian.oceanic.nw_solomonic.kazukuru",
    "Dororo": "austronesian.oceanic.nw_solomonic.dororo",
    "Guliguli": "austronesian.oceanic.nw_solomonic.guliguli",
    "Cheke Holo": "austronesian.oceanic.nw_solomonic.cheke_holo",
    "Mae": "austronesian.oceanic.nw_solomonic.mae",
    "Ghoighoi": "austronesian.oceanic.nw_solomonic.ghoighoi",
    "Zazao": "austronesian.oceanic.nw_solomonic.zazao",
    "Laghu": "austronesian.oceanic.nw_solomonic.laghu",
    # Southeast Solomonic
    "Gela": "austronesian.oceanic.se_solomonic.gela",
    "Lengo": "austronesian.oceanic.se_solomonic.lengo",
    "Ghari": "austronesian.oceanic.se_solomonic.ghari",
    "Tolo/Talise": "austronesian.oceanic.se_solomonic.talise",
    "Birao": "austronesian.oceanic.se_solomonic.birao",
    "Kwara'ae": "austronesian.oceanic.se_solomonic.kwaraae",
    "Are'are": "austronesian.oceanic.se_solomonic.areare",
    "Lau": "austronesian.oceanic.se_solomonic.lau",
    "Kwaio": "austronesian.oceanic.se_solomonic.kwaio",
    "To'abaita": "austronesian.oceanic.se_solomonic.toabaita",
    "Baenggu": "austronesian.oceanic.se_solomonic.baegu",
    "Baelelea": "austronesian.oceanic.se_solomonic.baelelea",
    "Wala": "austronesian.oceanic.se_solomonic.wala",
    "Gula'alaa": "austronesian.oceanic.se_solomonic.gulaalaa",
    "Dori'o": "austronesian.oceanic.se_solomonic.dorio",
    "Sa'a": "austronesian.oceanic.se_solomonic.saa",
    "Ulawa": "austronesian.oceanic.se_solomonic.ulawa",
    "Arosi": "austronesian.oceanic.se_solomonic.arosi",
    "Tairaha": "austronesian.oceanic.se_solomonic.bauro",
    "Kahua": "austronesian.oceanic.se_solomonic.kahua",
    "Owa": "austronesian.oceanic.se_solomonic.owa",
    # Temotu
    "Aiwoo": "austronesian.oceanic.temotu.aiwoo",
    "Nalögo": "austronesian.oceanic.temotu.nalogo",
    "Engdewu": "austronesian.oceanic.temotu.engdewu",
    "Noipa": "austronesian.oceanic.temotu.noipa",
    "Amba": "austronesian.oceanic.temotu.amba",
    "Asumnoa": "austronesian.oceanic.temotu.asumboa",
    "Tanibili": "austronesian.oceanic.temotu.tanimbili",
    "Lovono": "austronesian.oceanic.temotu.lovono",
    "Tanema": "austronesian.oceanic.temotu.tanema",
    # Polynesian outliers
    "RenBell": "austronesian.oceanic.rennell_bellona",
    "Anuta": "austronesian.oceanic.anuta",
    "Sikaiana": "austronesian.oceanic.sikaiana",
    "Tauma": "austronesian.oceanic.tauma",
    # Central Solomons (non-Austronesian)
    "Bilua": "papuan.central_solomons.bilua",
    "Touo": "papuan.central_solomons.touo",
    "Laube": "papuan.central_solomons.lavukaleve",
    # unnamed: local languages the report does not list, English, other
    "Not named in the report (other local languages, English, other)": "other",
}
EXCLUDED = ()
EXTRA_NODES = []


def resolve(name):
    return NAMES[name]
