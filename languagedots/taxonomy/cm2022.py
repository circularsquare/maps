"""Cameroon, Afrobarometer R5-R9 (2013-2022), home language -> node.

Keyed by the answers sources/cm_afro.py writes to data/normalized/cm.csv: the card's labels
(CODED there; a Bamileke language the card names by its chief town is read as that town's
language) and the languages its "Other (specify)" free text names (VERBATIM there). The tree,
and the Glottolog check of every branch, is taxonomy/tree.d/cm.txt.

  * Fulfulde is us.txt's Fula leaf; Fulfulde, Foufouldé, Peulh, Foulbé and Mbororo answers.
  * Arabic answers (Arabe, Arabe Choa) are Shuwa Arabic, ng.txt's node: 20 of 21 from the
    Extrême-Nord, Nord and Adamaoua.
  * Guiziga is one answer for North and South Giziga; Kotoko one for the Kotoko languages.

Remainders:
  * "Bamileke (language not named)": the card's "Bamileke", and chiefdoms not placed on one
    Bamileke language. The Bamileke group node.
  * "Grassfields (language not named)": Bamenda, Lebialem, Widikum, Donga-Mantung's "Mbesah".
  * "Bantu (language not named)": "Sawa" (the coastal peoples) and "Mbamois" (Mbam).
  * "Other Cameroonian language": a language named in free text by one respondent, a word not
    identifiable as a language, or "Other" with no text. `africa_other`.
"""
BA = "nigercongo.bantu"
BD = "nigercongo.bantoid"
GF = f"{BD}.grassfields"
BM = f"{GF}.bamileke"
CH = "afroasiatic.chadic"
AD = "nigercongo.adamawa"

NAMES = {
    # lingua francas
    "French": "indoeuropean.romance.french",
    "English": "indoeuropean.germanic.english",
    "Cameroonian Pidgin": "creole.english_based.cameroonian_pidgin",
    "Fulfulde": "nigercongo.atlantic.fulah",
    # Bantu
    "Ewondo": f"{BA}.ewondo", "Eton": f"{BA}.eton", "Bene": f"{BA}.bene", "Bulu": f"{BA}.bulu",
    "Fang": f"{BA}.fang", "Ntumu": f"{BA}.ntumu", "Mvele": f"{BA}.mvele",
    "Basaa": f"{BA}.basaa", "Duala": f"{BA}.duala", "Batanga": f"{BA}.batanga",
    "Akoose": f"{BA}.akoose", "Mokpwe": f"{BA}.mokpwe", "Oroko": f"{BA}.oroko",
    "Bakundu": f"{BA}.bakundu", "Mbo": f"{BA}.mbo", "Bakoko": f"{BA}.bakoko", "Abo": f"{BA}.abo",
    "Bafia": f"{BA}.bafia", "Tunen": f"{BA}.tunen", "Yambassa": f"{BA}.yambassa",
    "Nomaande": f"{BA}.nomaande", "Makaa": f"{BA}.makaa", "Koonzime": f"{BA}.koonzime",
    "Bajwe'e": f"{BA}.bajwee", "Pol": f"{BA}.pol", "Kako": f"{BA}.kako",
    "Bantu (language not named)": BA,
    # Grassfields
    "Ghomala'": f"{BM}.ghomala", "Fe'fe'": f"{BM}.fefe", "Medumba": f"{BM}.medumba",
    "Yemba": f"{BM}.yemba", "Ngiemboon": f"{BM}.ngiemboon", "Ngombale": f"{BM}.ngombale",
    "Ngwe": f"{BM}.ngwe", "Mbouda": f"{BM}.mbouda",
    "Bamileke (language not named)": BM,
    "Bamun": f"{GF}.bamun", "Mungaka": f"{GF}.mungaka", "Bafanji": f"{GF}.bafanji",
    "Ngemba": f"{GF}.ngemba", "Mankon": f"{GF}.mankon", "Bafut": f"{GF}.bafut",
    "Limbum": f"{GF}.limbum", "Yamba": f"{GF}.yamba", "Lamnso'": f"{GF}.lamnso",
    "Kom": f"{GF}.kom", "Oku": f"{GF}.oku", "Aghem": f"{GF}.aghem", "Mmen": f"{GF}.mmen",
    "Meta'": f"{GF}.meta", "Moghamo": f"{GF}.moghamo", "Ngie": f"{GF}.ngie", "Ngwo": f"{GF}.ngwo",
    "Oshie": f"{GF}.oshie", "Mundani": f"{GF}.mundani", "Befang": f"{GF}.befang",
    "Grassfields (language not named)": GF,
    # other Bantoid
    "Kenyang": f"{BD}.kenyang", "Esimbi": f"{BD}.esimbi", "Tikar": f"{BD}.tikar",
    "Ncane": f"{BD}.ncane", "Noni": f"{BD}.noni", "Ejagham": f"{BD}.ejagham",
    # Chadic
    "Hausa": f"{CH}.hausa", "Mafa": f"{CH}.mafa", "Kapsiki": f"{CH}.kapsiki",
    "Massa": f"{CH}.massa", "Guiziga": f"{CH}.guiziga", "Kotoko": f"{CH}.kotoko",
    "Wandala": f"{CH}.wandala", "Musgum": f"{CH}.musgum", "Guidar": f"{CH}.guidar",
    "Jimi": f"{CH}.jimi", "Daba": f"{CH}.daba", "Gavar": f"{CH}.gavar", "Hina": f"{CH}.hina",
    "Mada": f"{CH}.mada_cm", "Podoko": f"{CH}.podoko", "Zulgo": f"{CH}.zulgo", "Lele": f"{CH}.lele",
    # Adamawa, Gbaya
    "Tupuri": f"{AD}.tupuri", "Mundang": f"{AD}.mundang", "Dii": f"{AD}.dii", "Fali": f"{AD}.fali",
    "Mubako": f"{AD}.mubako", "Mbum": f"{AD}.mbum", "Gbaya": "nigercongo.gbaya.gbaya",
    # Saharan, Arabic, outside Cameroon
    "Kanuri": "nilosaharan.kanuri", "Shuwa Arabic": "afroasiatic.arabic.shuwa",
    "Igbo": "nigercongo.voltaniger.igbo",
    "Other Cameroonian language": "africa_other",
}

EXTRA_NODES = []


def resolve(name):
    return NAMES.get(name)
