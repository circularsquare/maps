"""Malawi: the answers of sources/mw_census.py -> node.

The census asked tribe, not language; sources/mw_census.py turns each district's count of the
13 tribes into home languages with Afrobarometer's shares (R5, R7, R8, R9). The answers are the
survey's language names, Chi- forms. tree.d/mw.txt has the levels and glottocodes.

Calls:
  * Chichewa, Chinyanja and Chimang'anja are three nodes, as the survey names them, though
    Glottolog has one language (Nyanja) with Chewa and Mang'anja as dialects.
  * "Chingoni" is a Nyanja variety in the Central and Southern regions (zm.txt's `ngoni`) and a
    Tumbuka one in the north; mw_census.py relabels the northern answers "Chingoni (northern)".
  * Chilomwe and Chisena are Malawi Lomwe and Malawi Sena, separate from Mozambique's nodes.
  * Chinkhonde and Chinyakyusa are named apart by the survey (Glottolog: dialects of one
    language), so two nodes. Chinyika is Nyiha (Malawi), on zm.txt's Nyiha.
  * Chiwiza is Bisa (Ethnologue lists Wiza among Bisa's names); only round 4 met it, so it is
    not drawn at present, kept for completeness.
  * Sindebele (one respondent) on Zimbabwe's Ndebele. "Other" with no verbatim on `other`.
"""
B = "nigercongo.bantu"

NAMES = {
    "Chichewa": f"{B}.nyanja_sena.chewa",
    "Chinyanja": f"{B}.nyanja_sena.nyanja",
    "Chimang'anja": f"{B}.nyanja_sena.manganja",
    "Chingoni": f"{B}.nyanja_sena.ngoni",
    "Chingoni (northern)": f"{B}.tumbuka.ngoni_tumbuka",
    "Chisena": f"{B}.nyanja_sena.sena_mw",
    "Chinyungwe": f"{B}.nyanja_sena.nyungwe",
    "Chikunda": f"{B}.nyanja_sena.chikunda",
    "Chitumbuka": f"{B}.tumbuka.tumbuka",
    "Chitonga": f"{B}.tumbuka.tonga_mw",
    "Chisenga": f"{B}.tumbuka.senga",
    "Chiyao": f"{B}.yao",
    "Chilomwe": f"{B}.makhuwa.lomwe_mw",
    "Chikhokhola": f"{B}.makhuwa.kokola",
    "Chinkhonde": f"{B}.nkhonde",
    "Chinyakyusa": f"{B}.nyakyusa",
    "Chindali": f"{B}.ndali",
    "Chisukwa": f"{B}.sukwa",
    "Chilambya": f"{B}.mambwe_nyiha.lambya",
    "Chinyika": f"{B}.mambwe_nyiha.nyiha",
    "Chinamwanga": f"{B}.mambwe_nyiha.namwanga",
    "Chiwandya": f"{B}.mambwe_nyiha.wandya",
    "Chimambwe": f"{B}.mambwe_nyiha.mambwe",
    "Chiwiza": f"{B}.bisa_lamba.bisa",
    "Swahili": f"{B}.swahili",
    "Sindebele": f"{B}.nguni.ndebele_zw",
    "English": "indoeuropean.germanic.english",
    "Portuguese": "indoeuropean.romance.portuguese",
    "Other": "other",
}


def resolve(name):
    return NAMES.get(name)
