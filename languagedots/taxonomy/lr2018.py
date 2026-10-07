"""Liberia, Afrobarometer R4-R7 (2008-2018) first-language answers by county (sources/lr_afro.py)
-> node. Labels are the survey's language card (its ethnic names).

  * Mande: Kpelle, Lorma (Loma), Mano, Gio (Dan, dann1241's leaf), Vai, Mende, Gbandi (Bandi,
    band1352, new), Mandingo (the Maninka-speaking Mandingo of Lofa and Nimba: Maninka's leaf).
  * Kru: Bassa, Krahn (existing leaves); Grebo (Glottolog's Grebo family, gbol/nort/sout
    varieties the survey does not tell apart: one leaf), Kru (Klao, klao1243), Sarpo (Sapo,
    sapo1251), Dei (Dewoin, dewo1238), Belle (Kuwaa, kuwa1247): new leaves.
  * Atlantic: Kissi (existing), Gola (gola1255; unclassified Atlantic-Congo in Glottolog, Mel
    / Atlantic as conventionally grouped): new leaf.
  * "English", "Liberian English", "Simple Liberian English": Liberian English (creole leaf),
    only for respondents with no ethnic language (sources/lr_afro.py moves the rest).
  * "Other" (0.3%): africa_other.
"""
M = "nigercongo.mande"
K = "nigercongo.kru"

NAMES = {
    "Kpelle": f"{M}.kpelle",
    "Lorma": f"{M}.loma",
    "Mano": f"{M}.mano",
    "Gio": f"{M}.dan",
    "Vai": f"{M}.vai",
    "Mende": f"{M}.mende",
    "Gbandi": f"{M}.bandi",
    "Mandingo": f"{M}.maninka",
    "Bassa": f"{K}.bassa",
    "Krahn": f"{K}.krahn",
    "Grebo": f"{K}.grebo",
    "Kru": f"{K}.klao",
    "Sarpo": f"{K}.sapo",
    "Dei": f"{K}.dewoin",
    "Belle": f"{K}.kuwaa",
    "Kissi": "nigercongo.atlantic.kissi",
    "Gola": "nigercongo.atlantic.gola",
    "Liberian English": "creole.english_based.liberian",
    "Other": "africa_other",
}


def resolve(name):
    return NAMES.get(name)
