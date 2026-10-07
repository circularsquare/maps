"""Gambia, 2013 census ethnic group by LGA read as language through Afrobarometer R7 mother
tongues (sources/gm_census.py) -> node. Labels are the survey's language answers (and, for
groups kept whole, the group's own language).

  * Mandinka, Jahanka (Jahanka is printed with Mandinka by the census; the survey's
    Mandinka/Jahanka respondents split them), Soninke ("Serahuleh"), Bambara: Mande leaves.
  * Fula (the census's Fula/Tukulor/Lorobo; Pulaar), Wolof, Serer, Manjak ("Manjago"):
    Atlantic leaves. "Jola" (the census's Jola/Karoninka; the survey names only Jola, and
    Gambian Jola is Jola-Fonyi) on the Jola (Joola) leaf, not the Jola group.
  * Krio ("Creole/Aku Marabou", kept whole): the Krio leaf.
  * "Other" (census group and survey answer): africa_other.
"""
NAMES = {
    "Mandinka": "nigercongo.mande.mandinka",
    "Jahanka": "nigercongo.mande.jahanka",
    "Serahuleh": "nigercongo.mande.soninke",
    "Bambara": "nigercongo.mande.bambara",
    "Fula": "nigercongo.atlantic.fulah",
    "Wolof": "nigercongo.atlantic.wolof",
    "Serer": "nigercongo.atlantic.serer",
    "Manjago": "nigercongo.atlantic.manjak",
    "Jola": "nigercongo.atlantic.jola.joola",
    "Krio (Aku)": "creole.english_based.krio",
    "Other": "africa_other",
}


def resolve(name):
    return NAMES.get(name)
