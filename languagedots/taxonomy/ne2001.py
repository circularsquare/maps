"""Niger, RGP/H 2001 ethnic group by département read as language, moved by Afrobarometer R5-R6
retention (sources/ne_census.py) -> node.

The census groups are ethnolinguistic; each is drawn as its language, and the share of each group
that named another language at home in the survey is moved onto that language.

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv):
  * Hausa (haus1257): the existing leaf.
  * Zarma: the census's "Djerma-Sonrai" and the survey's "Zarma/Songhay". Mostly Zarma-Kaado
    (zarm1239, with Dendi as its dialect in Gaya); the Songhay of Ayorou and the north-west of
    Tillabéri speak Koyraboro Senni (koyr1242), which neither source tells apart. On ng.txt's
    `songhay.zarma` leaf, the language of most of the group (sources/ne.md).
  * Tamasheq: the survey's "Tamasheq" and the census's "Touareg". Niger's Tuareg speak
    Tawallammat Tamajaq (tawa1286, the west and Tahoua) and Tayart Tamajeq (taya1257, Aïr and
    Agadez), separate Glottolog languages from Mali's Tamasheq. One node `berber.tamajaq`,
    "Tamajaq (Tuareg)", for both: neither source says which variety.
  * Fulfulde: Fula, the existing leaf (the Peulh group).
  * Kanuri: the census's "Kanouri-Manga". Manga Kanuri (mang1399) in Zinder and western Diffa,
    Central Kanuri (cent2050) and Tumari (tuma1248) around N'guigmi; on ng.txt's `kanuri` leaf.
  * Tubu (Toubou): Dazaga (daza1242) in Diffa and Zinder, Tedaga (teda1241) in Kawar; Saharan,
    a leaf `nilosaharan.tubu` under Nilo-Saharan as most readers know it, as sd.txt's Zaghawa.
  * Gourmanchéma (gour1243): bf.txt's leaf (the Gourma group, 88% of it in Tillabéri).
  * Arabic, split by place in sources/ne_census.py: Diffa and Zinder -> Shuwa (Chadian Arabic,
    chad1249), ng.txt's `arabic.shuwa`; elsewhere -> Hassaniya (hass1238), ml.txt's leaf.
  * French: one R5 answer, spread by the shrinkage to 6,611 people. Kept as answered.
"""
AA = "afroasiatic"
NS = "nilosaharan"

NAMES = {
    "Hausa": f"{AA}.chadic.hausa",
    "Zarma": f"{NS}.songhay.zarma",
    "Tamasheq": f"{AA}.berber.tamajaq",
    "Fulfulde": "nigercongo.atlantic.fulah",
    "Kanuri": f"{NS}.kanuri",
    "Tubu": f"{NS}.tubu",
    "Gourmanchéma": "nigercongo.gur.gourmanchema",
    "Arabic (Shuwa)": f"{AA}.arabic.shuwa",
    "Arabic (Hassaniya)": f"{AA}.hassaniya",
    "French": "indoeuropean.romance.french",
}


def resolve(name):
    return NAMES.get(name)
