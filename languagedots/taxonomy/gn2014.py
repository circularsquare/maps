"""Guinea RGPH 2014, main national language ("langue nationale habituellement parlée") by région
-> node.

Keyed by the row labels of Tableau 5.08 (sources/gn_rgph.py). One answer, people aged 3+;
the answers are Guinea's national languages, plus "Aucune" and "Autre langue nationale".

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv):
  * Poular: Fula (Pular, pula1262, is Guinea's Fula), the existing node.
  * Soussou: Susu (susu1250). Maninka: Eastern Maninkakan (east2426), the existing Maninka node
    (ci, ml). Koniaka: Konyanka Maninka (kony1250), a Manding language of Beyla, its own node as
    the census prints it apart.
  * Diakanka: Jahanka (jaha1245). Djalonké: Yalunka (yalu1240). Sarakolé/Maraka: Soninke
    (soni1259). Kouranko: Kuranko (kura1250). Lélé: Lele (lele1266). Mikiforè: Mixifore
    (mixi1241). All Mande.
  * Toma: Loma (toma1245, "Toma" in Guinea, "Loma" in Liberia), au.txt's `loma` node.
    Tomamania: Manya (many1261), the Toma-Manian of Macenta, a Manding language; not Loma.
  * Kpèlè: Kpelle (Guinea Kpelle, guin1254). Mano: Mano (mann1248). Kono: Kono of Guinea
    (kono1267), a node of its own (`kono_guinea`) so Sierra Leone's unrelated Kono can have its
    own.
  * Kissi (kiss1245), Baga (the Baga languages, Glottolog's Northern Mel), Landouma (land1256),
    Nalou (nalu1240): Atlantic. Badiaranké: Jaad-Badyara (bady1239), sn.txt's `jaad`. Bassari
    (bass1258) and Koniagui (Wamey, wame1240): sn.txt's Tenda nodes.

Remainders:
  * "Autre langue nationale" (0.4%): another Guinean language the table does not print, so
    `africa_other` (an indigenous remainder, kept off `other`).
  * "Aucune" (0.2%): the person usually speaks no national language (French or a foreign
    language; the census does not say which). `other`.
"""
AT = "nigercongo.atlantic"
MANDE = "nigercongo.mande"

NAMES = {
    "Soussou": f"{MANDE}.susu",
    "Poular": f"{AT}.fulah",
    "Maninka": f"{MANDE}.maninka",
    "Diakanka": f"{MANDE}.jahanka",
    "Baga": f"{AT}.baga",
    "Nalou": f"{AT}.nalu",
    "Mikiforè": f"{MANDE}.mixifore",
    "Landouma": f"{AT}.landuma",
    "Badiaranké": f"{AT}.jaad",
    "Bassari": f"{AT}.tenda.bassari",
    "Koniagui": f"{AT}.tenda.wamey",
    "Djalonké": f"{MANDE}.yalunka",
    "Sarakolé/Maraka": f"{MANDE}.soninke",
    "Kouranko": f"{MANDE}.kuranko",
    "Kissi": f"{AT}.kissi",
    "Lélé": f"{MANDE}.lele",
    "Toma": f"{MANDE}.loma",
    "Koniaka": f"{MANDE}.konyanka",
    "Tomamania": f"{MANDE}.manya",
    "Kpèlè": f"{MANDE}.kpelle",
    "Mano": f"{MANDE}.mano",
    "Kono": f"{MANDE}.kono_guinea",
    "Autre langue nationale": "africa_other",
    "Aucune": "other",
}


def resolve(name):
    return NAMES.get(name)
