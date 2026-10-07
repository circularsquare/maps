"""Senegal RGPH-5 2023, main language ("principale langue couramment parlée") by région -> node.

Keyed by the labels of chapter 1's Tableau I-32, which sources/sn_rgph.py folds the fourteen
regional reports' spellings onto. Question B18, "première langue la plus souvent parlée": the
language each resident aged 3 and over speaks most often, one answer.

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv):
  * Wolof (nucl1347). Pulaar: Fula, the existing node (Pulaar pula1263 is Senegal's Fula).
  * Sereer: Serer (Serer-Sine). The Cangin languages (Noon, Laalaa, Saafi, Ndut, Paloor), which
    the census does not print, are most likely inside Sereer or "Autres langues africaines";
    not split out.
  * Joola: the Jola cluster (jola1264: Jola-Fonyi, Kasa, Karon, ...), a leaf under a Jola group,
    with Bayot (bayo1262) beside it.
  * Màndienka: Mandinka (mand1436). Sóninke: Soninke (soni1259). Hasaniya (Maure): Hassaniyya
    (hass1238), ml.txt's node.
  * Balante: Balanta (bala1300; Balanta-Ganja bala1302 in Senegal). Mànjaku: Manjak (mand1419).
    Mànkaañ: Mankanya (mank1251).
  * Oniyan: Bassari (bass1258). Mënik: Bedik (bedi1235). Womey: Wamey/Konyagi (wame1240). The
    three Tenda languages, mostly in Kédougou and Tambacounda.
  * Kanjad: Jaad-Badyara (bady1239), 90% in Kolda. Guñuun: Bainouk-Gunyuño, the Bainouk of
    Ziguinchor and Kolda (Glottolog has Bainouk-Gunyaamolo bain1261 and Bainouk-Samik bain1260;
    Gunyuño's own code is not looked up here, so none is given).
  * Jalunga: Yalunka (yalu1240), Mande, 87% in Kédougou.
  * Tourka (Sénégal): printed among the national languages, not identified (sources/sn.md).
    `other.tourka`, a node of its own.
  * Langage des signes (Sourd-Muet): `signlanguage`. Français: French.

Remainders:
  * "Autres langues africaines" (201,613; 31.5% of Kédougou, 5.3% of Tambacounda): `africa_other`.
    It holds both Senegalese languages the table does not print and languages of other African
    countries; the census does not say which.
  * "Langues étrangères" (15,937) and "Autres langues étrangères non africaines" (30,800): both
    on `other`. The first is printed beside, not inside, the African remainder; neither names a
    language.
"""
AT = "nigercongo.atlantic"
MANDE = "nigercongo.mande"

NAMES = {
    "Wolof": f"{AT}.wolof",
    "Pulaar": f"{AT}.fulah",
    "Sereer": f"{AT}.serer",
    "Joola": f"{AT}.jola.joola",
    "Màndienka": f"{MANDE}.mandinka",
    "Sóninke": f"{MANDE}.soninke",
    "Hasaniya (Maure)": "afroasiatic.hassaniya",
    "Balante": f"{AT}.balanta",
    "Mànkaañ": f"{AT}.mankanya",
    "Mànjaku": f"{AT}.manjak",
    "Mënik": f"{AT}.tenda.bedik",
    "Oniyan": f"{AT}.tenda.bassari",
    "Guñuun": f"{AT}.gunyuno",
    "Kanjad": f"{AT}.jaad",
    "Jalunga": f"{MANDE}.yalunka",
    "Bayot": f"{AT}.jola.bayot",
    "Womey": f"{AT}.tenda.wamey",
    "Tourka (Sénégal)": "other.tourka",
    "Langage des signes (Sourd-Muet)": "signlanguage",
    "Français": "indoeuropean.romance.french",
    "Autres langues africaines": "africa_other",
    "Langues étrangères": "other",
    "Autres langues étrangères non africaines": "other",
}


def resolve(name):
    return NAMES.get(name)
