"""Guinea-Bissau RGPH 2009, principal ethnic language ("principal dialecto falado") -> node.

Keyed by the column labels of Anexo Quadro 4 (sources/gw_rgph.py). One write-in answer per
person, coded. The census defines a "dialecto" as the language of an etnia; Kriol (Crioulo),
Portuguese and foreign languages were asked separately as languages spoken (yes/no each).

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv):
  * Balanta: Balanta (Glottolog's Balanta family: Balanta-Kentohe bala1301, the Balanta of
    Guinea-Bissau), sn.txt's leaf.
  * Balanta Mane: the Balanta Mané, a Balanta people who took up Islam and much of Mandinka
    culture. The census prints their "dialecto" apart from Balanta, so a node of its own, a
    sibling of Balanta. No glottocode (Glottolog does not list them apart).
  * Manjaco, Mancanha: sn.txt's Manjak and Mankanya (mank1251). Papel: Papel (pape1239), the
    third of the Manjaku-Mankanya-Pepel group (manj1250), flat under Atlantic as the other two.
  * Felupe: Ejamat (ejam1238), the Jola language the Felupe speak, under sn.txt's Jola group.
  * Bijagos: the Bijagó languages (Glottolog splits Kamona Bijogo kamo1256 and Kanyaki-
    Kagbaaga-Kajoko Bidyogo bidy1244); one census answer, one leaf.
  * Beafada: Biafada (biaf1240). Mansoanca: Mansoanka (mans1259). Nalu: Nalu (nalu1240), gn's
    leaf. All Atlantic.
  * Fula: Fula (sn/gn's node). Mandinga: Mandinka (mand1436). Sosso: Susu (susu1250).
    Saracule: Soninke (soni1259). Mande, nodes already on the tree.

"Sem dialecto": the person named no ethnic language as their principal one. The census
filed there whoever's main language is Kriol, Portuguese or anything foreign (Kriol, by the
P.16 answers and the volume's prose, for nearly all of them). No node holds Kriol and
Portuguese together, so the narrowest honest place is a leaf of its own under `other`, labelled
for what the census says; the record (sources/gw.md) says why it is not drawn as Kriol.

"NA" (no answer recorded, mostly infants) is not drawn; it is not in gw.csv.
"""
AT = "nigercongo.atlantic"
MANDE = "nigercongo.mande"

NAMES = {
    "Sem dialecto": "other.no_ethnic_language",
    "Balanta": f"{AT}.balanta",
    "Balanta Mane": f"{AT}.balanta_mane",
    "Fula": f"{AT}.fulah",
    "Mancanha": f"{AT}.mankanya",
    "Mandinga": f"{MANDE}.mandinka",
    "Manjaco": f"{AT}.manjak",
    "Bijagos": f"{AT}.bijago",
    "Papel": f"{AT}.papel",
    "Beafada": f"{AT}.biafada",
    "Felupe": f"{AT}.jola.ejamat",
    "Mansoanca": f"{AT}.mansoanka",
    "Nalu": f"{AT}.nalu",
    "Sosso": f"{MANDE}.susu",
    "Saracule": f"{MANDE}.soninke",
}


def resolve(name):
    return NAMES.get(name)
