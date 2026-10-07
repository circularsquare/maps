"""Mali RGPH5 2022, mother tongue ("langue maternelle") by région -> node.

Keyed by the labels of annex A06 of *Caractéristiques culturelles de la population*
(sources/ml_rgph.py), plus annex A03's six named foreign languages, which A06 prints by région
only as one row ("Autre langue étrangère") and the source script splits in their national
proportions. One answer per person aged 3 and over: the first language learnt as a child.

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv):
  * Bambara/Bamanankan: Bambara (bamb1269). Malinké/Maninkakan: Maninka, ci.txt's node; in
    Mali it is mostly Kita and Western Maninkakan (kita1249, west2500).
  * Sarakole/Sooninke: Soninke (soni1259). Khassonké/Xhassonkakan: Xaasongaxango (xaas1235).
  * Peulh/Fulfulde: Fula, the existing node (Maasina Fulfulde maas1239 and Pulaar pula1263 in
    Mali); not cf.txt's `peulh`, which holds a separate "Peulh" answer printed beside Fulfulde.
  * Sonrhai/Songhoy/Zarma: one answer for Koyraboro Senni, Koyra Chiini, Humburi Senni and Zarma
    (song1307's members in Mali): one leaf under a Songhay group.
  * Sénoufo/Syenara: ci.txt's Senufo leaf (Syenara Senoufo is syen1235, Supyire supy1237).
  * Minianka/Mamara: Mamara Senoufo (mama1271). It IS a Senufo language, but `senufo` is a leaf
    Côte d'Ivoire draws its whole Senufo answer on, so Mamara hangs beside it under Gur rather
    than under it (a child would turn ci.txt's answer into a group node).
  * Dogon/Dôgôsô: one answer for the Dogon cluster (dogo1299, about 20 languages).
  * Maure/Hasaniya: Hassaniyya Arabic (hass1238), beside Arabic as `judeo_arabic` is; `arabic`
    is a leaf many countries draw on. Arabe: Arabic.
  * Tamasheq: Tamasheq (tama1365) and Tawallammat Tamajaq (tawa1286), Tuareg, under Berber.
  * Bobo/Bomu: Bomu (bomu1247), the Bwamu language of Mali, Gur by the usual classification. The
    census's "Bobo" is the Bwa here: the Mande Bobo (Bobo Madaré) is the separate Kunabere.
  * Kunabere: Konabéré, Northern Bobo Madaré (nort2819), Mande. Not under that name in Glottolog;
    identified from the MPI numeral database ("Konabéré / Northern Bobo Madare") and its
    regions (San, Koutiala, on the Burkina border).
  * Dafing: Marka (mark1256), Mande, the Marka-Dafin of San and the Sourou.
  * Samogo/Dungooma: Duungooma (duun1242), Mande; "Samogo" is the Malian name for the Duun-Seenku
    peoples of Sikasso.
  * Bozo/Tyako: one answer for the Bozo languages (bozo1252: Jenaama, Tiéyaxo, Tiemacèwè,
    Hainyaxo), Mande.
  * Haoussa: Hausa. Mossi/Moré: Mòoré.

Remainders:
  * "Autre langue du Mali" (40,509 drawn; 10.3% in Ménaka, where Dawsahak is the obvious
    candidate but is not named) and "Autre langue africaine" (21,441, non-Malian African
    languages) both sit on `africa_other`: they are both other African languages, the
    narrowest node holding either, and neither is a foreign language on `other`.
  * "Autre langue non africaine" (14,453): `other`.
"""
MANDE = "nigercongo.mande"
GUR = "nigercongo.gur"

NAMES = {
    "Bambara/Bamanankan": f"{MANDE}.bambara",
    "Malinké/Maninkakan": f"{MANDE}.maninka",
    "Peulh/Fulfulde": "nigercongo.atlantic.fulah",
    "Sonrhai/Songhoy/Zarma": "nilosaharan.songhay.songhay",
    "Sarakole/Sooninke": f"{MANDE}.soninke",
    "Khassonké/Xhassonkakan": f"{MANDE}.khassonke",
    "Sénoufo/Syenara": f"{GUR}.senufo",
    "Dogon/Dôgôsô": "nigercongo.dogon.dogon",
    "Maure/Hasaniya": "afroasiatic.hassaniya",
    "Tamasheq": "afroasiatic.berber.tamasheq",
    "Bobo/Bomu": f"{GUR}.bomu",
    "Kunabere": f"{MANDE}.konabere",
    "Dafing": f"{MANDE}.marka",
    "Minianka/Mamara": f"{GUR}.mamara",
    "Haoussa": "afroasiatic.chadic.hausa",
    "Mossi/Moré": f"{GUR}.moore",
    "Samogo/Dungooma": f"{MANDE}.duungooma",
    "Bozo/Tyako": f"{MANDE}.bozo",
    "Arabe": "afroasiatic.arabic",
    "Autre langue du Mali": "africa_other",
    "Autre langue africaine": "africa_other",
    # A06's "Autre langue étrangère", split by A03's national counts (sources/ml_rgph.py)
    "Français": "indoeuropean.romance.french",
    "Anglais": "indoeuropean.germanic.english",
    "Allemand": "indoeuropean.germanic.continental.german",
    "Russe": "indoeuropean.slavic.east.russian",
    "Chinois": "sinotibetan.sinitic",
    "Espagnol": "indoeuropean.romance.spanish",
    "Autre langue non africaine": "other",
}

# carried in ml.csv's national rows (annex A03 as printed), never drawn
EXCLUDED = ("ND",)


def resolve(name):
    return NAMES.get(name)
