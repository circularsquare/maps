"""Burkina Faso RGPH 2006, main language spoken ("principale langue parlée") by province -> node.

Keyed by the labels of Tableau A5.3 of *Thème 2: État et structure de la population*
(sources/bf_rgph.py), plus Tableau A5.2's ten foreign languages, which A5.3 prints only as two
groups ("Langues Africaines", "Langues Non Africaines") and the source script splits by région.
One answer per resident aged 3 and over: the language the person mainly speaks.

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv; tree.d/bf.txt has
the levels):
  * Mooré: Mossi (moss1236). Goulmancema: Gourmanchéma (gour1243). Fulfuldé (A5.2: "ou
    Peulh"): Fula, the existing node. Dioula (A5.2: "ou Bambara"): Dyula (dyul1238), ci.txt's
    node; Bambara is not printed apart, and in Burkina Faso the answer is the Jula of Bobo-
    Dioulasso and the towns.
  * Bissa: Bissa (biss1248), Mande. Bobo: Bobo Madaré, mostly Southern (sout2840; Houet
    140,269 of 181,373), Mande; not Bwamu, which is its own row.
  * Bwamu (A5.2: "ou Bwamou"): the Bwamu cluster (bwam1247: Buamu, Cwi, Láá Láá, Bomu), Gur.
    A sibling of ml.txt's Bomu leaf, which holds Mali's own "Bobo/Bomu" answer.
  * Dafing: Marka (mark1256), ml.txt's node. Minianka: Mamara (mama1271), ml.txt's node.
  * Dagara: Northern Dagara (nort2780), Gur; the south-west, Ioba 154,422.
  * Dogon (A5.2: "ou Kaado"): Dogon, ml.txt's node. Kaado is the Songhay name for the Dogon.
  * Gouin: Cerma (cerm1238), Gur; Comoé 50,050 of 51,908.
  * Kasséna: Kasem (kase1253), Gur; Nahouri 80,741.
  * Ko: Winyé (winy1241, also called Kõ or Kols), Gur; Balé 7,205, where Winyé is spoken.
  * Koussassé: Kusaal (kusa1250), Gur; Boulgou 12,186.
  * Lyélé: Lyélé (lyel1241). Nuni (A5.2: "Nounouma"): Nuni (nuni1253, Northern and Southern).
  * Lobiri: Lobi (lobi1245), ci.txt's node.
  * San (A5.2: "ou Samogho ou Samo"): the Samo languages, Matya Samo (maty1235) and Maya Samo
    (maya1281), Mande; Sourou 99,407, Nayala 68,636.
  * Sembla: Seeku (seek1238), Mande; Houet 13,594.
  * Sénoufo: Senufo, ci.txt's node (Kénédougou, Léraba, Comoé).
  * Siamou: Sɛmɛ (siam1242), an isolate in Glottolog; Kénédougou 16,631.
  * Sissaka: read as Sisaala (sisa1248), Gur: no language of that name exists, and 207 of the
    332 are in Sissili, the Sisaala country on the Ghana border. A misspelling; its own node.
  * Sonrhaï: Songhay, ml.txt's node (Oudalan, Séno, Soum). Djerma (A5.2): Zarma, a leaf beside
    it, as the census prints it apart.
  * Tamachèque (A5.2: "ou Bella"): Tamasheq (tama1365), ml.txt's node. Bella is the name of a
    Tamasheq-speaking group, not another language.
  * Gurunsi: the Grusi peoples (grus1239) named without the language. Kasem, Nuni, Lyélé and
    Sisaala are printed apart, so this is a leaf of its own, `gurunsi`, labelled as not naming
    the language; not a group node over the four, which would draw them washed out.
  * Foreign (A5.2): Ashanti is Asante Twi, its own leaf beside Akan; Haoussa Hausa; Ouolof
    Wolof; Français, Arabe, Anglais, Russe the existing nodes.

Remainders:
  * "Autres langues nationales" (633,565; Comoé 105,474, Koulpelogo 110,497, Soum 44,130) and
    A5.2's "Autre langue africaine" (10,718, foreign African languages) both sit on
    `africa_other`, as Mali's "Autre langue du Mali" and "Autre langue africaine" do: both are
    African languages, the narrowest node holding either, and neither is a foreign language on
    `other`. The national remainder is very likely Kurumfé in Soum, Karaboro, Komono and Turka
    in Comoé, Birifor in the south-west and Yaana in Koulpelogo, but the census does not name
    them, so they are not guessed.
  * "Autre langue non africaine" (418): `other`.
  * "ND" (192,924, 1.5%): not drawn.
"""
GUR = "nigercongo.gur"
MANDE = "nigercongo.mande"

NAMES = {
    "Mooré": f"{GUR}.moore",
    "Goulmancema": f"{GUR}.gourmanchema",
    "Fulfuldé": "nigercongo.atlantic.fulah",
    "Dioula": f"{MANDE}.dyula",
    "Bissa": f"{MANDE}.bissa",
    "Bobo": f"{MANDE}.bobo",
    "Bwamu": f"{GUR}.bwamu",
    "Dafing": f"{MANDE}.marka",
    "Dagara": f"{GUR}.dagara",
    "Dogon": "nigercongo.dogon.dogon",
    "Gouin": f"{GUR}.cerma",
    "Kasséna": f"{GUR}.kasem",
    "Ko": f"{GUR}.winye",
    "Koussassé": f"{GUR}.kusaal",
    "Lyélé": f"{GUR}.lyele",
    "Lobiri": f"{GUR}.lobi",
    "Minianka": f"{GUR}.mamara",
    "Nuni": f"{GUR}.nuni",
    "San": f"{MANDE}.san",
    "Sembla": f"{MANDE}.sembla",
    "Sénoufo": f"{GUR}.senufo",
    "Siamou": "isolate.siamou",
    "Sissaka": f"{GUR}.sisaala",
    "Sonrhaï": "nilosaharan.songhay.songhay",
    "Tamachèque": "afroasiatic.berber.tamasheq",
    "Gurunsi": f"{GUR}.gurunsi",
    "Autres langues nationales": "africa_other",
    # A5.3's two foreign groups, split by A5.2 (sources/bf_rgph.py)
    "Ashanti": "nigercongo.kwa.asante",
    "Djerma": "nilosaharan.songhay.zarma",
    "Haoussa": "afroasiatic.chadic.hausa",
    "Ouolof": "nigercongo.atlantic.wolof",
    "Autre langue africaine": "africa_other",
    "Français": "indoeuropean.romance.french",
    "Arabe": "afroasiatic.arabic",
    "Anglais": "indoeuropean.germanic.english",
    "Russe": "indoeuropean.slavic.east.russian",
    "Autre langue non africaine": "other",
}

EXCLUDED = ("ND",)


def resolve(name):
    return NAMES.get(name)
