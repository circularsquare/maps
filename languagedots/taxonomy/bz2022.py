"""Belize, 2022 Population and Housing Census, Table 7: languages spoken well enough to hold a
conversation, people aged 4 and over, several answers allowed (sources/bz_census.py). Census
label -> node, labels as data/normalized/bz.csv writes them.

Every label the census prints as a language gets a node (spec 3.1). Families from Glottolog
(data/raw/glottolog); see taxonomy/tree.d/bz.txt.

Calls worth knowing:
  Speaks Creole -> a new leaf, Belize Kriol (Glottolog beli1260, Belize Kriol English), under
      English-based creoles beside Nicaraguan Creole English and Jamaican Creole. The census's
      word is "Creole"; in Belize that can only mean Kriol (no French-based creole is spoken).
  Speaks Maya Ketchi -> Q'eqchi' (kekc1242), Speaks Maya Mopan -> Mopan (mopa1243), Speaks
      Maya Yucatec -> Maya (Yucatec) (yuca1254): gt.txt's and mx.txt's nodes.
  Speaks Garifuna -> arawakan.garifuna, gt.txt's and ni.txt's node.
  Speaks German -> German. Most of Belize's German speakers are Mennonites (sources/bz.md),
      whose everyday language is Plautdietsch (Glottolog lists a Belize Plautdietsch dialect,
      beli1263) and whose church and school language is Standard German. The census prints
      "German" and does not say which, so it is drawn as German, as py2002 does.
  Speaks Chinese -> sinotibetan.sinitic, as bo2024, py2002 and sr2004: "Chinese" names no
      variety (Belize's older Chinese community is largely Cantonese-speaking, its newer
      Taiwanese one largely Mandarin and Taiwanese Hokkien).
  Speaks Hindi -> Hindi.
Remainders:
  Speaks Other -> other. Two thirds of it (1,676 of 2,475) is in Orange Walk, where the
      Mennonite settlements of Shipyard, Blue Creek and Little Belize are, so much of it may be
      Plautdietsch or Dutch written in, but the census prints no breakdown, and it can also hold
      indigenous, foreign and sign languages. The narrowest node containing all of it is
      `other`.
Not drawn (gap): Cannot Speak.
"""

NAMES = {
    "Speaks English": "indoeuropean.germanic.english",
    "Speaks Spanish": "indoeuropean.romance.spanish",
    "Speaks Creole": "creole.english_based.belize_kriol",
    "Speaks Maya Ketchi": "mayan.kichean.qeqchi",
    "Speaks Maya Mopan": "mayan.yucatecan.mopan",
    "Speaks German": "indoeuropean.germanic.continental.german",
    "Speaks Garifuna": "arawakan.garifuna",
    "Speaks Other": "other",
    "Speaks Maya Yucatec": "mayan.yucatecan.maya",
    "Speaks Chinese": "sinotibetan.sinitic",
    "Speaks Hindi": "indoeuropean.indoaryan.central.hindi",
    "Cannot Speak": None,
}


def resolve(label):
    return NAMES[label]
