"""Suriname, Census 7 (2004), "Most Spoken Language in the household", households per ressort
turned into people (sources/sr_census.py). Census label -> node, labels as data/normalized/sr.csv
writes them (ABS's own spellings: Portugese, Aucaans).

Every label the census prints as a language gets a node (spec 3.1). Families from Glottolog
(data/raw/glottolog); see taxonomy/tree.d/sr.txt.

Calls worth knowing:
  Sarnami -> a new leaf under Bihari: Glottolog's Caribbean Hindustani (cari1275, GY;NL;SR;TT),
      with Sarnami Hindustani as its Suriname dialect (sarn1238), classified Bihari > Bhojpuric.
      Its own node, not Bhojpuri: the census names it, and it is a koine of Bhojpuri and Awadhi.
  Javanese -> austronesian.javanese, the node other countries draw Javanese on. Glottolog keeps
      Caribbean Javanese (cari1276) apart from Javanese of Java; the census prints only
      "Javanese", and a reader knows the language by that name.
  The three Maroon languages the census names each get a leaf under English-based creoles,
      beside Sranan (pl.txt's node): Saramaccaans -> Saramaccan (sara1340); Aucaans -> Ndyuka
      (Aukan, ndyu1242); Paramaccaans -> Pamaka (Paramaccan, para1317). Glottolog files Pamaka
      as a dialect of Aukan; the census prints it as its own answer, so it is a sibling leaf,
      not a child (a child would turn Ndyuka into a group node and wash it out).
  Arowaks -> arawakan.lokono (Lokono, araw1276), Caraib -> cariban.galibi_kali_na (Kari'na,
      gali1262), both nodes other countries made.
  Chinese -> sinotibetan.sinitic, as bo2024, py2002 and pl2021: "Chinese" names no variety
      (Suriname's older Chinese community is largely Hakka, its newer one largely not).
Remainders:
  Other -> other. ABS's own 2012 report says the smaller Maroon groups' languages are filed
      in "other" (Census 8 Volume 3, under table HWG-06a), and the ressort figures show it
      also holds the indigenous languages the census does not name: 261 of Coeroeni's 310
      households, where the Trio and Wayana live, answered "Other". The census cannot tell the
      indigenous part apart from the rest, so the narrowest node containing everything filed
      there is `other`, not `americas_other`.
Not drawn (gap): Unknown.
"""

NAMES = {
    "Dutch": "indoeuropean.germanic.continental.dutch",
    "Sranan tongo": "creole.english_based.sranan",
    "Sarnami": "indoeuropean.indoaryan.bihari.sarnami",
    "Javanese": "austronesian.javanese",
    "Arowaks": "arawakan.lokono",
    "Caraib": "cariban.galibi_kali_na",
    "Saramaccaans": "creole.english_based.saramaccan",
    "Aucaans": "creole.english_based.ndyuka",
    "Paramaccaans": "creole.english_based.pamaka",
    "Chinese": "sinotibetan.sinitic",
    "Portugese": "indoeuropean.romance.portuguese",
    "English": "indoeuropean.germanic.english",
    "French": "indoeuropean.romance.french",
    "Other": "other",
    "Unknown": None,
}


def resolve(label):
    return NAMES[label]
