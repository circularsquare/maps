"""French Polynesia, Recensement de la population 2012 (ISPF / INSEE), sheet LAN1b: language most
often spoken in the family, population 15+, by subdivision (sources/pf_rp2012.py). Census label
-> node, labels as data/normalized/pf.csv writes them for geo_level "subdivision".

The five labels are the finest split ISPF published below the territory: GROUPS, not languages.
Each sits on the narrowest node that holds everything ISPF filed under it (spec 3.2), using the
national "Chiffres clés" sheet of the same file, which names the members:

  Français -> French.
  Langue polynésienne -> austronesian.oceanic. Nationally 57,283: Tahitien 46,759, Marquisien
      5,137, Paumotu 2,599, Langues australes (Rapa, Tubuai, Rimatara, Raivavae, Rurutu) 2,366,
      Mangarévien 422. The tree has no Polynesian node (Tahitian, Maori, Samoan, Tongan sit flat
      under Oceanic in fi.txt, au.txt and cl.txt), so Oceanic is the narrowest node holding all
      five; it draws as "language not named", which is what this table says. Splitting it by
      the national mix would put Tahitian on the Marquesas; there is no table that splits it by
      subdivision (sources/pf.md §1).
  Langue asiatique -> other. Nationally Hakka 665, other Chinese 865, Japanese 63, other Asian
      (Indonesian, Vietnamese...) 48: Sino-Tibetan, Japonic and at least one more family.
  Langue européenne (sauf français) -> indoeuropean. Nationally German 126, English 465, Spanish
      66, Portuguese 4, Italian 22, "other European" 8. The 8 could in principle include a
      non-Indo-European language (Hungarian, Finnish, Basque); the node can leave out at most
      8 of 691.
  Autres -> other. Nationally regional languages of France other than French 142 (Kanak 59,
      Wallisian and Futunan 53, Antillean Creole 20...), Pacific 306 (Fijian 285...), other
      foreign 455 (Arabic 441...) and "Sourd et muet" 357, which ISPF counts as an answer to
      the language question. Several families plus sign: `other`.
"""

NAMES = {
    "Français": "indoeuropean.romance.french",
    "Langue polynésienne": "austronesian.oceanic",
    "Langue asiatique": "other",
    "Langue européenne (sauf français)": "indoeuropean",
    "Autres": "other",
}


def resolve(label):
    return NAMES[label]
