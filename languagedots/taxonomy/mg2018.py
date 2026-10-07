"""Madagascar RGPH-3 2018, ability to speak a language (people of 3 and over, several allowed):
census label -> node. Labels as sources/mg_rgph.py writes them into data/normalized/mg.csv, which
are Tableau 2.2's column heads.

  Malagasy        -> austronesian.malagasy   the census asks about Malagasy as one language; the
                                              18 regional varieties (Glottolog splits Plateau,
                                              Tandroy, Antankarana, Tsimihety, Masikoro... under
                                              aust1307) are not asked, so no dialect node
  Français        -> French
  Anglais         -> English
  Autres langues  -> other                   "autres langues étrangères" in the report (p.17),
                                              unnamed; `other` holds everything the census filed
                                              there

French, English and the other languages are second languages here and countries/mg.py draws
their shares as Malagasy (AGENT_BRIEF §2, learned second languages); the mapping still says what
each label means.

Not languages: "Population" (all ages, Tome 1 Tableau 6) and "Population 3+" (Tableau 2.2's
base) are the denominators; resolve() gives None.
"""
NAMES = {
    "Malagasy": "austronesian.malagasy",
    "Français": "indoeuropean.romance.french",
    "Anglais": "indoeuropean.germanic.english",
    "Autres langues": "other",
    "Population": None,
    "Population 3+": None,
}


def resolve(label):
    return NAMES[label]
