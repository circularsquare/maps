"""New Caledonia, Recensement de la population 2019 (INSEE-ISEE): Kanak language spoken, people
aged 15 and over -> node. Keyed by the categories sources/nc_rp2019.py writes, which are ISEE's
labels as printed in 'par commune-langue' of langues-vernaculaires-locuteurs.xls.

Every label is a language or a named dialect area and gets a leaf (spec 3.1; tree.d/nc.txt has
the groups and the Glottolog check). Label notes:
  * "Paicï" is ISEE's typing of Paicî (the ALK and Glottolog spelling); the node says Paicî.
  * "Yalâyu" is Nyelâyu (Glottolog nyal1254 "Belep", ISO yly), of Belep, Pouebo and Ouegoa.
  * "Fwa Kumak" is Nêlêmwa-Nixumwak (kuma1276) of Koumac and Poum, by its Kumak dialect's name.
  * "Dialectes de Voh-Koné" and "Dialectes de l'extrême sud" are kept as printed, one leaf
    each, since the census does not split them (spec 3.1, as Mexico's INALI clusters).
  * "Faga uvea" is Fagauvea (West Uvean), Polynesian.
  * "Tayo" is the French-based creole of Saint-Louis (Mont-Dore); ISEE counts it as Kanak.

NOT TABLE LANGUAGES (countries/nc.py turns them into rows):
  * Parle minus the mentions: people who said they speak a Kanak language but did not name it,
    on KANAK, the areal node over every Kanak language (spec 3.2). It leaves out Tayo, which
    ISEE also files as Kanak; Tayo is 1.6% of the named mentions and 86% of its speakers live in
    Mont-Dore, so the mismatch is at most a few dozen people.
  * Everyone else aged 15+ (only understands a Kanak language, or neither) on French, derived
    (spec 3.5): the census asks about no other language.
  * Population and Population 15+: universes, not answers.
"""
K = "austronesian.oceanic.kanak"
KANAK = K
FRENCH = "indoeuropean.romance.french"

NAMES = {
    # Extreme North
    "Caac": f"{K}.northern.caac",
    "Yalâyu": f"{K}.northern.yalayu",
    "Fwa Kumak": f"{K}.northern.fwa_kumak",
    "Yûâga": f"{K}.northern.yuaga",
    # North
    "Jawe": f"{K}.northern.jawe",
    "Nèmi": f"{K}.northern.nemi",
    "Pwâpwâ": f"{K}.northern.pwapwa",
    "Fwâi": f"{K}.northern.fwai",
    "Pwaamei": f"{K}.northern.pwaamei",
    "Pije": f"{K}.northern.pije",
    "Dialectes de Voh-Koné": f"{K}.northern.voh_kone",
    # Centre
    "Cèmuhî": f"{K}.northern.cemuhi",
    "Paicï": f"{K}.northern.paici",
    # South
    "Ajië": f"{K}.southern.ajie",
    "Arhâ": f"{K}.southern.arha",
    "Arhö": f"{K}.southern.arho",
    "Orowe": f"{K}.southern.orowe",
    "Nèku": f"{K}.southern.neku",
    "Tîrî": f"{K}.southern.tiri",
    "Zîchë": f"{K}.southern.ziche",
    "Xârâcùù": f"{K}.southern.xaracuu",
    "Xârâgùrè": f"{K}.southern.xaragure",
    # Far South
    "Drubea": f"{K}.southern.drubea",
    "Dialectes de l'extrême sud": f"{K}.southern.numee",
    "Tayo": "creole.french_based.tayo",
    # Loyalty Islands
    "Nengone": f"{K}.loyalty.nengone",
    "Drehu": f"{K}.loyalty.drehu",
    "Iaai": f"{K}.loyalty.iaai",
    "Faga uvea": f"{K}.west_uvean",
    # not languages
    "Total locuteurs": None,
    "Parle": None,
    "Comprend": None,
    "Aucune connaissance": None,
    "Population 15+": None,
    "Population": None,
}
EXCLUDED = ("Total locuteurs", "Parle", "Comprend", "Aucune connaissance", "Population 15+",
            "Population")
EXTRA_NODES = [KANAK, FRENCH]


def resolve(name):
    return NAMES[name]
