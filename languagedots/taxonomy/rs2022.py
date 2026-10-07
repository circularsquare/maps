"""Serbia, Popis 2022, mother tongue (maternji jezik) -> node. sources/rs_census.py.

Keyed by the English half of RZS's bilingual column headers, tidied for whitespace, exactly as
data/normalized/rs.csv carries them. RZS prints 18 named languages, "Other languages", "Did not
declare" and "Unknown"; nothing is split finer at any level it publishes.

CALLS (sources/rs.md says more):
  Serbian, Bosnian, Croatian, Montenegrin: four leaves, as printed. Glottolog files all four
    standards (serb1264, bosn1245, mont1282) as dialects of Serbian-Croatian-Bosnian; the census
    asks people to name their mother tongue and they named these, so each keeps its own node
    (as cz2021 does). The labels follow ethnicity closely, and the map says so.
  "Bunjevački" (Bunjevac, 3,319, 3,005 of them in Subotica): a new leaf
    `slavic.south.bunjevac`. Glottolog files Bunjevac (bunj1247) as a dialect of
    Serbian-Croatian-Bosnian; the census prints it apart from Serbian and Croatian.
  "Vlach language" (Vlach, 23,216; Kučevo 19.6%, Žagubica, Negotin, Petrovac na Mlavi, Boljevac):
    a new leaf `romance.vlach` beside Romanian. It is the Romanian speech of the Vlachs of
    eastern Serbia (the Timok, Mlava and Pek valleys); Glottolog's Romanian (roma1327) lists its
    two dialect groups, Ungureni and Tarani, as dialects. RZS prints it apart from Romanian
    (21,477, almost all in the Banat: Alibunar, Vršac, Žitište), so the two stay apart. A
    sibling, not a child, of Romanian: a child would turn Romanian into a group, drawn washed
    out (hu.txt's Boyash, same reasoning). Not Aromanian, which Greece and North Macedonia
    also call Vlach; the distribution settles it.
  "Ruthenian" (8,725, Ruski Krstur in Kula, Vrbas, Novi Sad): Rusyn, as ro2021 and hu2022.
    This is Pannonian Rusyn, which Glottolog files as a dialect of Rusyn (pann1240 under
    rusy1239). Ukrainian is printed apart (1,527) and stays apart.
  "Roma language" (79,687): the leaf `romani.romani`, variety not stated (as Hungary's and
    Romania's).
  "Other languages" (45,641): `other`. RZS gives no breakdown at any level and does not say
    which are regional and which migrant, so there is no indigenous remainder to keep apart.
    Its largest shares are Dimitrovgrad (9.7%) and Subotica (4.4%), both places with their own
    local identities; nothing published says what was written there.
  NOT_STATED: "Unknown" (303,179, 4.56%) and "Did not declare" (88,122, 1.33%): not drawn,
    the entry's gap.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic"

NAMES = {
    "Serbian": f"{SL}.south.serbian",
    "Albanian": f"{IE}.albanian.albanian",
    "Bosnian": f"{SL}.south.bosnian",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Bunjevački": f"{SL}.south.bunjevac",
    "Vlach language": f"{IE}.romance.vlach",
    "Hungarian": "uralic.hungarian",
    "Macedonian": f"{SL}.south.macedonian",
    "German": f"{IE}.germanic.continental.german",
    "Roma language": f"{IE}.indoaryan.romani.romani",
    "Romanian": f"{IE}.romance.romanian",
    "Russian": f"{SL}.east.russian",
    "Ruthenian": f"{SL}.east.rusyn",
    "Slovak": f"{SL}.west.slovak",
    "Slovenian": f"{SL}.south.slovenian",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Croatian": f"{SL}.south.croatian",
    "Montenegrin": f"{SL}.south.montenegrin",
    "Other languages": "other",
}
NOT_STATED = {"Unknown", "Did not declare", "Total"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"rs2022: unmapped label {label!r}")
    return NAMES[label]
