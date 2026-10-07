"""Luxembourg, census of 8 November 2021, main language (sources/lu_census.py) -> node. Keyed by
STATEC's French labels exactly as data/normalized/lu.csv carries them: the six boxes of the form
plus the 52 write-ins of RP2021 n°8 Tableau 3 and the unnamed remainder.

The question asked for ONE language, "the language in which you think and which you know best".
People who did not answer (10.4%) or were "not of an age to speak" (2.2%) are not in the table.

CALLS (sources/lu.md says more):
  "Luxembourgeois": Luxembourgish (ltz). The census does not split Moselle Franconian varieties.
  The six Yugoslav-successor labels each get their own node, as the census prints them (Tableau
    4 of the publication groups them as "BCMS" only in prose): Serbe, Bosniaque, Monténégrin,
    Croate, Serbo-Croate on the existing nodes; "Yougoslave" (331) gets a node of its own,
    `yugoslav`, because it is an answer people gave, not a spelling of Serbo-Croatian.
  "Créole" (1,148): the bare word, with "Créole du Cap-Vert" (1,510) printed apart. The
    publication itself (p. 6) says it "peut désigner les parlers de diverses régions", so it
    goes on the `creole` group, as au2021's "Creole, nfd" does. Many are probably Cape Verdean
    speakers who did not write the island, but nothing printed says so.
  "Persan" (592) and "Farsi" (147): two names for one language, merged on Persian (Farsi), the
    node's own label. A spelling variant of one answer, which §3 allows.
  "Philippin" (188) and "Tagalog" (121): kept apart on the existing Filipino and Tagalog nodes.
  "Flamand" (158): the existing `flemish` node, apart from "Néerlandais" (3,661).
  "Pular" (113): Fula, the existing node, as gn2014 maps Guinea's "Poular" (Pular, pula1262, is
    Guinea's Fula, and no other Fula label is printed here).
  "Chinois" (2,855): the Sinitic group "Chinese", drawn in full colour (build.py), as bo2024,
    bz2022 and pl2021 do: the word names no variety.
  "Kurde": Kurdish. "Hindi", "Bengali", "Népalais", "Tamoul": the existing leaves.
  The unnamed remainder of the write-ins, 3,436 people in languages with 100 speakers or fewer,
    goes on `other`: the write-ins can be any language in the world.
"""
GE = "indoeuropean.germanic"
RO = "indoeuropean.romance"
SL = "indoeuropean.slavic"

NAMES = {
    "Luxembourgeois": f"{GE}.continental.luxembourgish",
    "Portugais": f"{RO}.portuguese",
    "Français": f"{RO}.french",
    "Anglais": f"{GE}.english",
    "Italien": f"{RO}.italian",
    "Allemand": f"{GE}.continental.german",
    "Espagnol": f"{RO}.spanish",
    "Arabe": "afroasiatic.arabic",
    "Néerlandais": f"{GE}.continental.dutch",
    "Russe": f"{SL}.east.russian",
    "Polonais": f"{SL}.west.polish",
    "Roumain": f"{RO}.romanian",
    "Chinois": "sinotibetan.sinitic",
    "Serbe": f"{SL}.south.serbian",
    "Bosniaque": f"{SL}.south.bosnian",
    "Grec": "indoeuropean.hellenic.greek",
    "Monténégrin": f"{SL}.south.montenegrin",
    "Créole du Cap-Vert": "creole.portuguese_based.kabuverdianu",
    "Albanais": "indoeuropean.albanian.albanian",
    "Hongrois": "uralic.hungarian",
    "Créole": "creole",
    "Serbo-Croate": f"{SL}.south.serbocroatian",
    "Danois": f"{GE}.north.danish",
    "Turc": "turkic.turkish",
    "Bulgare": f"{SL}.south.bulgarian",
    "Suédois": f"{GE}.north.swedish",
    "Tigrigna": "afroasiatic.ethiosemitic.tigrinya",
    "Croate": f"{SL}.south.croatian",
    "Lithuanien": "indoeuropean.baltic.lithuanian",
    "Slovaque": f"{SL}.west.slovak",
    "Tchèque": f"{SL}.west.czech",
    "Finnois": "uralic.finnish",
    "Persan": "indoeuropean.iranian.persian",
    "Farsi": "indoeuropean.iranian.persian",       # the same language as "Persan" (see above)
    "Hindi": "indoeuropean.indoaryan.central.hindi",
    "Ukrainien": f"{SL}.east.ukrainian",
    "Yougoslave": f"{SL}.south.yugoslav",
    "Thaïlandais": "kradai.thai",
    "Estonien": "uralic.estonian",
    "Kurde": "indoeuropean.iranian.kurdish",
    "Japonais": "japonic.japanese",
    "Slovène": f"{SL}.south.slovenian",
    "Macédonien": f"{SL}.south.macedonian",
    "Islandais": f"{GE}.north.icelandic",
    "Catalan": f"{RO}.catalan",
    "Vietnamien": "austroasiatic.vietnamese",
    "Philippin": "austronesian.philippine.filipino",
    "Afrikaans": f"{GE}.continental.afrikaans",
    "Flamand": f"{GE}.continental.flemish",
    "Tamoul": "dravidian.southern.tamil",
    "Népalais": "indoeuropean.indoaryan.pahari.eastern.nepali",
    "Arménien": "indoeuropean.armenian.armenian",
    "Letton": "indoeuropean.baltic.latvian",
    "Tagalog": "austronesian.philippine.tagalog",
    "Pular": "nigercongo.atlantic.fulah",
    "Norvégien": f"{GE}.north.norwegian",
    "Bengali": "indoeuropean.indoaryan.eastern.bengali",
    "Coréen": "koreanic.korean",
    "Autre langue (moins de 100 locuteurs)": "other",
}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"lu2021: unmapped label {label!r}")
    return NAMES[label]
