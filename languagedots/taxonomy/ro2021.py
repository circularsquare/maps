"""Romania, RPL 2021, mother tongue (limba maternă), table 2.3 (sources/ro_census.py) -> node.

Keyed by INS's column headers exactly as data/normalized/ro.csv carries them. The 22 named languages
are Romania's recognised minorities' languages plus Italian, Greek and Yiddish; nothing else is
named.

CALLS (sources/ro.md says more):
  "Tatara" (Tatar, 13,805, Constanţa and Tulcea): Crimean Tatar. The Tatars of Dobruja came from
    Crimea and the Nogai steppe; Glottolog files their speech, Dobruja Tatar (dobr1234), as a
    dialect of Crimean Tatar (crim1257), and Crimean Tatar's Glottolog countries include RO. Volga
    Tatar (`turkic.tatar`) is not spoken there.
  "Rusa" (Russian, 14,414): Russian. Almost all of it is the Lipovans of the Danube delta (Tulcea),
    whose Russian is an Old Believer dialect; the census names Russian and that is the node.
  "Ruteana" (Ruthenian, 594): Rusyn, the recognised Ruthenian minority of Maramureş and the Banat.
    Ukrainian is printed apart (40,861), so the two stay apart.
  "Macedoneana" (Macedonian, 201): Macedonian (South Slavic), as printed. Romania's recognised
    Macedonian minority is the Slavic one; Aromanian (Macedo-Romanian) has no column of its own,
    so its speakers are in Romanian or "Alta limba materna" and cannot be drawn.
  "Romani" (199,050): the leaf `romani.romani`, variety not stated (as Ukraine's).
  "Idis" (Yiddish, 597): Yiddish.
  "Alta limba materna" (other mother tongue, 19,741): `other`. Nothing in the table says whether
    it is a regional language (Aromanian, Csángó) or a migrant one, so it is not split.
  NOT_STATED: "Informatie nedisponibila" (2,502,378, 13.1%): not drawn, the entry's gap. The 2021
    census was built largely from registers, which hold no language.
"""
RO = "indoeuropean.romance"
SL = "indoeuropean.slavic"
TU = "turkic"

NAMES = {
    "Româna": f"{RO}.romanian",
    "Maghiara": "uralic.hungarian",
    "Romani": "indoeuropean.indoaryan.romani.romani",
    "Ucraineana": f"{SL}.east.ukrainian",
    "Germana": "indoeuropean.germanic.continental.german",
    "Turca": f"{TU}.turkish",
    "Rusa": f"{SL}.east.russian",
    "Tatara": f"{TU}.crimean_tatar",
    "Sarba": f"{SL}.south.serbian",
    "Slovaca": f"{SL}.west.slovak",
    "Bulgara": f"{SL}.south.bulgarian",
    "Croata": f"{SL}.south.croatian",
    "Italiana": f"{RO}.italian",
    "Greaca": "indoeuropean.hellenic.greek",
    "Ceha": f"{SL}.west.czech",
    "Polona": f"{SL}.west.polish",
    "Ruteana": f"{SL}.east.rusyn",
    "Armeana": "indoeuropean.armenian.armenian",
    "Albaneza": "indoeuropean.albanian.albanian",
    "Macedoneana": f"{SL}.south.macedonian",
    "Idis": "indoeuropean.germanic.continental.yiddish",
    "Alta limba materna": "other",
}
NOT_STATED = {"Informatie nedisponibila", "POPULATIA REZIDENTA TOTAL"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"ro2021: unmapped label {label!r}")
    return NAMES[label]
