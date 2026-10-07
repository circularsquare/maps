"""Croatia, Popis 2021, mother tongue (materinski jezik) -> node. sources/hr_census.py.

Keyed by the English half of DZS's bilingual column headers, exactly as data/normalized/hr.csv
carries them. DZS prints Croatian, 24 named languages, "Other languages" and "Unknown"; nothing
is split finer at any level it publishes. The census defines mother tongue as the language first
learned in early childhood, or the one the person considers theirs in a multilingual household.

CALLS (sources/hr.md says more):
  Croatian, Serbian, Bosnian, Montenegrin: four leaves, as printed (rs2022, cz2021). Glottolog
    files all four standards as dialects of Serbian-Croatian-Bosnian (sout1528).
  "Serbo-Croatian" (8,182) and "Croato-Serbian" (4,278): two columns, two leaves. They are the
    two word orders of one Yugoslav-era name (srpskohrvatski, the usual one; hrvatskosrpski, the
    official name in socialist Croatia), and DZS prints them apart. Both sit in the old Serb
    areas (Knin, Kistanje, Biskupija, Dvor, Donji Lapac, Vukovar, Erdut), so the word order is
    not a regional difference; it is what people chose to write. Serbo-Croatian goes on the
    existing `serbocroatian` leaf (us, fi, cz); Croato-Serbian gets a new sibling
    `croatoserbian` rather than merging, because the census offered and counted them as two
    answers and the brief keeps every printed label.
  "Romani" (15,269; Medimurje: Orehovica 34%, Pribislavec 27%, Mala Subotica 21%,
    Nedelisce 15%): the leaf `romani.romani`, variety not stated. Many of Croatia's Roma,
    Medimurje's among them, speak Boyash (Bayash, Ljimba d'bjas), a Romanian dialect
    (Glottolog baya1255 under roma1327). The census has no Boyash answer; what each person
    wrote is unknowable from the table, so Romani stays Romani, and the map says so.
  "Romanian" (671; Slavonski Brod 171, Darda 53, Sveti Durd 11): `romance.romanian` as printed.
    The places are Boyash Roma settlements rather than a Romanian community, so much of this is
    likely Boyash too; not moved, for the same reason.
  "Vlach" (40; 28 in Slavonski Brod): `romance.vlach`, the node Serbia's Vlaski uses, same word.
    None is in Istria, so this is not Istro-Romanian (istr1245), whose speakers (Zejane,
    Susnjevica) must sit under Croatian or Other. Slavonski Brod again suggests Boyash Roma;
    40 people, kept as printed.
  "Ruthenian" (1,011; Bogdanovci 18%, Tompojevci 16%, Vukovar): Rusyn, as rs2022. Pannonian
    Rusyn. Ukrainian printed apart (1,198) and kept apart.
  "Hebrew" (82, mostly central Zagreb): `afroasiatic.hebrew`.
  "Other languages" (9,910): `other`. DZS gives no breakdown at any level. It is highest in
    the big cities (Split, Rijeka, central Zagreb), so mostly migrant; Croatia's own regional
    minority languages are all printed by name, and the census does not let an indigenous
    remainder be told apart, so there is nothing to put on a regional node.
  NOT_STATED: "Unknown" (20,840, 0.54%): not drawn, the entry's gap.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic"

NAMES = {
    "Croatian": f"{SL}.south.croatian",
    "Croato-Serbian": f"{SL}.south.croatoserbian",
    "Albanian": f"{IE}.albanian.albanian",
    "Bosnian": f"{SL}.south.bosnian",
    "Bulgarian": f"{SL}.south.bulgarian",
    "Montenegrin": f"{SL}.south.montenegrin",
    "Czech": f"{SL}.west.czech",
    "Hungarian": "uralic.hungarian",
    "Macedonian": f"{SL}.south.macedonian",
    "German": f"{IE}.germanic.continental.german",
    "Polish": f"{SL}.west.polish",
    "Romani": f"{IE}.indoaryan.romani.romani",
    "Romanian": f"{IE}.romance.romanian",
    "Russian": f"{SL}.east.russian",
    "Ruthenian": f"{SL}.east.rusyn",
    "Slovak": f"{SL}.west.slovak",
    "Slovenian": f"{SL}.south.slovenian",
    "Serbian": f"{SL}.south.serbian",
    "Serbo-Croatian": f"{SL}.south.serbocroatian",
    "Italian": f"{IE}.romance.italian",
    "Turkish": "turkic.turkish",
    "Ukrainian": f"{SL}.east.ukrainian",
    "Vlach": f"{IE}.romance.vlach",
    "Hebrew": "afroasiatic.hebrew",
    "Other languages": "other",
}
NOT_STATED = {"Unknown", "Total"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"hr2021: unmapped label {label!r}")
    return NAMES[label]
