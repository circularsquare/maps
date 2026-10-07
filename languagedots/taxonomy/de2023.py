"""Germany, Mikrozensus 2023, language spoken predominantly at home -> node.

Keyed by Destatis' own labels from table 12211-40 (sources/de_mz.py writes them into de.csv's
`land` rows): German and 32 languages. Calls (sources/de.md says more):
  * "Deutsch" is "nur Deutsch" plus "vorwiegend deutsch", both German as the main home language.
  * "Chinesisch" (Chinese) names a group, not a language: it sits on Sinitic, as cz2021, pl2021,
    us2024 and uk2021 put unspecified Chinese.
  * "Kurdisch" is one label for Kurmanji, Sorani and Zazaki speakers alike (Zazaki is often
    counted as Kurdish by its speakers); it goes on the Kurdish leaf.
  * "Eine andere in Europa gesprochene Sprache" and "Eine andere in Asien gesprochene Sprache"
    name parts of the world, not languages or families (Europe's holds Czech, Swedish, Finnish,
    Sorbian, Romani and Basque alike; Asia's Tamil, Thai, Korean and Armenian): `other`, as
    bw2011 and na2011 put "Other European languages". "Eine andere in Afrika gesprochene Sprache"
    goes on `africa_other`, kept apart from `other` (Anita, 2026-10-04).
  * "Eine sonstige Sprache" (any other language) is `other`.
Nobody is "not stated": the Mikrozensus imputes, and 12211-40 has no such row.
"""
IE = "indoeuropean"
GE = f"{IE}.germanic"
RO = f"{IE}.romance"
SL = f"{IE}.slavic"
IR = f"{IE}.iranian"
IA = f"{IE}.indoaryan.central"

NAMES = {
    "Deutsch": f"{GE}.continental.german",
    # J2's west-European group
    "Englisch": f"{GE}.english",
    "Französisch": f"{RO}.french",
    "Italienisch": f"{RO}.italian",
    "Spanisch": f"{RO}.spanish",
    "Niederländisch": f"{GE}.continental.dutch",
    # J2's own columns
    "Polnisch": f"{SL}.west.polish",
    "Russisch": f"{SL}.east.russian",
    "Türkisch": "turkic.turkish",
    "Arabisch": "afroasiatic.arabic",
    # J2's other European group
    "Albanisch": f"{IE}.albanian.albanian",
    "Bosnisch": f"{SL}.south.bosnian",
    "Bulgarisch": f"{SL}.south.bulgarian",
    "Dänisch": f"{GE}.north.danish",
    "Griechisch": f"{IE}.hellenic.greek",
    "Kroatisch": f"{SL}.south.croatian",
    "Mazedonisch": f"{SL}.south.macedonian",
    "Portugiesisch": f"{RO}.portuguese",
    "Rumänisch": f"{RO}.romanian",
    "Serbisch": f"{SL}.south.serbian",
    "Ukrainisch": f"{SL}.east.ukrainian",
    "Ungarisch": "uralic.hungarian",
    "Eine andere in Europa gesprochene Sprache": "other",
    # J2's other group
    "Chinesisch": "sinotibetan.sinitic",
    "Hindi": f"{IA}.hindi",
    "Kurdisch": f"{IR}.kurdish",
    "Paschtu": f"{IR}.pashto",
    "Persisch": f"{IR}.persian",
    "Urdu": f"{IA}.urdu",
    "Vietnamesisch": "austroasiatic.vietnamese",
    "Eine andere in Afrika gesprochene Sprache": "africa_other",
    "Eine andere in Asien gesprochene Sprache": "other",
    "Eine sonstige Sprache": "other",
}
NOT_STATED = set()


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"de2023: unmapped label {label!r}")
    return NAMES[label]
