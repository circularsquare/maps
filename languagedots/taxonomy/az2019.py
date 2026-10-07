"""Azerbaijan 2019 census, mother tongue by nationality (sources/az_census.py) -> node.

Keyed by data/normalized/az.csv's `source_category`: Table 30's 12 named mother-tongue columns in
English as printed (Talish -> "Talysh", Sakhur -> "Tsakhur", Udin -> "Udi", Kurd -> "Kurdish"
renamed in the normaliser), plus "own language: <nationality>" for the column "language of the
nationality (ethnic group) that it belongs", which names a different language on every row.

CALLS.
  own language, Ingiloy: Ingilo, a node of its own beside Georgian (tree.d/az.txt). The census
    keeps Ingiloys apart from Georgians and prints their own language apart; 842 people.
  own language, Griz: Kryts. own language, Khaput: Haput, beside Kryts though Glottolog has it as
    a Kryts dialect; the census asks them as two nationalities.
  own language, Jews: `other.jewish`. The census names no language: Azerbaijan's Jews are mostly
    Mountain Jews (Juhuri), with Ashkenazi and Georgian Jews in Baku, so Juhuri, Hebrew and Yiddish
    are all possible answers and the narrowest node holding them all is the root. A named label
    may not sit on a group, so it is a leaf under `other`, as Ethiopia's unclassifiable labels are.
    4,966 people.
  own language, Tatarian: Tatar. own language, Ukrainian, Russian, Armenian, Turkish, Kurd: the
    obvious language.
  own language, Other: the other nationalities' own languages, unnamed: `other`. 2,447 people.
  "Other languages" column: `other`. 1,960 people.
"""
ND = "nakhdaghestanian"
IR = "indoeuropean.iranian"

LANGS = {
    "Azerbaijani": "turkic.azerbaijani",
    "Turkish": "turkic.turkish",
    "Russian": "indoeuropean.slavic.east.russian",
    "Talysh": f"{IR}.talysh",
    "Lezgi": f"{ND}.lezgic.lezgian",
    "Tat": f"{IR}.tat",
    "Kurdish": f"{IR}.kurdish",
    "Georgian": "kartvelian.georgian",
    "Avar": f"{ND}.avarandic.avar",
    "Tsakhur": f"{ND}.lezgic.tsakhur",
    "Udi": f"{ND}.lezgic.udi",
    "Other languages": "other",
}
# nationality (Table 30's English row label) -> its own language's node
OWN = {
    "Azerbaijani": "turkic.azerbaijani",
    "Lezgi": f"{ND}.lezgic.lezgian",
    "Talish": f"{IR}.talysh",
    "Russian": "indoeuropean.slavic.east.russian",
    "Ukrainian": "indoeuropean.slavic.east.ukrainian",
    "Avar": f"{ND}.avarandic.avar",
    "Turkish": "turkic.turkish",
    "Tat": f"{IR}.tat",
    "Sakhur": f"{ND}.lezgic.tsakhur",
    "Georgian": "kartvelian.georgian",
    "Ingiloy": "kartvelian.ingilo",
    "Kurd": f"{IR}.kurdish",
    "Tatarian": "turkic.tatar",
    "Griz": f"{ND}.lezgic.kryts",
    "Jews": "other.jewish",
    "Udin": f"{ND}.lezgic.udi",
    "Khinalig": f"{ND}.khinalug",
    "Budug": f"{ND}.lezgic.budukh",
    "Armenian": "indoeuropean.armenian.armenian",
    "Khaput": f"{ND}.lezgic.haput",
    "Other": "other",
}
NAMES = {**LANGS, **{f"own language: {k}": v for k, v in OWN.items()}}


def resolve(label):
    if label not in NAMES:
        raise KeyError(f"az2019: unmapped label {label!r}")
    return NAMES[label]
