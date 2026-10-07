"""Sint Maarten, Census 2011, Table F-07, language most spoken in the household (persons in
private households) -> node. sources/sx_census.py; keyed by the table's own lower-case labels.

CALLS (sources/sx.md says more):
  french creole (2,972): `creole.french_based.haitian`, as cw2023's 2011 check row. The label
    could also hold Antillean Creole (Kweyol of Dominica and St Lucia), but Haitian nationals
    alone were about 2,300 of the 2011 population (6.9%, STAT's Population 2022 report, p.11),
    against a few hundred Dominica- and St Lucia-born; Haitian is the language meant.
  creole (42): `creole`. The census names no base language; Sint Maarten has English-, French-
    and Portuguese-based creole speakers, so the narrowest node holding them all is the root.
    It draws as "language not named", which is what the answer is.
  chinese: `sinotibetan.sinitic`, as cw, aw and other countries' unsplit "Chinese".
  hindi: Hindi. Much of the island's Indian community is Sindhi by origin, but the census
    printed Hindi and the map draws the answer given.
  filipino: `austronesian.philippine.filipino`.
  surnamese (5): Sranan Tongo, the language Surinamese people mean by "Surinaams"; Dutch and
    Sarnami speakers would have answered Dutch or Hindi.
  libanese (1), jordanian (2): Arabic. Nationality words given for a language; the census also
    prints arabic (100), and Arabic has no varieties in the tree.
  nigerian (8): `africa_other`. A nationality, not a language; Nigeria's languages span
    several families and the narrowest node holding them is the African remainder.
  other (4): `other`. no response (347): not drawn, in gap.
"""
NAMES = {
    "english": "indoeuropean.germanic.english",
    "spanish": "indoeuropean.romance.spanish",
    "french creole": "creole.french_based.haitian",
    "dutch": "indoeuropean.germanic.continental.dutch",
    "papiamentu": "creole.portuguese_based.papiamento",
    "hindi": "indoeuropean.indoaryan.central.hindi",
    "chinese": "sinotibetan.sinitic",
    "french": "indoeuropean.romance.french",
    "arabic": "afroasiatic.arabic",
    "creole": "creole",
    "filipino": "austronesian.philippine.filipino",
    "italian": "indoeuropean.romance.italian",
    "turkish": "turkic.turkish",
    "german": "indoeuropean.germanic.continental.german",
    "portuguese": "indoeuropean.romance.portuguese",
    "nigerian": "africa_other",
    "swedish": "indoeuropean.germanic.north.swedish",
    "telugu": "dravidian.southcentral.telugu",
    "vietnamese": "austroasiatic.vietnamese",
    "libanese": "afroasiatic.arabic",
    "hebrew": "afroasiatic.hebrew",
    "gujarati": "indoeuropean.indoaryan.gujarati.gujarati",
    "other": "other",
    "surnamese": "creole.english_based.sranan",
    "jordanian": "afroasiatic.arabic",
}
SKIP = {"no response", "Total"}


def resolve(label):
    if label in SKIP:
        return None
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"sx2011: unmapped label {label!r}")
