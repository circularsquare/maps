"""Slovenia, Popis 2002, mother tongue (materni jezik) -> node. sources/si_popis2002.py.

Keyed by SURS's Slovene labels as data/normalized/si.csv carries them (table 05W1007S). One
answer per person. The last Slovenian census to ask; 2011 and 2021 are register-based.

CALLS (sources/si.md says more):
  "Srbsko-hrvaški" (Serbo-Croatian, 36,265, more than Serbian or Bosnian): its own label, so the
    existing `serbocroatian` leaf (hr2021, us, fi, cz). Not folded into Serbian or Croatian.
  "Hrvaško-srbski" (Croato-Serbian, 126) is printed only in the national table and sits inside
    the municipal "Drugi"; it is not drawn apart.
  "Romski": `romani.romani`, variety not stated.
  "Drugi" (6,240): the 21 further languages of the national table (Russian, Montenegrin, Czech,
    Ukrainian, English, Slovak, Polish, Romanian, Turkish, Chinese...) plus 1,588 SURS left as
    other. Several families, so `other`. Nothing indigenous to Slovenia is in it: the
    autochthonous minorities (Italian, Hungarian) and Romani have their own columns.
  "Neznano" (unknown, 52,316, 2.66%): not drawn, the entry's gap.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic.south"

NAMES = {
    "Slovenski": f"{SL}.slovenian",
    "Italijanski": f"{IE}.romance.italian",
    "Madžarski": "uralic.hungarian",
    "Romski": f"{IE}.indoaryan.romani.romani",
    "Albanski": f"{IE}.albanian.albanian",
    "Bosanski": f"{SL}.bosnian",
    "Hrvaški": f"{SL}.croatian",
    "Makedonski": f"{SL}.macedonian",
    "Nemški": f"{IE}.germanic.continental.german",
    "Srbski": f"{SL}.serbian",
    "Srbsko-hrvaški": f"{SL}.serbocroatian",
    "Drugi": "other",
}
NOT_DRAWN = {"Neznano", "Materni jezik - SKUPAJ"}


def resolve(label):
    if label in NOT_DRAWN:
        return None
    if label not in NAMES:
        raise KeyError(f"si2002: unmapped label {label!r}")
    return NAMES[label]
