"""Algeria, Arab Barometer VI-VII (2020-2022) ethnic group read as language, plus French first
language from the earlier waves (sources/dz_survey.py) -> node.

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv):
  * Arab, and "Other" ethnic group: Algerian Arabic (alge1239), a node of its own beside
    `arabic`, as Morocco's census put Darija on `afroasiatic.darija` (taxonomy/ma2024.py).
    "Other" names no language; the national language is the brief's place for it. Wave VII's
    22 "Tourag" answers sit there too: none came from the Tuareg south (sources/dz.md 3).
  * Amazigh: the survey does not ask which Berber language. Where a wilaya has one Berber
    language and no other (spec section 3, a label whose meaning depends on place; the wilaya
    sets are in sources/dz_survey.py), the answer goes on it:
      Kabyle-speaking wilayas  -> Kabyle (kaby1243), ca.txt's `berber.kabyle`
      Chaoui-speaking wilayas  -> Chaouia of the Aures (tach1249)
      Ghardaia                 -> Tumzabt, the Mozabite language (tumz1238)
      Tamanrasset and Illizi   -> Tahaggart Tamahaq (taha1241), the Algerian Tuareg language;
                                  Mali's Tamasheq (tama1365) is a different language
    Everywhere else (Algiers, Oran, the West, the Sahara's north) the answer is unnamed and
    goes on the Berber group node, drawn as "language not named".
  * French (first language): French.
"""
AA = "afroasiatic"

NAMES = {
    "Arab": f"{AA}.algerian_arabic",
    "Other ethnic group (incl. Tourag)": f"{AA}.algerian_arabic",
    "Amazigh": f"{AA}.berber",
    "Amazigh (in a Kabyle-speaking wilaya)": f"{AA}.berber.kabyle",
    "Amazigh (in a Chaoui-speaking wilaya)": f"{AA}.berber.chaouia",
    "Amazigh (in Ghardaia)": f"{AA}.berber.tumzabt",
    "Amazigh (in a Tuareg wilaya)": f"{AA}.berber.tamahaq",
    "French (first language)": "indoeuropean.romance.french",
}


def resolve(name):
    return NAMES.get(name)
