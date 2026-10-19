"""Algeria, Arab Barometer VI-VII (2020-2022) ethnic group read as language, plus French first
language from the earlier waves (sources/dz_survey.py) -> node.

Labels -> languages (Glottolog codes from data/raw/glottolog/languages.csv):
  * Arab, and "Other" ethnic group: Algerian Arabic (alge1239), a node of its own beside
    `arabic`, as Morocco's census put Darija on `afroasiatic.darija` (taxonomy/ma2024.py).
    "Other" names no language; the national language is the brief's place for it. Wave VII's
    22 "Tourag" answers sit there too: none came from the Tuareg south (sources/dz.md 3).
  * Amazigh: the survey does not ask which Berber language. Every wilaya's Amazigh answers are
    split over Algeria's Berber varieties by place (sources/dz_survey.py SPLIT, recorded in
    sources/dz.md section 4; 2026-10-19, before that only some wilayas were named and the rest,
    3.4M people, sat on the Berber group node). One label per variety, "Amazigh (<variety>)",
    below. The variety is Glottolog's language where it has one:
      Kabyle kaby1243, Chaouia of the Aures tach1249, Tumzabt (Mozabite) tumz1238,
      Western Algerian Berber chen1266 (Chenoua; Beni Snous is its dialect), Ouargli taga1278,
      Oued Righ tema1243, Sud Oranais-Gourara tazn1238 (Taznatit; Gourara and Touat are its
      dialects), Tidikelt-Tuat Tamazight tidi1241, Tahaggart Tamahaq taha1241 (Algeria's Tuareg;
      Mali's Tamasheq tama1365 is a different language), Tachelhit tach1250 (Tindouf only).
  * French (first language): French.
"""
AA = "afroasiatic"

# key -> (node id, label, Glottolog core latitude, longitude). The core is Glottolog's point for
# the language (data/raw/glottolog/languages.csv); sources/dz_survey.py uses it only to share a
# wilaya between two varieties spoken in it (Adrar, Tamanrasset, Ouargla).
VARIETIES = {
    "kabyle": (f"{AA}.berber.kabyle", "Kabyle", 36.4103, 4.76137),
    "chaouia": (f"{AA}.berber.chaouia", "Chaoui", 35.1867, 6.88901),
    "chenoua": (f"{AA}.berber.chenoua", "Chenoua", 36.5917, 2.19899),
    "tumzabt": (f"{AA}.berber.tumzabt", "Tumzabt", 32.5014, 3.85035),
    "ouargli": (f"{AA}.berber.ouargli", "Ouargli", 31.9454, 5.28444),
    "righ": (f"{AA}.berber.oued_righ", "Oued Righ", 33.3489, 6.06562),
    "taznatit": (f"{AA}.berber.taznatit", "Taznatit", 29.1107, 0.01992),
    "tidikelt": (f"{AA}.berber.tidikelt", "Tidikelt", 27.019, 1.86218),
    "tamahaq": (f"{AA}.berber.tamahaq", "Tamahaq", 22.5908, 8.73169),
    "tachelhit": (f"{AA}.berber.tachelhit", "Tachelhit", 29.7854, -7.77879),
}


def label(key):
    return f"Amazigh ({VARIETIES[key][1]})"


NAMES = {
    "Arab": f"{AA}.algerian_arabic",
    "Other ethnic group (incl. Tourag)": f"{AA}.algerian_arabic",
    "French (first language)": "indoeuropean.romance.french",
}
NAMES.update({label(k): v[0] for k, v in VARIETIES.items()})


def resolve(name):
    return NAMES.get(name)
