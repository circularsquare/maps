"""Mauritius 2022 Housing and Population Census, Table D9, language usually spoken at home -> node.

Keyed by the column labels sources/mu_hpc.py writes. Twelve columns, each person in exactly one:
a home that speaks two languages is filed by Statistics Mauritius under the first-named one
(sources/mu_hpc.py's docstring proves the rule against Table D8), so `Bhojpuri` holds the
63,101 people of "Bhojpuri & Creole" homes and `Creole` the 36,762 of "Creole & French".

  * Creole is Mauritian Creole (Morisyen), on ca.txt's `morisyen` leaf. The same column on
    Rodrigues (42,832 people) is Rodrigues Creole, which Glottolog files as a dialect of
    Morisyen (rodr1234 under mori1278); the census prints one label for both, so one node.
  * Bhojpuri is Mauritian Bhojpuri, a dialect of Bhojpuri in Glottolog (maur1239 under
    bhoj1244): tree.txt's `bhojpuri` leaf.
  * Bangla (14,483, 81% male) is Bengali, overwhelmingly Bangladeshi contract workers: on
    `bengali`. D8 also prints a separate "Bengali" (110), which D9 files under Other.
  * "Chinese languages" (997) is D9's pooling of D8's Cantonese 17, Chinese 369, Hakka 60,
    Mandarin 406 and Other Chinese 145. Not split at any geography, so it is the unnamed Sinitic
    remainder on `sinitic`, drawn as "Chinese, language not named".
  * English, French, Hindi, Marathi, Tamil, Telugu, Urdu: their own leaves. Marathi, Tamil,
    Telugu and Urdu are single-language homes only (their combinations are under Creole,
    Bhojpuri or Hindi by the first-named rule).
  * "Other & Not stated" (5,917) pools named languages (Malagasy 3,172, Sinhala 463, Other
    Mixed European 705, Other European 192, Arabic, Gujarati, Afrikaans, Swahili, Russian,
    Bengali...) with 246 people who gave no answer, with no split at any geography. Nothing
    narrower than `other` holds it, and the non-answers cannot be taken out, so the whole cell
    is drawn on `other` (as religiondots draws Mauritius's "Other & Not stated" religion cell).
  * "Total" is the universe row, not a language: resolve() returns None.
"""

NAMES = {
    "Bangla": "indoeuropean.indoaryan.eastern.bengali",
    "Bhojpuri": "indoeuropean.indoaryan.bihari.bhojpuri",
    "Chinese languages": "sinotibetan.sinitic",
    "Creole": "creole.french_based.morisyen",
    "English": "indoeuropean.germanic.english",
    "French": "indoeuropean.romance.french",
    "Hindi": "indoeuropean.indoaryan.central.hindi",
    "Marathi": "indoeuropean.indoaryan.southern.marathi",
    "Tamil": "dravidian.southern.tamil",
    "Telugu": "dravidian.southcentral.telugu",
    "Urdu": "indoeuropean.indoaryan.central.urdu",
    "Other & Not stated": "other",
}


def resolve(label):
    if label == "Total":
        return None
    return NAMES[label]
