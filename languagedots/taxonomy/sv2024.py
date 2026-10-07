"""El Salvador, Censo 2024, "Habla otro idioma aparte del espanol? Cuales?" (people of 3 and
over, several allowed): census label -> node. Labels exactly as sources/sv_censo.py writes them
into data/normalized/sv.csv, which are the BCR's own.

The census prints eight answers besides Spanish and every one gets a node (spec 3.1):
  Inglés, Francés, Italiano      the shared European nodes
  Náhuat -> utoaztecan.pipil     the Nahua language of western El Salvador (Glottolog Pipil,
                                 pipi1250); the census spells it the way its speakers do
  Pisbi -> misumalpan.cacaopera  Pisbi is the BCR's name for Cacaopera (it prints "Pisbi
                                 (Cacaopera)" on the department layer), Misumalpan, caca1247
  Potón -> isolate.lenca_salvador  "Potón (Lenca)" on the department layer: Salvadoran Lenca,
                                 lenc1243
  LESSA -> signlanguage.lessa    "Lengua de señas (LESSA)", Salvadoran Sign Language
  Otro -> other                  any other language, unnamed. The census lists El Salvador's
                                 three indigenous languages by name, so this is mostly foreign
                                 languages, but nothing separates an indigenous answer from a
                                 foreign one inside it, and `other` is the narrowest node that
                                 holds both (spec 3.2).
  Español -> Spanish             someone who named Spanish among their languages anyway; the
                                 question already counts them as Spanish speakers, and
                                 countries/sv.py does not count them twice.

Not languages: "Sí" (persons who speak another language) and "Población" (everyone counted)
are the denominators countries/sv.py shares each person across (spec 3.6); resolve() gives
None for them.
"""

SPANISH = "indoeuropean.romance.spanish"

NAMES = {
    "Inglés": "indoeuropean.germanic.english",
    "Francés": "indoeuropean.romance.french",
    "Italiano": "indoeuropean.romance.italian",
    "Náhuat": "utoaztecan.pipil",
    "Pisbi": "misumalpan.cacaopera",
    "Potón": "isolate.lenca_salvador",
    "LESSA": "signlanguage.lessa",
    "Otro": "other",
    "Español": SPANISH,
    # denominators, not languages
    "Sí": None,
    "Población": None,
}


def resolve(label):
    return NAMES[label]
