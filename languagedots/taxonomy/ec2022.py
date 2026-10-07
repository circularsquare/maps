"""Ecuador, Censo 2022, "Idiomas o lenguas que habla o se comunica" (languages a person speaks or
communicates in, several allowed, age 1 and over): census label -> node. Labels exactly as
sources/ec_censo.py writes them into data/normalized/ec.csv, which are INEC's own spellings.

The census asks which classes a person speaks (an indigenous language, Castellano, a foreign
language, Ecuadorian Sign Language) and, for an indigenous language, which one: 14 named and
"Otras Lenguas Indigenas". Every named language gets a node (spec 3.1). Reuses:
  Kichwa -> quechuan.kichwa          pe.txt's Kichwa: the Quechua of Ecuador and of Peru's
                                     Loreto and San Martin, one name across the border
  Achuar Chicham -> chicham.achuar   pe.txt
  A'Ingae -> isolate.cofan           br.txt / co.txt, Glottolog cofa1242
  Awapit -> barbacoan.awa_pit        co.txt, Glottolog awac1239 (CO, EC)
  Bai Coca -> tucanoan.siona         Ecuadorian Siona is called Baicoca by its speakers
  Paaikoka -> tucanoan.secoya        Paicoca, the Secoya's name for their language (Siona and
                                     Secoya are close; INEC prints them apart, so are they here)
  Siapedee -> chocoan.embera.eperara the Epera of Esmeraldas, the Eperara Siapidara of
                                     Colombia's Pacific coast (Glottolog Epena, epen1239)
Remainders:
  "Otras Lenguas Indigenas"          americas_other: indigenous languages INEC did not name
                                     (25,566 speakers), any family. Kept apart from `other`.
  "Idioma extranjero"                other: any foreign language, unnamed.
Not drawn (gap): "No habla/No se comunica"; under-1s, who are not asked.
"""

NAMES = {
    "Castellano o Español": "indoeuropean.romance.spanish",
    "Idioma extranjero": "other",
    "Lengua de señas ecuatoriana": "signlanguage.lsec",
    "Otras Lenguas Indigenas": "americas_other",
    "Kichwa": "quechuan.kichwa",
    "Shuar Chicham": "chicham.shuar",
    "Achuar Chicham": "chicham.achuar",
    "Shiwiar Chicham": "chicham.shiwiar",
    "Andwa Pukwano": "zaparoan.andoa",
    "Sapara": "zaparoan.zaparo",
    "Awapit": "barbacoan.awa_pit",
    "Chaa´palaa": "barbacoan.chachi",
    "Tsa´Fiki": "barbacoan.tsafiki",
    "A´Ingae": "isolate.cofan",
    "Wao Tededo": "isolate.waorani",
    "Bai Coca": "tucanoan.siona",
    "Paaikoka": "tucanoan.secoya",
    "Siapedee": "chocoan.embera.eperara",
    # not drawn
    "No habla/No se comunica": None,
    "Menores de 1 año (no se pregunta)": None,
}


def resolve(label):
    return NAMES[label]
