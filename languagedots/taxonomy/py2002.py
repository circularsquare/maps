"""Paraguay, Censo Nacional de Poblacion y Viviendas 2002, HOGAR.idiohog, "Idioma del hogar":
the language spoken in the household most of the time, every member of the household counted
(sources/py_censo.py). Census label -> node. Labels exactly as the REDATAM engine prints them
into data/normalized/py.csv.

Every label the census prints as a language gets a node (spec 3.1). Families from Glottolog
(data/raw/glottolog/languages.csv); see taxonomy/tree.d/py.txt.

Calls worth knowing:
  Guarani -> tupian.tupiguarani.guarani.paraguayan, the node ar.txt made for Paraguayan Guarani
      (Glottolog para1311), not the Guarani group (that would draw it as "language not named").
  Aleman -> German, as the census names it. Many of its 36,097 are Mennonites of the Chaco and
      eastern colonies, whose home language is Plautdietsch; the census does not say so, so the
      dots stay on German (bo2024 does the same with Bolivia's colonies).
  Cross-border names of one people's language, merged onto the node a neighbour's census made
  (spec 3.1 allows a merge where the variant is one community's name for the same speech):
    PAI-TAVYTERA -> guarani.kaiowa. Pai Tavytera is the Paraguayan name of the Kaiowa
        (Glottolog kaiw1246, Kaiwa, AR;BR;PY); Brazil's census calls them Guarani Kaiowa.
    GUARANI OCCIDENT -> guarani.chiriguano. Paraguay's Guarani Occidental (Guarayo) of the
        Chaco speak Eastern Bolivian Guarani (Glottolog east2555, AR;BO;PY), the language bo2024
        draws as Bolivia's Guarani.
    ÑANDEVA -> guarani.tapiete. Paraguay's Guarani Nandeva are the Tapiete of Bolivia and
        Argentina (Glottolog tapi1253, AR;BO;PY). Not Brazil's "Nhandeva", which is Chiripa.
    MANJUY -> matacoan.chorote. The Manjui are the Paraguayan Iyojwa'ja Chorote (Glottolog
        manj1251, a dialect of Iyojwa'ja Chorote), ar.txt's Chorote.
    AVA-GUARANI -> guarani.ava_guarani, br.txt's Ava Guarani (Glottolog chir1286, Chiripa).
  TOBA -> enlhet_enenlhet.toba_maskoy. In Paraguay's indigenous census "Toba" without a suffix
      is the Toba-Maskoy people (Glottolog toba1268, Toba-Enenlhet), and the Guaicuruan Toba are
      listed separately as TOBA-QOM -> guaicuruan.toba.
  MASKOY (8 people) -> toba_maskoy too: the Toba-Maskoy are also called simply Maskoy, and INE
      later named the people that way; a spelling variant of TOBA's answer, not the family name.
  GUANA -> enlhet_enenlhet.guana, Paraguay's Guana (Kaskiha; Glottolog guan1268, in
      Lengua-Mascoy), not the Arawakan Guana/Terena of Brazil.
  YBYTOSO, TOMARAHO -> two leaves under zamucoan.chamacoco (Glottolog dialects of Chamacoco).
  Chino -> sinotibetan.sinitic, as bo2024 and pl2021: "Chinese" names no variety.
Remainders:
  "Otros", "NE Otro idioma" -> other. Every indigenous language is listed by name, so neither is
  an indigenous remainder.
Not drawn (gap): "Viv. Colectivas" (people in collective dwellings: the question is asked of a
household), "No especificado", "No habla", "Psv".
"""

RO = "indoeuropean.romance"
GC = "indoeuropean.germanic.continental"
GU = "tupian.tupiguarani.guarani"
EE = "enlhet_enenlhet"

NAMES = {
    "Guaraní": f"{GU}.paraguayan",
    "Castellano": f"{RO}.spanish",
    "Portugués": f"{RO}.portuguese",
    "Alemán": f"{GC}.german",
    "Inglés": "indoeuropean.germanic.english",
    "Francés": f"{RO}.french",
    "Italiano": f"{RO}.italian",
    "Japonés": "japonic.japanese",
    "Chino": "sinotibetan.sinitic",
    "Coreano": "koreanic.korean",
    "Arabe": "afroasiatic.arabic",
    # Tupi-Guarani
    "ACHE": "tupian.tupiguarani.ache",
    "AVA-GUARANI": f"{GU}.ava_guarani",
    "MBYA": f"{GU}.mbya",
    "PAI-TAVYTERA": f"{GU}.kaiowa",
    "GUARANI OCCIDENT": f"{GU}.chiriguano",
    "ÑANDEVA": f"{GU}.tapiete",
    # Enlhet-Enenlhet (Lengua-Maskoy)
    "ENLHET NORTE": f"{EE}.enlhet_norte",
    "ENXET SUR": f"{EE}.enxet_sur",
    "SANAPANA": f"{EE}.sanapana",
    "ANGAITE": f"{EE}.angaite",
    "TOBA": f"{EE}.toba_maskoy",
    "MASKOY": f"{EE}.toba_maskoy",
    "GUANA": f"{EE}.guana",
    # Matacoan
    "NIVACLE": "matacoan.nivacle",
    "MAKA": "matacoan.maka",
    "MANJUY": "matacoan.chorote",
    # Zamucoan
    "AYOREO": "zamucoan.ayoreo",
    "YBYTOSO": "zamucoan.chamacoco.ybytoso",
    "TOMARAHO": "zamucoan.chamacoco.tomaraho",
    # Guaicuruan
    "TOBA-QOM": "guaicuruan.toba",
    # remainders
    "Otros": "other",
    "NE Otro idioma": "other",
    # not drawn
    "Viv. Colectivas": None,
    "No especificado": None,
    "No habla": None,
    "Psv": None,
}


def resolve(label):
    return NAMES[label]
