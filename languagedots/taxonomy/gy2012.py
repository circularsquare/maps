"""Guyana, Census 2012 Table 2.3 ethnic background -> {node: share}. No language question; every
row `derived` (sources/gy.md). Built as Barbados and Trinidad (bb2010.py, tt2011.py).

  White                      English
  Amerindian                 20% on americas_other ("Amerindian language, not named": the census
                             has one Amerindian category, covering Arawakan, Cariban and Warao
                             peoples, so the narrowest node holding all of them), 80% on Guyanese
                             Creole. The 20% is the IDB's Guyana's Indigenous Peoples 2013 Survey
                             (Bollers, Clarke, Johnny, Wenner, 2019, pp. 71-72; 337 households in
                             11 villages): "only 20% of households were fluent in their own
                             language", fluency rising with distance from Georgetown.
  everyone else              Guyanese Creole (creo1235). Indo-Guyanese included: Caribbean
                             Hindustani (cari1275) survives among a few elderly speakers, and no
                             source gives a figure.
"""
CREOLE = "creole.english_based.guyanese"
EN = "indoeuropean.germanic.english"
AMERINDIAN = "americas_other"
AMERINDIAN_RETENTION = 0.20

CODES = {"African / Black": CREOLE, "Chinese": CREOLE, "East Indian": CREOLE, "Mixed": CREOLE,
         "Portuguese": CREOLE, "Other": CREOLE, "White": EN, "Amerindian": AMERINDIAN}


def shares(category):
    if category == "Amerindian":
        return {AMERINDIAN: AMERINDIAN_RETENTION, CREOLE: 1 - AMERINDIAN_RETENTION}
    return {CODES[category]: 1.0}
