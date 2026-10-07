"""Honduras, Censo 2013: self-identified people (P05, P06) -> language node, keyed by the GRP
code of sources/hn_censo.py. NO LANGUAGE QUESTION: each people is read as its language under
AGENT_BRIEF section 2's ethnicity rule (Anita, 2026-10-05). Every row is `derived`.

Retention, people by people (sources/hn.md section 2 has the sources):

  Miskito 3, Tawahka 7, Pech 5, Garifuna 8, Negro de habla inglesa 9 -> their own language.
      Carlos Palacios (UNAH, "Pueblos indigenas y negros de Honduras") describes each as still
      speaking it. No source gives a retention share that can be used: ENDESA-MICS 2019 asks
      the respondent's native language (Spanish, English, Miskito, Garifuna only), but it
      records 2 of 101 Black English-speaking Bay Islanders and 0 of 55 Garifuna women in
      Atlantida as native speakers, which no other account supports; it is set aside and its
      figures are in the record.
  Tolupan 6 -> Tol in Orica (0814) and Marale (0811), the Montana de la Flor, where Tol is still
      spoken; Spanish elsewhere (Yoro, where most Tolupan live, has shifted to Spanish;
      Palacios: Tol "mas fuertemente arraigad[o] en las tribus de la Montana de la Flor").
  Lenca 2, Nahua 4, Maya-Chorti 1 -> Spanish. Lenca died out around 1900 (Palacios, after
      Herranz); the Nahua "no conservan su lengua" (Palacios); Chorti speakers in Honduras are
      "muy pocos" and mostly from Guatemala (Palacios).
  Otro pueblo 10, Mestizo 14, Blanco 15, Otro 16 -> Spanish.

Families from Glottolog (data/raw/glottolog/languages.csv): Miskito misk1235 Misumalpan; Tawahka
a Mayangna (maya1285) variety, its own node since the census names the people; Pech pech1241
Chibchan; Tol toll1241 Jicaquean (jica1245, a family of one living language: own root);
Garifuna gari1256 Arawakan; Bay Islands English is not separate in Glottolog (Western Caribbean
Creole, west2854, is the grouping); its own node under English-based creoles.
"""

SPANISH = "indoeuropean.romance.spanish"
MISKITO = "misumalpan.miskito"
TAWAHKA = "misumalpan.tawahka"
PECH = "chibchan.pech"
TOL = "jicaquean.tol"
GARIFUNA = "arawakan.garifuna"
BAY_ISLANDS = "creole.english_based.bay_islands"

TOL_MUNICIPIOS = {"0814", "0811"}       # Orica, Marale: the Montana de la Flor

CODES = {1: SPANISH, 2: SPANISH, 3: MISKITO, 4: SPANISH, 5: PECH, 6: SPANISH, 7: TAWAHKA,
         8: GARIFUNA, 9: BAY_ISLANDS, 10: SPANISH, 14: SPANISH, 15: SPANISH, 16: SPANISH}
EXTRA_NODES = [TOL]


def resolve(code, municipio=None):
    code = int(code)
    if code == 6 and municipio in TOL_MUNICIPIOS:
        return TOL
    return CODES[code]
