"""Nicaragua, Censo 2005: "habla la lengua o idioma del pueblo indigena o comunidad etnica a la
que pertenece" -> node, keyed by the LNG code of sources/ni_censo.py (10 * pueblo + answer).

THE CENSUS NAMES A PEOPLE, NOT A LANGUAGE, as in Argentina and Colombia. A yes to P08 means the
person speaks the language of the pueblo they gave in P07, so each pueblo maps to that people's
language. P08 was asked only of the seven Caribbean-coast peoples. Families from Glottolog
(data/raw/glottolog/languages.csv): Miskito misk1235, Mayangna maya1285 and Ulwa ulwa1239 are
Misumalpan (misu1242); Rama rama1270 is Chibchan; Garifuna gari1256 Arawakan; Nicaragua Creole
English nica1252 an English-based creole.

  Mayagna-Sumu   -> Mayangna. Glottolog's Mayangna is Panamahka plus Tuahka, the two Sumu
                    varieties still spoken; Ulwa, the southern Sumu language, is its own pueblo
                    on the form and its own node here.
  Creole (Kriol) -> Nicaraguan Creole English, the Creole of Bluefields, Pearl Lagoon and Corn
                    Island (Glottolog's Nicaragua Creole English, which also covers Rama Cay
                    Creole).
  Mestizo de la costa del caribe, speaks the language of their community -> Spanish. The
                    community is the Spanish-speaking mestizo population of the two autonomous
                    regions; the form gives them the same question as the indigenous peoples and
                    a yes means Spanish. Drawn `derived` with the rest of Spanish, since the
                    census never names Spanish.

Spanish, tier `derived` (spec §3.5): code 2 (not indigenous), 1 (the Pacific and central
peoples, Xiu-Sutiaba, Nahoa-Nicarao, Chorotega-Nahua-Mange, Cacaopera-Matagalpa, and "Otro",
"No sabe", "Ignorado", none of them asked P08), 3 (P06 not declared), and every x2 (indigenous,
does not speak their people's language). Rama and Garifuna non-speakers mostly speak
Nicaraguan Creole rather than Spanish on the coast; the census does not say so and they are
drawn as Spanish with everyone else (sources/ni.md).

Not drawn: every x9, P08 no answer (11,297), in `gap`.
"""

SPANISH = "indoeuropean.romance.spanish"
MISKITO = "misumalpan.miskito"
MAYANGNA = "misumalpan.mayangna"
ULWA = "misumalpan.ulwa"
RAMA = "chibchan.rama"
GARIFUNA = "arawakan.garifuna"
CREOLE = "creole.english_based.nicaraguan"

PUEBLO_NODE = {1: RAMA, 2: GARIFUNA, 3: MAYANGNA, 4: MISKITO, 5: ULWA, 6: CREOLE,
               7: SPANISH}               # Mestizo de la costa caribe

CODES = {1: SPANISH, 2: SPANISH, 3: SPANISH}
for _p, _node in PUEBLO_NODE.items():
    CODES[10 * _p + 1] = _node             # speaks the language of their people
    CODES[10 * _p + 2] = SPANISH           # does not
    CODES[10 * _p + 9] = None              # no answer: not drawn, `gap`


def resolve(code):
    return CODES[int(code)]
