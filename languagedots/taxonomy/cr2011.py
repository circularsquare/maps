"""Costa Rica, Censo 2011: "Habla (nombre) alguna lengua indigena?" -> node, keyed by the LNG code
of sources/cr_censo.py (10 * pueblo + answer; 2 = not indigenous).

THE CENSUS NAMES A PEOPLE AND ASKS ABOUT ANY INDIGENOUS LANGUAGE. P09 is asked only of people who
consider themselves indigenous (P07), after P08 has recorded their pueblo, and it does not ask
which language. That is looser than Nicaragua's or Argentina's question ("the language of your
people"), so a pueblo's yes is drawn on that people's language only where the language is alive
and the pueblo's speakers can only plausibly be speaking it. Families from Glottolog
(data/raw/glottolog/languages.csv): every Costa Rican indigenous language is Chibchan
(chib1249).

  Bribri 11            Bribri (brib1243). 8,203 speakers.
  Cabecar 31           Cabecar (cabe1245). 12,596; the largest indigenous language in the
                       country, as INEC says in its own report.
  Ngobe o Guaymi 71    Ngabere (ngab1239), the Ngabe language of Panama and southern Costa Rica.
  Maleku o Guatuso 61  Maleku Jaika (male1297). 441; the Guatuso territory reads 67.5% speakers in
                       INEC's CUADRO 3, which fits a small living language.

  Brunca o Boruca 21, Teribe o Terraba 81 -> `chibchan`, the family, drawn as "language not
                       named". Boruca's last documentation in Glottolog is 2010 and Costa Rican
                       Teribe has a handful of speakers; INEC's own CUADRO 3 has the Boruca and
                       Curre territories at 5.9% and 4.4% speakers and Terraba at 9.9%, against
                       60-97% in the Bribri, Cabecar and Ngabe territories. The 325 Brunca and 176
                       Teribe who said yes may speak their heritage language as learners, or
                       Bribri, Cabecar or Ngabere, their neighbours in Buenos Aires canton, all
                       Chibchan. The census does not say which, so the family is the narrowest
                       node holding all of them (AGENT_BRIEF §3, the unnamed-remainder rule).
  Chorotega 41, Huetar 51 -> `americas_other`. Chorotega (an Oto-Manguean language) and Huetar
                       went out of use long ago; the 179 and 56 who said yes speak some other
                       indigenous language, which could be from any family.
  De otro pais 91      -> `americas_other`: indigenous people of another country (Ngabe from
                       Panama, Miskitu from Nicaragua and others); 1,904 speakers, language not
                       named.
  Ningun pueblo 101    -> `americas_other`: indigenous, no pueblo; 1,343 speakers.

Spanish, tier `derived` (spec §3.5): code 2 (not indigenous, not asked: 4,197,569) and every x2
(indigenous, does not speak an indigenous language: 72,457). Costa Rica's other minority
languages, Limon Creole English above all, were never asked about and are inside the Spanish
here (sources/cr.md, note_public).
"""

SPANISH = "indoeuropean.romance.spanish"
AM = "americas_other"
CHIBCHAN = "chibchan"
BRIBRI = "chibchan.bribri"
CABECAR = "chibchan.cabecar"
NGABERE = "chibchan.ngabere"
MALEKU = "chibchan.maleku"

PUEBLO_NODE = {1: BRIBRI, 2: CHIBCHAN, 3: CABECAR, 4: AM, 5: AM, 6: MALEKU, 7: NGABERE,
               8: CHIBCHAN, 9: AM, 10: AM}

CODES = {2: SPANISH}
for _p, _node in PUEBLO_NODE.items():
    CODES[10 * _p + 1] = _node             # speaks an indigenous language
    CODES[10 * _p + 2] = SPANISH           # does not


def resolve(code):
    return CODES[int(code)]
