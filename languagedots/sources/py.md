# Paraguay: Censo 2002, household language (HOGAR.idiohog)

**Drawn 2026-10-05.** 5,122,472 people (99.2% of the 2002 census), 229 census districts on 224
polygons, 37 census labels on 31 nodes. Files: `sources/py_censo.py`, `taxonomy/py2002.py`,
`taxonomy/tree.d/py.txt`, `countries/py.py`; geography is religiondots' (read-only).

## 1. Which question, and why 2002 rather than 2022

Both censuses are on the same open REDATAM deployment, `prod.redatam.org/binpry/`, bases
`CPV2002` and `CPV2022` (route and traps in `../religiondots/sources/py.md` §2).

| census | variable | what it is |
|---|---|---|
| 2022 | `PERSONA.idiogrn, idiospa, idiopor, idiodeu, idioeng, idiofra, lenind, lenotr, leno` | yes/no per language: languages spoken, several allowed; indigenous languages are one unnamed yes/no |
| 2002 | `PERSONA.P16A-E` | "Lenguas o idiomas que habla", up to five answers: languages spoken, several allowed |
| 2002 | `HOGAR.idiohog` | "Idioma del hogar": ONE language per household, the one spoken most of the time; twenty indigenous languages named |

**Drawn: 2002 `idiohog`.** Spec §1 ranks home language above languages spoken, and the brief
(§2) says a single-answer table from the same office beats a multi-answer one. It also names
every indigenous language, which 2022 does not, and its districts are exactly religiondots'
2002 units. The cost is the vintage (twenty years) and that the answer belongs to the
household: every member is drawn in the household's language. The coverage sweep's note
("2022 person variables have no 'most used' language ... 2002's home-language question is the
better first-language source") is confirmed; 2022's household entity has no language item
either (its `HOGAR` variables are dwelling, assets, deaths, NBI).

The household variable is counted in PEOPLE by the engine's own cross with a person variable
(`CrossTab ITEM=CRUZCOMBI`, `HOGAR.idiohog` x `PERSONA.sexo`), which counts persons. The
household frequency (`FREQHOG`) counts 1,109,536 households instead: Guarani 58.9% of
households, 60.9% of people.

**What 2022 says, for context.** National cross of `idiogrn` x `idiospa` (5,449,521 people
who answered; 379,818 not stated; 280,564 outside the universe):

    speaks both Guarani and Spanish   3,964,584   72.8%
    Guarani, not Spanish                689,118   12.6%
    Spanish, not Guarani                633,719   11.6%
    neither                             162,100    3.0%

Shared 1/k under spec §3.6 that would be roughly Guarani 49%, Spanish 48%. 2002's household
language is 61.4% Guarani and 34.1% Spanish of those drawn. They measure different things (what
a person can speak against what a home runs on), so this is not a trend; but a 2022 map would
show a country split nearly evenly where 2002 shows a Guarani majority. Not built: the
remainder of the 2022 microdata (combinations at district level, `lenind` without names) can
be had from the same engine if a 2022 version is ever wanted.

## 2. Checks (`sources/py_censo.py`, all asserted)

1. 229 districts; every district's label rows sum to its own printed Total, and Varon + Mujer
   = Total on every row.
2. The districts sum to **5,163,198**, the 2002 census population, to the person.
3. A second run by DEPTO (18 departments) equals the district sums on all 295
   (department, label) cells.
4. **All 229 district totals equal religiondots' religion tabulation** (its 10+ Total plus its
   No Aplica, from `religiondots/data/raw/py/py_freq_DISTRITO.html`, read-only) to the person:
   a different variable tabulated separately from the same microdata.

National, people:

    Guarani 3,142,660 (60.87%)   Castellano 1,746,972 (33.84%)   Portugues 124,314 (2.41%)
    Aleman 36,097   Mbya 12,344   Nivacle 11,959   Enlhet Norte 7,174   Ava Guarani 6,998
    Pai Tavytera 6,130   Enxet Sur 3,817   Japones 3,155   Coreano 2,809   Ayoreo 2,077
    Nandeva 1,733   Chino 1,726   Toba 1,449   Ybytoso 1,447   Toba-Qom 1,339   Maka 1,274
    Otros 1,093   Ache 1,057   Ingles 1,053   Sanapana 923   Angaite 780   Guarani Occid. 607
    Manjuy 489   Arabe 423   Frances 276   Italiano 165   Tomaraho 102   Guana 19   Maskoy 8
    NE Otro idioma 3
    not drawn: Viv. Colectivas 40,216; No especificado 303; No habla 156; Psv 51

## 3. Geography

religiondots' `py_hexes.gpkg` and `py_lookup.csv`, unchanged: the REDATAM area codes are the
same codes its religion run used, and the lookup maps all 229. Asuncion's six census districts
(0010-0015) share polygon `0000`; the 26 districts created after 2002 are dissolved into their
2002 parent there (its `py_geo.py`, with its close calls listed in its record). Placement by the
hexes' Kontur `pop` (`pop_weight`). Scatter: 5,109 dots and 9 single weighted dots, no Kontur
cap block, 13,472 people (0.26%) in languages under one dot nationally.

## 4. Mapping calls (`taxonomy/py2002.py` has each one)

- **Guarani** on `guarani.paraguayan` (ar.txt's node, Glottolog para1311).
- **Cross-border names merged onto a neighbour's node**, as spec §3.1 allows for one
  community's name for the same speech: Pai Tavytera on Kaiowa (kaiw1246), Guarani Occidental
  on Eastern Bolivian Guarani (east2555, bo2024's Guarani), Nandeva on Tapiete (tapi1253),
  Manjuy on Chorote (manj1251 is a dialect of Iyojwa'ja Chorote). The census's own names are
  therefore not what the legend shows for these four; `note_public` lists them. The
  alternative was four new sibling nodes, which would colour one people differently on either
  side of a border.
- **TOBA is Toba-Maskoy**, not Qom: the indigenous census lists the Guaicuruan Toba separately
  as TOBA-QOM. **MASKOY** (8 people) merged into it as another name of the same people.
- **GUANA is Paraguay's Guana (Kaskiha)**, Lengua-Mascoy family, not Brazil's Arawakan Guana.
- **New root `enlhet_enenlhet`** (Glottolog Lengua-Mascoy, leng1261): Enlhet Norte, Enxet Sur,
  Sanapana, Angaite, Toba-Maskoy, Guana. A family in its own right, so a root under the
  "families as most readers know them" rule; no ask.
- **Chamacoco**: YBYTOSO and TOMARAHO are Glottolog dialects of Chamacoco (cham1315); each a
  leaf under a new group `zamucoan.chamacoco`, beside Ayoreo.
- **Ache** under Tupi-Guarani (Glottolog ache1246, Tupian).
- **Aleman** on German. Many are Plautdietsch-speaking Mennonites (Boqueron's one 2002 district
  is 25.0% German-household, 9,915 people), but the census says German; bo2024 does the same.
- **Otros, NE Otro idioma** on `other`: every indigenous language is listed by name, so neither
  is an indigenous remainder.

## 5. Colours

New nodes hand-coloured in py.txt: Enlhet-Enenlhet greens (Enlhet Norte green, Enxet Sur
yellow-green, the small ones lighter, darker or teal steps), Chamacoco pinks beside Ayoreo's
magenta, Ache purple. **Four nodes owned by other fragments were moved** and logged in
`taxonomy/COLOURS.md` (2026-10-05, Paraguay), each checked against the country that owns it:
Paraguayan Guarani (ar; 0.088 from the new yellow Spanish, now 0.108), Ava Guarani (br; was
0.016 from Paraguayan Guarani), Mbya (br; would have been 0.053 from it), Nivacle (ar; was
0.056 from German in Boqueron). Left: Portuguese and German are 0.059 apart and both drawn in
the Alto Parana and Itapua colonies; they are shared by every country, so not touched here.
Judged by OKLab distance only, not yet on the map.

## 6. What the map shows

- Guarani households are the rural majority: above 98% in San Pablo, Leandro Oviedo, Maciel,
  3 de Febrero; 93.5% of San Pedro department. Spanish is the capital (79.5% of Asuncion) and
  Central (57.9%).
- Portuguese is the Brazilian frontier: San Alberto 72.0%, Katuete 71.3%, Nueva Esperanza 69.0%,
  Santa Rita 65.0%; 21.3% of Canindeyu, 11.6% of Alto Parana.
- Boqueron (one district in 2002): 25.0% German, 24.7% Nivacle, 9.0% Enlhet Norte, 17.3%
  Guarani. Nueva Germania, the 1887 colony, is still 21.9% German-household.

## Plautdietsch (2026-10-06, session 5d7dac7e-br)

Reverses the "Aleman on German" call above for the Mennonite colonies' districts only: there the
census's "Alemán" is drawn as Plautdietsch (`countries/py.py` `COLONIES`), a place-dependent
split of one label (AGENT_BRIEF §3, India's Pahari). Elsewhere (Itapúa's Hohenau, Obligado,
Bella Vista; Alto Paraná's German-Brazilian settlers, whose German is often Hunsrik; Asunción)
it stays German. 23,901 people moved; German left 12,196.

Districts (2002 codes) and colonies: 1602 Mcal. Estigarribia = all of Boquerón in 2002 (Menno,
Fernheim, Neuland; Wikipedia, Mennonites in Paraguay, colony table "West (Boquerón)"); 1504
Villa Hayes (the Menno colony's land runs into Presidente Hayes; uncited, the weakest of the
list); 0512 Dr. J. Eulogio Estigarribia, Campo 9 (Sommerfeld, Bergthal; GAMEO via search,
"along the highway 210 km east of Asunción"); 0218 Santa Rosa del Aguaray, 0207 Nueva Germania,
0210 Tacuatí (Río Verde, Nuevo México, Santa Clara, Manitoba "distributed between" these three;
Wikipedia, San Pedro Department); 1403 Curuguaty (Nueva Durango, 30 km from Curuguaty, district
Maracaná since; Aguaray Noticias); 0205 Itacurubí del Rosario (Friesland) and 0213 Villa del
Rosario (Volendam) (Wikipedia, Menonitas en Paraguay). Left German though they may hold colony
land: Dr. Juan Manuel Frutos (947), Raúl Arsenio Oviedo (557). Nueva Germania's 909 include the
1887 German colony's descendants, drawn as Plautdietsch with the rest. Witness: Paraguay's
Mennonites number 38,731 in 2022 (Wikipedia's table); 23,901 in 2002 German-speaking households
in these districts, against a 1987 colony total near 16,000 and growth since, is in range.
Fernheim and Neuland families who use Standard German at home are drawn as Plautdietsch too.
