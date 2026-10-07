# Nicaragua: the record

Drawn 2026-10-04 (agent d9e44929-ni). 5,130,801 people drawn on 153 municipios; 142,151
speakers of an indigenous or Creole language, the rest drawn as Spanish.

## Source

INIDE, VIII Censo de Población y IV de Vivienda 2005, tabulated on INIDE's own REDATAM server
(redatam.inide.gob.ni, base VIVPOB05, open, no login), the route religiondots found for religion
(`../religiondots/sources/ni.md` §1). No census since 2005.

- The questions, asked of everyone, all ages: P06 identifies with an indigenous people or ethnic
  community (yes 443,847 / no 4,536,003 / not declared 162,248); P07 which of thirteen (plus
  ignorado); P08 "¿Habla la lengua o idioma del pueblo indígena o comunidad étnica a la que
  pertenece?", asked **only of the seven Caribbean-coast peoples** (P07 1-7: Rama, Garífuna,
  Mayangna-Sumu, Miskitu, Ulwa, Creole (Kriol), Mestizo de la costa caribe). The 172,977 Pacific
  and central people (Xiu-Sutiaba, Nahoa-Nicarao, Chorotega-Nahua-Mange, Cacaopera-Matagalpa,
  Otro, No sabe, Ignorado) are all "No aplica" on P08. The language is implied by the people.
- One Redatam program defines LNG per person (10 x pueblo + answer, or 1/2/3 for the people not
  asked) and tabulates it per municipio and department. Programs are saved beside their output
  in `data/raw/ni/`. The VARLIST in the CmdSet page lists the variables.
- Printed volume: Vol. I "Población: Características Generales" (2006), `data/raw/ni/volI.pdf`,
  CUADRO 10 (pueblo by department, printed p184) and CUADRO 11 (speakers by department, pueblo
  and age, p189). Vol. IV (municipios) has no pueblo or language table.

`python sources/ni_censo.py --fetch` runs seven programs (~1 min) and writes
`data/normalized/ni.csv`.

## Checks (numbers from the run)

1. 153 municipios; codes sum to 5,142,098, the 2005 census population; every person once.
2. The 153 municipios rebuild the 17 departments on every code (separate query).
3. Per municipio, LNG agrees with separate tabulations of the raw P06 and P08: code 2 = P06 no,
   code 3 = P06 not declared, code 1 + all pueblo codes = P06 yes, and the speaks / does not /
   no-answer sums = P08's columns. All 153, 0 failures.
4. Per pueblo, municipio sums equal the national P07 x P08 crosstab.
5. Printed witness, typed from the PDF: all 14 pueblo totals of CUADRO 10 and 24 speaker figures
   of CUADRO 11 (national 244,305 and per pueblo; R.A.A.N. 165,669 and all seven pueblos;
   R.A.A.S. 65,069; Managua 2,843 and all seven) reproduced exactly.

National: Miskitu 113,855 speak / 5,219 do not / 1,743 no answer; Mestizo de la costa 102,154 /
6,711 / 3,388; Creole 18,420 / 877 / 593; Mayangna 8,537 / 512 / 707; Rama 744 / 1,212 / 2,229;
Garífuna 512 / 728 / 2,031; Ulwa 83 / 9 / 606.

## Calls

- **Pueblo to language** (`taxonomy/ni2005.py`): Miskitu, Mayangna-Sumu (Glottolog Mayangna,
  Panamahka + Tuahka) and Ulwa under a new root **Misumalpan** (misu1242); Rama under Chibchan
  (co.txt's root); Garífuna on gt.txt's `arawakan.garifuna`; Creole (Kriol) as Nicaraguan Creole
  English (nica1252) under English-based creoles.
- **Mestizo de la costa caribe, "speaks the language of their community", is Spanish.** The
  coast's mestizos are Spanish speakers; the form asked them the same question. Drawn `derived`
  with the rest of Spanish since the census never names Spanish.
- **Spanish, `derived`**: not indigenous, P06 not declared, the Pacific and central peoples (not
  asked; their languages went out of use long ago), and every non-speaker of the seven. Many Rama
  and Garífuna non-speakers speak Creole English, not Spanish (Rama Cay, Orinoco and Pearl
  Lagoon); the census does not say so, so they follow the spec §3.5 rule. Said in note_public.
- **Not drawn**: P08 no answer, 11,297, in `gap`. High for the small peoples (Rama 53%,
  Garífuna 62%, Ulwa 87%) because most of those no-answers are self-identifications scattered
  across the Pacific (Managua alone: 600 Rama, 373 Garífuna, 100 Ulwa), where the follow-up was
  evidently not put; the speakers themselves sit on the coast (Rama: El Rama 244, Bluefields
  132; Garífuna: Laguna de Perlas 325). Ulwa speakers are only 83, scattered (Bluefields 25,
  Desembocadura de Río Grande 9, where Karawala is); drawn as recorded.
- **Second source for the Spanish remainder**: none. The same server carries a dictionary for a
  later base (entities SEGCEN/VIV/HOG/POB, a 2018-dated questionnaire with the same pueblo and
  language questions, P07-P09), but its data files are not on the server ("INIDED2.ptr ... no
  fue encontrado", path `...\cpv18\BaseR\`). Probably a 2018 census pilot; worth one look if
  INIDE ever publishes it.
- **Geography**: religiondots' Nicaragua layer, read-only: `ni_lookup.csv` (INIDE code -> COD
  p-code, joined on name because ten codes were renumbered; see religiondots/sources/ni_geo.py)
  and its 47,270 Kontur hexes, population-weighted. Same 153 units, no new geography.
- **Colours**: Misumalpan sky blue (h 225): Miskito saturated, Mayangna paler and bluer (they
  border in Waspám, Bonanza, Siuna), Ulwa darker and greener. Away from Spanish's red-orange and
  the creole and Garífuna greens. Rama and Creole take generated colours near their groups.

## Scatter

5,127 dots at 1:1000, 3 rings (Rama, Garífuna, Ulwa under one dot); 3,801 people under one dot
nationally. Median dot positions: Miskito -83.91, 14.14 (Puerto Cabezas and Waspám), Mayangna
-84.62, 14.03 (Bonanza), Creole -83.78, 12.09 (Bluefields and Pearl Lagoon). Three coastal
units over 95% sea left unclipped by water.py, as in religiondots.

## Files

`sources/ni_censo.py`, `taxonomy/ni2005.py`, `taxonomy/tree.d/ni.txt`, `countries/ni.py`,
`data/raw/ni/` (7 tables + programs, volI.pdf), `data/normalized/ni.csv`,
`data/processed/dots_ni.geojson`, `rings_ni.geojson`.

## Immigrant languages (2026-10-05, session edd42a8c-latn)

Out of the Spanish remainder, `derived`: Spanish 4,988,650 -> 4,986,178. English 1,084, German
232, Russian 124, French 107, then Italian, Portuguese, Chinese, Antiguan Creole and smaller.
5,126 dots, 111 rings. Method and retention rule: `sources/mx.md`, "Immigrant and settler
languages" §2 (`sources/latam_immig.py`). Creole English and Garifuna were already drawn from
the census's own question; no source counts the Rama and Garifuna who speak Creole instead, so
they stay Spanish.

Files: `sources/ni_imm.py`, `data/raw/ni/ni_nat_pais{lab,cod}.htm`, `ni_mun_paisx.htm`
(+ programs), `data/normalized/ni_imm.csv`; `countries/ni.py`, `taxonomy/tree.d/ni.txt`
(borrowed-node block).

- P09A = 3 (mother lived abroad at the birth: 34,693) and P09B, the country. P09B prints names;
  a plain-integer copy prints codes; the two pair 118 of 118 rows with equal counts. Honduras
  10,745, Costa Rica 9,343, El Salvador 2,121 and the other Spanish-speaking origins stay Spanish,
  with regional remainders and "Ignorado" (1,732). Non-Spanish-speaking: 5,462, of whom US
  3,085 (1,822 under 18, drawn as Spanish as in Mexico), Germany 261, Canada 219, Russia 169,
  Italy 166, France 130, China 121, Brazil 107, Antigua 104. Checks pass: 153 municipios; per
  country, municipio sums equal the national frequency; every person once.
