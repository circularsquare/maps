# Costa Rica: the record

Drawn 2026-10-04 (agent d9e44929-cr). 4,301,712 people on 472 distritos; 31,686 indigenous people
who speak an indigenous language, the rest drawn as Spanish (`derived`).

## Source

INEC, X Censo Nacional de Población y VI de Vivienda 2011, tabulated on INEC's own REDATAM server
(`https://sistemas.inec.cr:8443/bininec/`, base CP2011, open, no login). The 2022 census's
indigenous results are not published (coverage sweep, IWGIA 2025).

- **The certificate chain on sistemas.inec.cr is incomplete** (religiondots/sources/cr.md met it
  as curl exit 60). Python's certifi bundle fails; Windows completes the chain from the AIA link
  and caches the intermediate, so `sources/cr_censo.py` verifies against the Windows store
  (`ssl.create_default_context()`, urllib). If it fails, one `Invoke-WebRequest` to the base URL
  fills the cache. TLS verification is never turned off.
- The questions (boleta, printed p54 of the Territorios Indígenas volume): P07 "¿Se considera
  indígena?", P08 which pueblo (Bribri, Brunca o Boruca, Cabécar, Chorotega, Huetar, Maleku o
  Guatuso, Ngöbe o Guaymí, Teribe o Térraba, de otro país, ningún pueblo), P09 "¿Habla (nombre)
  alguna lengua indígena?". **P09 is asked only of P07 yes** (104,143; everyone else is No Aplica)
  **and asks about ANY indigenous language, not the pueblo's.** No row is missing P07 or P09.
- One Redatam program defines LNG = 10 x P08 + P09 (2 for not indigenous) and tabulates it per
  distrito and province. Programs are saved beside their output in `data/raw/cr/`. The dictionary
  is at `RpWebStats.exe/Dictionary?BASE=CP2011&ITEM=DICALL&lang=esp`.
- Printed witness: INEC, "X Censo ... 2011: Territorios Indígenas, principales indicadores
  demográficos y socioeconómicos" (2013), `data/raw/cr/territorios_indigenas_2011.pdf`, from
  inie.ucr.ac.cr (inec.cr itself is an Akamai wall).

`python sources/cr_censo.py --fetch` runs nine programs (~1 min), writes `data/normalized/cr.csv`.

## Checks (numbers from the run)

1. 472 distritos, REDATAM's own list; codes sum to 4,301,712, the 2011 population.
2. The distritos rebuild the 7 provinces on every code (separate query).
3. Per distrito, LNG agrees with separate tabulations of raw P07, P08 and P09: 0 failures in 472.
   This also pins P08's code order to its labels.
4. Distrito sums equal the national P08 x P09 crosstab per pueblo; 104,143 indigenous and 26,070
   of no pueblo, as printed (p34).
5. CUADRO 3 (p35): the share of indigenous people speaking an indigenous language in each of the
   24 territories, one decimal, reproduced exactly by a TERRINDI x P09 crosstab (Chirripó 96.7,
   Salitre 53.4, Boruca 5.9, Térraba 9.9, Matambú 0.4 ...).

National, speak / do not: Cabécar 12,596 / 4,389; Bribri 8,203 / 9,995; Ngöbe 6,463 / 3,080;
Maleku 441 / 1,339; Brunca 325 / 5,230; Chorotega 179 / 11,263; Teribe 176 / 2,489; Huetar 56 /
3,405; de otro país 1,904 / 6,540; ningún pueblo 1,343 / 24,727. 10,104 speakers (32%) live
outside the territories.

## Calls

- **Pueblo to language** (`taxonomy/cr2011.py`). Because P09 asks about any indigenous language,
  a yes is drawn on the pueblo's language only where that language is alive and is the only
  plausible one: Bribri (brib1243), Cabécar (cabe1245), Ngäbere (ngab1239), Maleku Jaíka
  (male1297), all Chibchan, under co.txt's `chibchan` root. Their speakers sit where expected
  (Bribri: Telire 3,720, Bratsi 1,209, Buenos Aires 1,494; Cabécar: Valle La Estrella 4,997,
  Chirripó 3,517; Ngöbe: Sixaola 1,440, Limoncito 975, Pavón 740; Maleku: San Rafael de Guatuso
  348 of 441).
- **Brunca and Teribe speakers on `chibchan`** (drawn as "language not named"). Glottolog's last
  documentation of Boruca is 2010 and Costa Rican Teribe has a handful of speakers; CUADRO 3 puts
  the Boruca, Curré and Térraba territories at 5.9%, 4.4% and 9.9% speakers. The 325 Brunca yes
  answers are scattered (Boruca 93, then Buenos Aires, Palmar, Sabalito), and the 176 Teribe
  ones are mostly in Potrero Grande (72), which holds Térraba beside Bribri and Cabécar land.
  They may be learners of their heritage language or speakers of Bribri, Cabécar or Ngäbere;
  all Chibchan, so the family is the narrowest node. Reversal: map 21 and 81 to their own nodes
  (Glottolog boru1252, teri1250), 501 people.
- **`americas_other`**: Chorotega (179) and Huetar (56) yes answers, whose own languages are long
  gone; "de otro país" (1,904) and "ningún pueblo" (1,343). Of "de otro país", Sabalito 375 and
  Sixaola 193, both on the Panama border, are very probably Panamanian Ngäbe (and Bribri in
  Sixaola), but the census does not say so and they stay unnamed.
- **Spanish, `derived`**: everyone not indigenous (not asked) and every indigenous person who
  said no. Limón Creole English (Mekatelyu) on the Caribbean coast, the Chinese and other
  immigrant languages are not recorded and are inside the Spanish; said in note_public. No
  second source for the remainder was found: INEC's ENAHO has no language question
  (religiondots/sources/cr.md swept the NADA catalogue, 594 variables); not chased further.
- **No gap**: no row lacks P07 or P09.
- **Colours** (`taxonomy/tree.d/cr.txt`): Chibchan blue-violet. Cabécar, the biggest, the
  saturated middle (0.66 0.16 265); Bribri, its neighbour in Talamanca and Buenos Aires, lighter
  and pinker (0.80 0.12 310); Ngäbere darker and bluer (0.58 0.14 235); Maleku a pale violet.

## Geography (`sources/cr_geo.py`)

COD-AB Costa Rica admin3 (valid_on 2024-12-03, 492 distritos), read-only from religiondots'
download with `engine="fiona"` (pyogrio reads zero features from this .gdb). Religiondots draws
Costa Rica by province only, so the units are built here and written to
`data/geo/cr/cr_units.gpkg`.

- **The 2011 distritos are rebuilt from 2024's.** 468 codes are shared. Río Cuarto (20306),
  Monteverde (60109) and Puerto Jiménez (60702) became cantons and are renumbered (21601-3,
  61201, 61301). Seventeen distritos created since 2011 go back to their parents: one parent
  for most; Quitirrisí (four of Mora's), Caldera (Espíritu Santo, San Juan Grande) and Gutiérrez
  Braun (San Vito, Sabalito) are cut on a 250 m lattice by nearest parent, weighted so each
  parent gets back the land it lost (the census base's EXTTER area against COD's); La Colonia
  (Pococí) by plain nearest parent, because Pococí's lines also moved. Isla del Coco (no
  residents) is left out: 2011's Puntarenas area excludes it.
- **Join on code, witnessed by area.** REDATAM labels 60111-60116 one place out (60111
  "Chacarita" is 316.6 km2, Cóbano), so names are not a key. Every rebuilt distrito's area is
  compared with EXTTER: median ratio 1.004, 456 of 472 within 15% (or 2 km2), against at most
  7.6% over 200 shuffles. The 16 outside are lines moved between existing distritos since 2011,
  pinned in `AREA_MOVED` (Pococí-Guácimo ~220 km2 of lowland; Orosi against Tayutic, Santa Rosa
  and Pejibaye; inside San Carlos and Sarchí; La Sierra and Unión).
- **COD swaps Puriscal's 10405 and 10408.** COD calls 10405 "San Antonio"; the official order
  and the census have 10405 San Rafael, 10408 San Antonio. The polygons follow COD's names:
  Kontur (at the national ratio) puts 3,927 in COD's 10405 against the census's 3,889 for San
  Antonio. `POLYGON_FOR` swaps them; the witness is asserted (2,146 / 1,730 and 3,903 / 3,889).
  Worth telling religiondots if it ever goes below province.
- **Placement**: `_grid.py`'s hex_layer on Kontur CR 2023 (copied from religiondots' raw .gz into
  `data/geo/kontur/`). Of 559 hexes outside every distrito, 156 (14,797 people) are inside
  Natural Earth's Nicaragua or Panama and dropped (Paso Canoas and the like); 398 (32,206) on the
  coast are snapped within 2 km (Puntarenas's sandspit 7,683, Limón 4,526); 5 (66) further out
  dropped. San Francisco de Goicoechea (10802, 0.6 km2) holds no hex centroid and is placed on its
  own polygon. Kontur / census per distrito, normalised: p10 0.83, median 1.00, p90 1.25 (hex_layer's
  shuffle control is skipped because 10802 has no hex; the area witness above does that job).

## Scatter

4,299 dots at 1:1000 and 2 rings (Maleku 441 at Guatuso, `chibchan` 501 near Buenos Aires).
Median dot positions: Cabécar -83.13, 9.74 (Chirripó to Talamanca), Bribri -83.01, 9.52
(Talamanca), Ngäbere -82.95, 8.87 (Coto Brus). No Kontur cap block stopped the scatter.

## Files

`sources/cr_censo.py`, `sources/cr_geo.py`, `taxonomy/cr2011.py`, `taxonomy/tree.d/cr.txt`,
`countries/cr.py`, `data/raw/cr/` (9 tables + programs, territorios_indigenas_2011.pdf),
`data/normalized/cr.csv`, `data/geo/cr/cr_units.gpkg`, `cr_hexes.gpkg`,
`data/geo/kontur/kontur_population_CR_20231101.gpkg(.gz)`, `data/processed/dots_cr.geojson`,
`rings_cr.geojson`.

## Immigrant and Creole languages (2026-10-05, session edd42a8c-latn)

Out of the Spanish remainder, all `derived`: Spanish 4,270,026 -> 4,230,559. Limonese Creole
18,142, English 12,307, Mandarin 1,629, German 1,305, French 773, Italian 771, Portuguese 508,
Dutch 373, Min Nan 328 (Taiwan), Russian 322, Cantonese 234. 4,291 dots, 202 rings. Method and
retention rule: `sources/mx.md`, "Immigrant and settler languages" §2 (`sources/latam_immig.py`).

Files: `sources/cr_imm.py`, `sources/iso_numeric.py` (ISO numeric -> alpha-2 from Natural Earth),
`data/raw/cr/cr_{nat_paiscode,nat_pais,dist_etnia,nat_etnia,dist_paisx0-3}.htm` (+ programs),
`data/normalized/cr_imm.csv`; `countries/cr.py`, `taxonomy/tree.d/cr.txt` changed.

- **Country of birth**: P05C LUGPA, ISO 3166 numeric (INEC's 822-824 are Scotland, Wales,
  England). 385,899 foreign-born; 138 non-Spanish-speaking countries hold 31,880 (US 15,898,
  China 3,281, Canada 1,679, Italy 1,494, Germany 1,412, France 936, Taiwan 797, Brazil 605).
  An AREALIST over LUGPA answers HTTP 500 (it makes a column per value of the range), so a
  derived PAISX numbers the countries 1..k, in chunks of 35. Checks pass: per country the
  distrito sums equal the national frequency; each chunk's zero column plus its countries
  equals 4,301,712. Nicaraguans (the big group) and every Spanish-speaking origin stay Spanish.
  US-born are drawn at the US mix with TeO2 retention, not split by age as Mexico's are: Costa
  Rica's are mostly North American retirees and settlers, by the distritos they live in.
- **Limonese Creole (Mekatelyu)**: P10 ETNIA "Negro(a) o afrodescendiente" in Limón province,
  18,142, each read as a speaker (AGENT_BRIEF §2's ethnicity rule; no retention share found).
  Mulatos (33,202 in Limón) and Black identifiers elsewhere (27,086, many Limón-born in San José,
  some Panamanian or Nicaraguan) stay Spanish. Ethnologue's only count is 55,000 (1986, native
  and second-language), which this does not try to match. P10 was not asked of the 104,143
  self-identified indigenous; ETNIA's distritos sum to its national frequency on all seven
  answers. New node `creole.english_based.limonese` (no Glottolog entry of its own; Jamaican,
  jama1262, nearest), generated colour.
- "Chino(a)" (9,170, mostly Costa Rica-born) stays Spanish; only the China-, Taiwan- and
  Hong Kong-born are drawn on Chinese languages, at China's home mix (no source for the
  Cantonese share of Costa Rica's Chinese).
