# Bolivia: Censo 2024, mother tongue, municipality

**Drawn.** 10,509,185 people on 339 municipality polygons (343 census areas), 77 nodes, 10,489
dots at 1:1000, 57 rings.

## Source

- INE Bolivia, Censo de Población y Vivienda 2024. Variable `PERSONA.IDIOMAT`, question 34.1
  "Primer idioma o lengua en el que aprendió a hablar en su niñez": the first language learned as
  a child, asked of everyone. That is mother tongue in the brief's sense.
- Route: INE's own REDATAM webserver, `https://redatam.ine.gob.bo/binbol/`, base `PHCCEN24ESPV1`,
  the open route religiondots already uses for Bolivia's 2024 count
  (`../religiondots/sources/bo_census.py`). No login, no terms gate, generic browser UA. The
  server sends an incomplete certificate chain, so verification is off.
- PROGRED (free Redatam programs) works on this server but refuses AREALIST and CROSSTABS on
  this variable ("Too many categories": `IDIOMAT` is a recode of the sparse code list
  `P341_IDIOMAT_COD`). So every table is the web form's FREQUENCY with AREABREAK, one labelled
  table per area. Nothing is paired by position, which removes the trap Peru's second endpoint
  was there to catch.
- `sources/bo_censo.py --fetch` runs five queries: the variable by MUNIC, PROVIN and DEPTO, one
  unbroken national run, and SEXO by MUNIC (all ages) for the population. Raw pages in
  `data/raw/bo/`.
- Labels: 79 categories. 36 native languages, Castellano, 37 foreign languages,
  "Afroboliviano", "Joaquiniano", "Otras declaraciones", "Otro idioma extranjero", "Lenguaje de
  señas", "Sin especificar". "No Aplica" is "does not speak" (the derived `IDIOMA_MAT` calls it
  "No habla"; 264,500 at all ages, mostly infants). Some labels are stored double-encoded
  (UTF-8 read as Latin-1, "KabineÃ±a", "ZudáÃ±ez"); `fix()` repairs them pair by pair.
- Not used: question 33 (languages spoken, by most use, up to three). A single-answer mother
  tongue item beats it. The 2012 base (`CPV2012COM`) asks the same two questions and was not
  queried; 2024 is the vintage.

## The universe, and why

The question was asked of everyone, but INE publishes it for **people aged 4 and over who
usually live in Bolivia** (question 36: "here, in this municipality" or "in another municipality
of the country"). INE's figures released in September 2025 (reported in eju.tv, La Época and El
Fulgor, "Censo 2024: El castellano, quechua y aymara predominan...") are not reproduced by any
age cut alone; with the residence condition every one of them comes out to the person. So the
map uses INE's universe, `PERSONA.EDAD >= 4 AND PERSONA.LUGRES <= 2`, and `how` says "aged 4 and
over". Under-1s are 67% "not specified" and 73% of all under-1s "do not speak", so the infants
would add mostly noise.

What is left out (`gap`): 661,467 children under 4 (5.8%); 98,649 aged 4+ who usually live
abroad (21,186) or did not say where they live (77,463); 21,443 who do not speak and 74,589 who
named no language. 10,509,185 + 96,032 + 98,649 + 661,467 = 11,365,333, the census count.

## Checks (all pass, printed by the script)

| check | result |
|---|---|
| areas | 9 departments, 113 provinces (the 112 plus the TIM, code 0809), 343 municipality areas |
| labels sum to each area's Total | all 343 |
| municipalities rebuild the PROVIN break / the DEPTO break | 9,153 and 729 cells, 0 failures |
| municipalities sum to the MUNIC break's RESUMEN, and to an unbroken national run | 81 cells each, 0 failures |
| INE's published national figures for this universe | 24 of 24 exact |
| municipal populations, all ages (SEXO by MUNIC) | sum to 11,365,333 |
| each municipality's universe is within its population | all 343 |

The 24 published figures: Castellano 8,141,600, Quechua 1,395,229, Aimara 774,874, Guaraní
43,870, Tsimane' 16,556, Weenhayek 4,515, Mojeño Trinitario 1,835, Canichana 9, Moré 9,
Cayubaba 7, Machaj Juyay Kallawaya 7, Joaquiniano 8, Guarasu'we 1, Baure 11, Machineri 11,
Pacahuara 23, Tapiete 59, Leco 62, Puquina 94, Yaminawa 118, Yuqui 246, sign language 1,822,
not specified 74,589, no language 21,443. This is the check independent of the server's own
arithmetic.

## Geography (sources/bo_geo.py)

religiondots draws Bolivia from a survey at department level, so its hexes are keyed to 9
departments; this map needs municipalities. Units are COD-AB's adm3 (OCHA cod-ab-bol v02,
valid from 2024-09-16), read in place from `../religiondots/data/raw/bo/shp/bol_admin3.shp`.
The Kontur extract is religiondots' download (`../religiondots/data/raw/bo/...gpkg.gz`), copied
into `data/geo/kontur/` so `_grid.hex_layer` finds it. Nothing written into religiondots.

- **Join on code.** COD's p-codes are INE's six-digit codes with "BO" in front. All 339 COD
  municipalities are census areas. Ten names differ on a matching code, all the same place (long
  autonomy names like "Charagua (Autonomía Guaraní Charagua Iyambae)", "Sopachui"/"Sopachuy",
  "Pampa Grande"/"Pampagrande"); the script prints them.
- **Four census areas have no polygon**, each folded into the municipality it was carved from
  (`data/geo/bo/bo_lookup.csv`): TIOC-Raqaypampa (8,056 people) into Mizque, of which it was the
  fifth district; San Pedro de Macha (21,011) into Colquechaca; TIOC-Jatun Ayllu Yura (5,819)
  into Tomave; TIOC-Territorio Indígena Multiétnico (3,973) into San Ignacio de Moxos. Ley 1497
  of 2023 carved the TIM from San Ignacio de Moxos and Santa Ana de Yacuma with no split
  published; religiondots made the same call. Kontur/census for the four targets after folding:
  0.78, 1.29, 1.56, 1.20, inside the national spread.
- **Placement checks** (`_grid.hex_layer`): 548 hexes (47,717 people) outside every unit, the
  border overrun. Kontur/census nationally 1.090; per unit p10 0.70, median 1.21, p90 1.66; 8 of
  339 outside a factor of 3. Lowest are the five tiny municipalities of Oruro's province 0405
  (Huachacalla, Escara, Cruz de Machacamarca, Yunguyo del Litoral, Esmeralda; 0.09-0.20), where
  Kontur sees almost no one on the altiplano; their dots crowd into the few hexes it does see.
  Highest are Pando's (up to 3.9), Kontur overcounting the Amazon. Log correlation r = 0.938
  against a best of 0.162 over 500 shuffles. Every unit has populated hexes. No Kontur cap block
  stopped the scatter.

## Mapping calls (taxonomy/bo2024.py, taxonomy/tree.d/bo.txt)

- Every printed language label has its own node. Families from Glottolog, checked per language.
- **New roots:** Mosetenan (Tsimane', Mosetén), Uru-Chipaya, Zamucoan (Ayoreo). Glottolog makes
  Mosetén-Chimané one isolate with dialects; the census names two languages and most readers
  know a two-language family, so it is a root with two leaves.
- **Guaraní → Ava Guaraní (Chiriguano)** (ar.txt's node): Bolivia's Guaraní is Eastern Bolivian
  Guaraní (Glottolog east2555), the Ava, Isoseño and Simba varieties. Not the Guarani group node,
  which the viewer would wash out as "not named".
- **Bésiro → Chiquitano** (br.txt's node; Bésiro is the language's own name). **Zamuco →
  Ayoreo**: the constitution's "zamuco" is the living Ayoreo; Glottolog's Zamuco is the extinct
  Jesuit-era language. **Moré → Itene**, **Maropa → Reyesano**, **Guarasu'we → Pauserna**,
  **Machineri → Manchineri** (br.txt): the census uses the people's names.
- **Joaquiniano**, a Baure dialect in Glottolog, is a sibling of Baure so Baure stays a leaf.
  **Valenciano** likewise beside Catalan. **Afroboliviano** is Afro-Bolivian Spanish, a node
  beside Spanish under Romance (whether it is a creole is contested).
- **Kallawaya** (Glottolog Callawalla, a speech register: Quechua grammar, mostly Puquina words)
  sits under Quechuan by its grammar. 7 people.
- **Alemán → German.** 75,852, mostly in the Santa Cruz Mennonite colonies (Pailón 19,556, San
  José de Chiquitos 12,966, Cabezas 8,336, Charagua 7,355, Cuatro Cañadas 5,504), whose home
  language is Plautdietsch. The census label is German, so the dots are German; note_public
  says so.
- **Chino → the Chinese group** (`sinotibetan.sinitic`), as pl2021 and ca2021 do; **Taiwanés →
  Min Nan**. **Suizo → `indoeuropean`**, as au2021 does with "Swiss": it names no language.
- **Lenguaje de señas → `signlanguage`**: the label does not name which sign language.
- **"Otras declaraciones" and "Otro idioma extranjero" → `other`.** Neither is filed as
  indigenous, so there is no indigenous remainder to keep apart.

## Colours

Hand-picked: Mosetenan (Tsimane' deep cyan-blue, Mosetén paler), apart from Mojeño's greens and
the Tacanan pinks around San Borja; Mojeño group with Trinitario and Ignaciano as two greens;
Uru-Chipaya yellow-green inside Aymara's lavender; Ayoreo magenta in the Chiquitanía. Reused
colours left alone. **One near-clash not fixed:** Guarayu (ar.txt, #bdac4a) and Chiquitano
(br.txt, #b9ac5f) are almost the same olive, and Guarayos borders Lomerío and Concepción. Both
nodes belong to other fragments, so this is left for a colour pass.

## Not done

- 2012 census (same question, `CPV2012COM`) as a second witness for the municipal pattern.
- Plautdietsch: nothing in the census separates it from German.

## Plautdietsch (2026-10-06, session 5d7dac7e-br)

Reverses "Alemán → German" outside the cities: the census's "Alemán" is drawn as Plautdietsch in
every municipality except the nine department capitals and El Alto (`countries/bo.py`
`CITIES`). 72,976 of 75,852 people moved; German keeps 2,876 (Santa Cruz de la Sierra 2,344, La
Paz 277, Cochabamba 111...). The German-speaking municipalities are a list of colony places
(Pailón 19,556, San José de Chiquitos 12,966, Cabezas 8,336, Charagua 7,355, Cuatro Cañadas
5,504, San Ignacio 4,336, Yacuiba 1,961, San Julián, San Javier in Beni, Villa Montes, Ixiamas
in La Paz...), and Wikipedia's Mennonites in Bolivia gives the census's 75,852 as the colonies'
German-speaking population (against Mongabay's ~150,000 Mennonites in 2023: the census figure is
smaller, possibly because colonies answered for some members only; not adjusted). A
place-dependent split of one label (AGENT_BRIEF §3). Small rural German families outside the
colonies are drawn as Plautdietsch too; La Guardia (122, a Santa Cruz suburb) is the likeliest
case. 10,488 dots, 57 rings.
