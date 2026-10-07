# Chile: the record

Drawn 2026-10-04 (agent d9e44929-cl). 17,506,629 people aged 5+ drawn on 345 comunas; 510,404
speakers of an indigenous language (speak or understand), the rest drawn as Spanish.

## Source

INE, Censo de Población y Vivienda 2024, fourth results release (30 June 2025), workbook
`P3_Lenguas-indigenas.xlsx`, sheet 2 (re-issued 4 December 2025 for the renaming of Paihuano):
comuna x pertenencia a pueblo indígena (Total / indigenous / not indigenous / not declared) x
language. Open download, no login:
https://censo2024.ine.gob.cl/wp-content/uploads/2025/06/P3_Lenguas-indigenas.xlsx. Population by
comuna and age from `D1_Poblacion-censada-por-sexo-y-edad-en-grupos-quinquenales.xlsx` (27 March
2025, sheet 4), the same file religiondots uses.

- The question (English questionnaire, `Cuestionario-Ingles_CPV2024.pdf`): P30 "Do you speak or
  understand one of the following indigenous languages?", for people aged 5 and over, everyone,
  indigenous or not. "Do not count people who understand only isolated words or greetings." "If a
  person speaks or understands more than one indigenous language, select the one they speak
  best." One answer: the ten columns sum exactly to the 5+ population in every comuna.
- Options: Mapuzugun, Aymara, Quechua, Rapa Nui, Ckunza, Kawésqar, Yagán, another indigenous
  language of Chile (no write-in), does not speak or understand any. Spanish and foreign languages
  are not asked. The 2017 census asked pueblo only (coverage sweep note).
- The coverage sweep had it as "indigenous_only" at comuna on the REDATAM server; the published
  workbook already has comuna, so REDATAM was not needed.

`python sources/cl_censo.py --fetch` downloads both workbooks to `data/raw/cl/` and writes
`data/normalized/cl.csv` (346 comunas x 10 columns) and `cl_units.csv`.

## Checks (numbers from the run)

1. 346 comunas; on every Total row the ten answer columns sum to the 5+ population.
2. Per comuna and column, the three pertenencia rows sum to the Total row.
3. Summed over comunas, every column equals the national row and all 16 regions on sheet 1.
4. Per comuna, D1's population less its "0 a 4" row equals P3's 5+ population exactly (a second
   table of the same census); nationally 18,480,432 people, 870,693 under 5, 17,609,739 asked.

National: Mapuzugun 381,762; Aymara 60,605; Quechua 39,430; Rapa Nui 7,006; Ckunza 2,616;
Kawésqar 656; Yagán 785; other 17,604; none 16,996,225; not declared 103,050. Of the speakers,
the share not identifying as indigenous: Mapuzugun 19%, Aymara 12%, Quechua 28%, Rapa Nui 43%,
Ckunza 20%, Kawésqar 52%, Yagán 69%, other 66%.

## Calls

- **Spanish remainder** (spec §3.5): "does not speak or understand any" is drawn as Spanish, tier
  `derived`. Hides Haitian Creole and other immigrant languages. Not corroborated: no second
  source asking everyone's home or first language was found (not searched hard; wanted, not
  required). The limitation is said in note_public.
- **Under-5s not drawn** (870,693, 4.7%), as Mexico leaves out its under-3s: the census does not
  ask them, and drawing them as Spanish would be an invention. In `gap`.
- **"Speaks or understands best"** is not a first-language question; drawn as measured, said in
  `how` and note_public. Ckunza and Yagán have no native speakers; their counts (and Kawésqar's,
  half of whom are not Kawésqar) are revival, heritage and learners, drawn as recorded, like
  Argentina's Diaguita and Huarpe.
- **Alto Biobío (08314)**: "another indigenous language of Chile" is 1,330 people there, 23.5% of
  the comuna's 5+ and 1,290 of them indigenous, against at most 0.61% in any other comuna (Queilén).
  Alto Biobío is the Pewenche upper Biobío; 2,460 there ticked Mapuzungun. The remainder there is
  read as Pewenche (Chedungun), a Mapudungun dialect in Glottolog, and sits on the group node
  `araucanian` (drawn washed out as an unnamed Araucanian language), not on the named Mapuche
  node. A place-dependent split under §3's rule; elsewhere `americas_other`. Small clusters in
  Chiloé (Ancud 182, Quellón 128) are probably Williche/Veliche talk but too few to split.
- **Quechua** on the shared `quechuan.quechua`, as Peru and Bolivia; no variety named.
- **Kawésqar** is a new root `kawesqar` (Glottolog family kawe1237, Alacalufan) with one language
  node; mid blue (0.72 0.12 245). Its 656 speakers fall under one dot at 1:1,000, so it draws no
  dot today.
- **Antártica (12202)**, 60 people, has no polygon in religiondots' layer and is dropped, as there.
- **Geography**: religiondots' `cl_hexes.gpkg` (Kontur, 345 comunas, unit = CUT code), read-only;
  counts() asserts every drawn comuna has a polygon. Population weight inside each comuna.
- **Colours**: nothing re-picked. Mapuche green against Spanish red-orange is the main edge; the
  northern Aymara lavender / Quechua peach / Spanish set is Peru's and Bolivia's.

## Scatter

17,502 dots at 1:1,000, 2 rings; 4,629 people (Kawésqar and the smallest per-language remainders)
under one dot nationally.

## Immigrant languages (2026-10-05, session edd42a8c-lats)

Before this, everyone not an indigenous speaker was Spanish, Haitians included.

- **Source**: INE Censo 2024, second release, `D4_Inmigracion-Internacional.xlsx` sheet 4
  (`data/raw/cl/`), comuna x birthplace in 13 categories: Argentina, Bolivia, Colombia, Haiti,
  Peru, Venezuela, and continent groups. The public microdata carry the same 13 (dictionary,
  `p25_lug_nacimiento_esp`), so nothing finer exists for 2024. `sources/cl_immig.py` ->
  `data/normalized/cl_immig.csv`. Checks: 346 comunas, categories sum to each comuna's total,
  comunas sum to the national rows (1,608,650), same comunas as cl.csv.
- **Groups split** at the 2017 census's national country counts (INE 2018, "Características de
  la inmigración internacional en Chile, Censo 2017", Tabla 5, copy in `data/raw/cl/`): other
  South America = Ecuador 27,692 / Brazil 14,227 / Uruguay 5,172 / Paraguay 4,492; Central
  America and Caribbean = Dominican Rep. / Cuba / Mexico (INE's codes are UN M49, so Mexico is
  there); Northern America = US; Europe = Spain 16,675 / Germany 5,736 / France 5,447 / Italy
  4,097; Asia = China (the only Asian country Tabla 5 names; overstates Chinese); Oceania =
  Australia; Africa (1,580) no country named, left on Spanish.
- **Aged 5+**: foreign-born scaled by each comuna's own 5+ share (1,529,810 of 1,608,650).
- **Languages and retention** as Argentina (`sources/latam_immig.py`): `origin_mix.mix(iso,
  "cl")`, Spanish shares stay Spanish, the rest kept at TeO2's regional rate (Americas 77.4%).
  Venezuelans now Spanish everywhere in South America (new uncited override in origin_mix, §2b:
  their home mix had 1.1% Wayuu).
- **Double counting**: the census asked everyone 5+, immigrants included, about Quechua, Aymara
  and Mapuzugun, so those are measured already; only an estimate above the comuna's measured
  count is added (39,676 of the estimate dropped).
- **Result**: estimate 164,475, added 124,799, out of the Spanish remainder: Haitian Creole
  59,480, Portuguese 16,165, Quechua +12,088 (now 51,518), Mandarin 8,222 (+ other Sinitic),
  English 8,217, German 3,843, French 3,644, Paraguayan Guarani 3,174, Italian 2,048. Spanish
  16.99M -> 16.87M. 40 nodes; `tree.d/cl.txt` borrowed block regenerated. Scatter 17,493 dots,
  22 rings.
- **Not drawn**: German of the south's 19th-century settlers (Valdivia, Osorno, Llanquihue).
  Only unsourced estimates exist (Wikipedia's "20,000-40,000 native speakers"); a survey would
  be needed. Chilean-born children of immigrants are Spanish.
